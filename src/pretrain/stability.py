"""Periodic, fixed-input measurements of conditioning and update sensitivity.

Only rank zero owns this observer. It runs outside DDP and autograd, before
and after an already-computed training update. Four retained examples at
three timesteps form a small diagnostic panel, not a validation dataset.
The panel is saved so resumed runs and experiment arms can reuse it exactly.
No gradient, optimizer state, training RNG, or model parameter is modified.
"""
from __future__ import annotations

import contextlib
from functools import partial
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from ..flow.paths import shift_timesteps


def rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean().sqrt()


def record_metrics(run_dir, step, metrics) -> None:
    """Durable scalar history independent of cloud logging availability."""
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "training_metrics.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(step=step, metrics=metrics), allow_nan=False) + "\n")


def feature_metrics(value: torch.Tensor, rows: int) -> dict[str, torch.Tensor]:
    """Separate variation across captions at fixed t from variation across t."""
    panel = value.float().reshape(3, rows, -1)
    scale = rms(panel)
    return {
        "rms": scale,
        "caption_ratio": rms(panel - panel.mean(1, keepdim=True)) / scale.clamp_min(1e-12),
        "time_ratio": rms(panel - panel.mean(0, keepdim=True)) / scale.clamp_min(1e-12),
    }


def update_metrics(old: torch.Tensor, new: torch.Tensor, decay: float) -> dict[str, torch.Tensor]:
    """Decompose actual squared-norm change into decay, radial step and energy.

    With new = a*old + u, the three fractional contributions sum to
    (||new||² - ||old||²) / ||old||². This makes no random-walk assumption.
    ``decay`` is the applied LR * weight_decay, zero for a skipped update.
    """
    old, new = old.float(), new.detach().float()
    delta = new - old
    a = 1.0 - decay
    u = delta + decay * old
    weight_sq = old.square().sum()
    denom = weight_sq.clamp_min(1e-24)
    return {
        "weight_rms": rms(new),
        "update_rms": rms(delta),
        "update_ratio": delta.square().sum().sqrt() / denom.sqrt(),
        "radial": 2 * a * (old * u).sum() / denom,
        "energy": u.square().sum() / denom,
        "decay": (a * a - 1) * weight_sq / denom,
        "norm_sq_change": (new.square().sum() - weight_sq) / denom,
    }


@contextlib.contextmanager
def eager_observation(model):
    """Temporarily bypass per-block compile wrappers for the small probe."""
    forwards = []
    modes = [(module, module.training) for module in model.modules()]
    try:
        for module, _ in modes:
            original = getattr(module.forward, "_torchdynamo_orig_callable", None)
            if original is not None:
                forwards.append((module, module.forward))
                module.forward = original
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, forward in forwards:
            module.forward = forward
        for module, training in modes:
            module.training = training


class StabilityMonitor:
    """One cadence, a fixed panel, and measurements tied to the spike evidence."""

    def __init__(self, model, optimizers, run_dir, autocast=contextlib.nullcontext):
        self.model = model
        self.autocast = autocast
        self.run_dir = Path(run_dir)
        self.panel_path = self.run_dir / "stability_inputs.pt"
        self.panel = None
        self.panel_id = None
        if self.panel_path.exists():
            self._load_panel()
        self.selected = {}
        groups = {id(p): group for opt in optimizers for group in opt.param_groups
                  for p in group["params"]}
        # Sample the beginning, middle and end. These are explicitly sampled
        # matrix statistics, not an estimate of the entire optimizer group.
        indices = {0, len(model.blocks) // 2, len(model.blocks) - 1}
        for name, param in model.named_parameters():
            if param.ndim != 2:
                continue
            conditioner = name.startswith(("c_mlp.", "txt_pooled_proj."))
            block = name.startswith("blocks.") and int(name.split(".")[1]) in indices
            projection = any(key in name for key in (".proj", ".down_proj", ".modulation"))
            if conditioner or (block and projection):
                self.selected[name] = (param, groups[id(param)])
        self.before = None

    def _load_panel(self):
        device = next(self.model.parameters()).device
        self.panel = {key: value.to(device) if torch.is_tensor(value) else value
                      for key, value in torch.load(self.panel_path, map_location="cpu",
                                                   weights_only=True).items()}
        self.panel_id = hashlib.sha256(self.panel_path.read_bytes()).hexdigest()[:16]

    def ensure_panel(self, z0, z1, txt, txt_pooled, txt_mask):
        if self.panel is not None:
            return
        rows = min(4, z0.shape[0])
        panel = {key: value[:rows].detach().cpu().clone() if value is not None else None
                 for key, value in dict(z0=z0, z1=z1, txt=txt, txt_pooled=txt_pooled,
                                        txt_mask=txt_mask).items()}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(panel, self.panel_path)
        self._load_panel()

    def _inputs(self):
        p = self.panel
        rows = p["z0"].shape[0]
        values = {key: value.repeat((3,) + (1,) * (value.ndim - 1))
                  if value is not None else None for key, value in p.items()}
        t = p["z0"].new_tensor([0.1, 0.5, 0.9], dtype=torch.float32).repeat_interleave(rows)
        t = shift_timesteps(t, p["z0"], patch_size=self.model.patch_size)
        zt = ((1 - t[:, None, None, None]) * values["z0"]
              + t[:, None, None, None] * values["z1"]).to(p["z0"].dtype)
        kwargs = dict(txt=values["txt"], txt_pooled=values["txt_pooled"],
                      txt_mask=values["txt_mask"])
        return zt, t, kwargs, (values["z1"] - values["z0"]).float(), rows

    def _forward(self, replace_condition=None, inspect=False):
        zt, t, kwargs, target, rows = self._inputs()
        handles, values, features = [], {}, {}
        block_inputs = {}

        def capture_c(module, args, output):
            features["condition"] = output.detach().clone()
            if replace_condition is not None:
                return replace_condition.to(output)

        def capture_hidden(module, args, output):
            features["hidden"] = args[0].detach().clone()

        def capture_preactivation(module, args, output):
            values["conditioning/negative_tail_fraction"] = (output.detach() < -5).float().mean()

        def block_start(index, module, args):
            block_inputs[index] = args[0].detach()
            values[f"blocks/{index:02d}/input_rms"] = rms(args[0])

        def after_attention(index, module, args):
            old = block_inputs[index]
            # norm2 sees the residual after the gated attention add. Observe
            # image tokens only, so padded text cannot bias these statistics.
            new = args[0][:, :old.shape[1]].detach()
            values[f"blocks/{index:02d}/attention_ratio"] = rms(new.float() - old.float()) / rms(old).clamp_min(1e-12)
            block_inputs[index] = new

        def block_end(index, module, args, output):
            old = block_inputs.pop(index)
            new = output[0].detach()
            values[f"blocks/{index:02d}/mlp_ratio"] = rms(new.float() - old.float()) / rms(old).clamp_min(1e-12)
            values[f"blocks/{index:02d}/output_rms"] = rms(new)

        def modulation(name, module, args, output):
            chunks = output.detach().float().split(self.model.hidden_size, dim=-1)
            for offset, label in enumerate(("shift", "scale", "gate")):
                value = torch.cat(chunks[offset::3], dim=-1)
                values[f"{name}/{label}_rms"] = rms(value)

        try:
            handles.append(self.model.c_mlp.register_forward_hook(capture_c))
            if inspect:
                handles.append(self.model.c_mlp[-1].register_forward_hook(capture_hidden))
                handles.append(self.model.c_mlp[0].register_forward_hook(capture_preactivation))
                for index, block in enumerate(self.model.blocks):
                    handles.append(block.register_forward_pre_hook(partial(block_start, index)))
                    norm2 = getattr(block, "norm2_img", None)
                    if norm2 is None:
                        norm2 = block.norm2
                    handles.append(norm2.register_forward_pre_hook(partial(after_attention, index)))
                    handles.append(block.register_forward_hook(partial(block_end, index)))
                    for name, module in block.named_children():
                        if name.startswith("modulation"):
                            handles.append(module.register_forward_hook(
                                partial(modulation, f"blocks/{index:02d}/{name}")))
            with self.autocast():
                prediction = self.model(zt, t, **kwargs).detach().float()
            losses = (prediction - target).square().flatten(1).mean(1).reshape(3, rows).mean(1)
            if inspect:
                for name, value in features.items():
                    values.update({f"conditioning/{name}_{key}": metric
                                   for key, metric in feature_metrics(value, rows).items()})
                values["panel/rows_per_timestep"] = losses.new_tensor(rows)
                values["panel/text_tokens"] = losses.new_tensor(kwargs["txt"].shape[1])
            return prediction, features, losses, values
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def before_update(self):
        if self.before is not None:
            raise RuntimeError("stability observation already in progress")
        started = time.monotonic()
        with eager_observation(self.model):
            prediction, features, losses, values = self._forward(inspect=True)
        gains = [module.weight.detach().float().abs().max()
                 for name, module in self.model.named_modules()
                 if (".norm_msa" in name or ".norm_mlp" in name)
                 and isinstance(module, torch.nn.RMSNorm)]
        if gains:
            values["summary/branch_norm_gain_max"] = torch.stack(gains).max()
        snapshots = {}
        for name, (param, group) in self.selected.items():
            snapshots[name] = (param.detach().float().clone(),
                               float(group["lr"]) * float(group.get("weight_decay", 0))
                               if param.grad is not None else 0.0)
            if param.grad is not None:
                values[f"weights/{name}/grad_rms"] = rms(param.grad)
        # The small conditioner snapshot also enables an exact attribution of
        # its last linear layer's bias-like shift, on frozen hidden features.
        last = self.model.c_mlp[-1]
        self.before = (features, losses, values, snapshots,
                       last.bias.detach().float().clone(), prediction)
        self.before_seconds = time.monotonic() - started

    @torch.no_grad()
    def after_update(self, *, step: int, applied: bool):
        started = time.monotonic()
        features, old_losses, values, snapshots, old_bias, old_prediction = self.before
        # Clear ownership even if the probe raises, rather than retaining a
        # whole set of snapshots into another training forward.
        self.before = None
        for name, (old, decay) in snapshots.items():
            param = self.selected[name][0]
            values.update({f"weights/{name}/{key}": value for key, value in
                           update_metrics(old, param, decay if applied else 0).items()})
        last_name = f"c_mlp.{len(self.model.c_mlp) - 1}.weight"
        delta_weight = self.model.c_mlp[-1].weight.float() - snapshots[last_name][0]
        delta_bias = self.model.c_mlp[-1].bias.float() - old_bias
        shared = F.linear(features["hidden"].float().mean(0), delta_weight)
        values["conditioning/c2_shared_shift_rms"] = rms(shared)
        values["conditioning/c2_bias_shift_rms"] = rms(delta_bias)
        del snapshots, delta_weight

        with eager_observation(self.model):
            prediction, after_features, losses, _ = self._forward()
            old_c = features["condition"]
            delta_c = after_features["condition"].float() - old_c.float()
            held_prediction, _, held_losses, _ = self._forward(replace_condition=old_c)
            _, _, twice_losses, _ = self._forward(replace_condition=old_c.float() + 2 * delta_c)

        values["conditioning/update_rms"] = rms(delta_c)
        values["conditioning/shared_update_rms"] = rms(delta_c.mean(0))
        values["conditioning/centered_update_rms"] = rms(delta_c - delta_c.mean(0))
        values["response/prediction_change_rms"] = rms(prediction - old_prediction)
        values["response/conditioning_prediction_change_rms"] = rms(prediction - held_prediction)
        values["response/conditioning_gain"] = rms(prediction - held_prediction) / rms(delta_c).clamp_min(1e-12)
        for index, label in enumerate(("t010", "t050", "t090")):
            for name, tensor in (("before", old_losses), ("after", losses),
                                 ("held_condition", held_losses), ("twice_condition_update", twice_losses)):
                values[f"response/{label}/loss_{name}"] = tensor[index]
            values[f"response/{label}/conditioning_loss_delta"] = losses[index] - held_losses[index]
            values[f"response/{label}/second_difference"] = twice_losses[index] - 2 * losses[index] + held_losses[index]
        values["update_applied"] = losses.new_tensor(float(applied))
        for suffix, summary in (("/gate_rms", "gate_rms_max"),
                                ("/output_rms", "residual_rms_max"),
                                ("/attention_ratio", "attention_ratio_max"),
                                ("/mlp_ratio", "mlp_ratio_max")):
            observed = [value for key, value in values.items()
                        if key.startswith("blocks/") and key.endswith(suffix)]
            if observed:
                values[f"summary/{summary}"] = torch.stack(observed).max()
        for role, pattern in (("conditioning", "c_mlp."), ("attention", ".attn.proj"),
                              ("ffn", ".down_proj"), ("modulation", ".modulation")):
            for statistic in ("weight_rms", "update_ratio", "radial", "energy", "decay", "norm_sq_change"):
                observed = [value for key, value in values.items()
                            if key.startswith("weights/") and pattern in key
                            and key.endswith("/" + statistic)]
                if observed:
                    values[f"sampled_weights/{role}/{statistic}_mean"] = torch.stack(observed).mean()
        # One host transfer for scalar metrics instead of a sync per hook.
        names = list(values)
        scalars = torch.stack([values[name].float().reshape(()) for name in names]).cpu().tolist()
        metrics = {f"stability/{name}": value for name, value in zip(names, scalars)}
        metrics["stability/probe_seconds"] = self.before_seconds + time.monotonic() - started
        with (self.run_dir / "stability.jsonl").open("a") as fh:
            fh.write(json.dumps(dict(step=step, panel_id=self.panel_id, scope="rank0_fixed_panel",
                                     metrics=metrics), allow_nan=False) + "\n")
        # Keep per-layer/tensor detail in JSONL; publish a compact dashboard.
        return {key: value for key, value in metrics.items()
                if not key.startswith(("stability/blocks/", "stability/weights/"))}

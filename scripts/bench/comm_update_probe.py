"""One-GPU numerical diagnostic for lossy gradient communication.

Capture distinct synthetic DiT backward gradients at the same weights, once.
Compare FP32 and BF16 pre-divided sequential sums of those identical captures,
then apply the actual clipping and fresh Muon/AdamW optimizers to both. This is
NOT NCCL-order emulation, an evolving training trajectory, quality acceptance,
or a distributed throughput measurement. Production communication is untouched.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.bench.ddp_compile_probe import make_inputs
from src.models.artflow import ArtFlow
from src.train.config import load_config
from src.train.muon import build_param_groups


@torch.no_grad()
def sequential_mean(values, dtype, device):
    """Cast, pre-divide each rank, then sequentially sum in the wire dtype.

    Addition order is explicit but not asserted to match a NCCL algorithm.
    Never mutate the common CPU gradient captures.
    """
    if not values:
        raise ValueError("need at least one rank gradient")
    result = torch.zeros_like(values[0], dtype=dtype, device=device)
    for value in values:
        result.add_(value.to(device=device, dtype=dtype, copy=True).div_(len(values)))
    return result.float()


@torch.no_grad()
def tensor_error(actual, reference):
    # FP32 elementwise arithmetic; FP64 sum avoids tiny-gradient underflow in
    # the accumulator without allocating full-size FP64 parameter copies.
    delta = actual.float() - reference.float()
    return dict(finite=bool(torch.isfinite(actual).all() & torch.isfinite(reference).all()),
                exact=torch.equal(actual, reference),
                max_abs=float(delta.abs().max()),
                error_sq=float(delta.square().sum(dtype=torch.float64)),
                reference_sq=float(reference.float().square().sum(dtype=torch.float64)))


def summarize(rows):
    error_sq = sum(row["error_sq"] for row in rows)
    reference_sq = sum(row["reference_sq"] for row in rows)
    return dict(finite=all(row["finite"] for row in rows),
                exact=all(row["exact"] for row in rows),
                max_abs=max((row["max_abs"] for row in rows), default=0.0),
                error_l2=math.sqrt(error_sq), reference_l2=math.sqrt(reference_sq),
                relative_l2=(math.sqrt(error_sq / reference_sq) if reference_sq
                             else (0.0 if error_sq == 0 else None)))


def optimizers_for(model, config):
    return build_param_groups(
        model, muon_lr=config.optim.muon_lr, muon_wd=config.optim.muon_wd,
        adam_lr=config.optim.learning_rate, muon_momentum=config.optim.muon_momentum,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Model safetensors from a completed training smoke checkpoint")
    parser.add_argument("--micro", type=int, default=64)
    parser.add_argument("--forward-micro", type=int, default=8,
                        help="Bound eager activation memory while accumulating --micro examples")
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--text", type=int, default=128)
    parser.add_argument("--simulated-ranks", type=int, nargs="+", default=[2, 8])
    args = parser.parse_args()
    if min(args.micro, args.forward_micro, args.latent, args.text, *args.simulated_ranks) < 1:
        parser.error("shapes and rank counts must be positive")
    if args.micro % args.forward_micro:
        parser.error("micro must be divisible by forward-micro")
    if not torch.cuda.is_available():
        parser.error("the DiT probe requires one CUDA GPU")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = load_config(args.config)
    model = ArtFlow(**asdict(config.model)).cuda().train()
    # Scratch AdaLN-zero gates would leave most hidden gradients zero. Load
    # actual nonzero smoke-run weights instead; optimizer state is still fresh.
    from safetensors.torch import load_file
    checkpoint_state = load_file(str(args.checkpoint), device="cpu")
    model.load_state_dict(checkpoint_state, strict=True)
    del checkpoint_state
    with args.checkpoint.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    initial = {name: p.detach().cpu() for name, p in model.named_parameters()}
    captures, losses = [], []
    shape = SimpleNamespace(micro=args.micro, latent=args.latent, text=[args.text])
    for rank in range(max(args.simulated_ranks)):
        torch.manual_seed(1234 + rank)
        model.zero_grad(set_to_none=True)
        full_inputs = make_inputs(model, shape, "cuda")[0]
        loss_value = 0.0
        accumulation = args.micro // args.forward_micro
        for offset in range(0, args.micro, args.forward_micro):
            x, target, text, pooled, mask, t = (
                value[offset:offset + args.forward_micro] for value in full_inputs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(x, t, txt=text, txt_pooled=pooled, txt_mask=mask,
                               fast_attn=True, native_flash_varlen=True)
                loss = (output.float() - target.float()).square().mean() / accumulation
            loss.backward()
            loss_value += float(loss.detach())
            del output, loss
        gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters()
                     if p.grad is not None}
        if captures and gradients.keys() != captures[0].keys():
            raise RuntimeError("different gradient parameter sets across captures")
        if not all(bool(torch.isfinite(g).all()) for g in gradients.values()):
            raise RuntimeError("nonfinite raw gradients")
        captures.append(gradients)
        losses.append(loss_value)
        print(json.dumps(dict(captured_rank=rank, loss=losses[-1],
                              gradient_parameters=len(gradients))), flush=True)
        del x, target, text, pooled, mask, t, full_inputs
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    results = []
    for ranks in sorted(set(args.simulated_ranks)):
        reference_gradients, reference_updates = {}, {}
        variants = {}
        comparisons = {"gradients": {}, "updates": {}}
        routing = {}
        for variant, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.copy_(initial[name])
                    p.grad = None
            optimizers = optimizers_for(model, config)
            owner = {id(p): type(opt).__name__ for opt in optimizers
                     for group in opt.param_groups for p in group["params"]}
            for name, p in model.named_parameters():
                routing[name] = owner[id(p)]
                if name not in captures[0]:
                    continue
                p.grad = sequential_mean([capture[name] for capture in captures[:ranks]],
                                         dtype, p.device)
                if variant == "fp32":
                    reference_gradients[name] = p.grad.cpu()
                else:
                    comparisons["gradients"][name] = tensor_error(
                        p.grad, reference_gradients[name].to(p.device))
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optim.max_grad_norm, error_if_nonfinite=True)
            for opt in optimizers:
                opt.step()
            for name, p in model.named_parameters():
                update = p.detach() - initial[name].to(p.device)
                if not bool(torch.isfinite(update).all()):
                    raise RuntimeError("nonfinite optimizer update")
                if variant == "fp32":
                    reference_updates[name] = update.cpu()
                else:
                    comparisons["updates"][name] = tensor_error(
                        update, reference_updates[name].to(p.device))
            variants[variant] = dict(grad_norm_before_clip=float(grad_norm),
                                     max_grad_norm=config.optim.max_grad_norm)
            # Optimizer state must start fresh for the other branch, and must
            # be freed before the next allocation (loop variables also own it).
            del opt, optimizers, update
            model.zero_grad(set_to_none=True)
        summaries = {kind: {group: summarize([row for name, row in rows.items()
                                             if group == "all" or routing[name] == group])
                           for group in ("all", "Muon", "AdamW")}
                     for kind, rows in comparisons.items()}
        if not all(value["finite"] for groups in summaries.values() for value in groups.values()):
            raise RuntimeError("nonfinite comparison")
        result = dict(simulated_ranks=ranks, variants=variants, summaries=summaries,
                      per_parameter=comparisons)
        (args.out / f"simulated-{ranks}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(simulated_ranks=ranks, variants=variants,
                              summaries=summaries)), flush=True)
        results.append(dict(simulated_ranks=ranks, variants=variants, summaries=summaries))
    metadata = dict(torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                    physical_gpus=1, parameters=sum(p.numel() for p in model.parameters()),
                    model=asdict(config.model), optim=asdict(config.optim),
                    micro=args.micro, forward_micro=args.forward_micro,
                    latent=args.latent, text=args.text, losses=losses,
                    checkpoint_sha256=checkpoint_sha256,
                    checkpoint_name=args.checkpoint.name,
                    synthetic_inputs=True, smoke_checkpoint_weights=True,
                    shared_captured_gradients=True, fresh_optimizer_base_lr=True,
                    sequential_sum_not_nccl_order=True, component_only=True,
                    training_quality_established=False, production_hook_changed=False,
                    results=results)
    (args.out / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

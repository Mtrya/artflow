"""Ascend pretraining from one explicit config for the full resolution curriculum.

Model architecture and measured execution mechanisms are native code. The CLI
selects a stage, resumes a complete checkpoint, or enables diagnostics.
"""

import os
import gc
import argparse
import contextlib
import functools
import json
import re
import time
import warnings
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

from ..models.artflow import ArtFlow
from ..dataset.sampler import (
    LenBucket,
    BucketPlan,
    RowLengthQueueBatchSampler,
    RowDescriptorDataset,
    row_length_collate_fn,
    pad_text_to_hi,
)
from ..dataset.captions import CaptionPolicy
from .caption_telemetry import CaptionTelemetry, PolicyState
from .caption_loss_weights import CaptionLossWeights, StepLossAccumulator, weighted_mean
from .finite_guard import require_finite_update
from .update_ops import divide_gradients, ema_decay_at, update_ema
from .checkpoint_retention import prune_checkpoints
from .infra_metrics import InfraRecorder
from .stability import StabilityMonitor, record_metrics
from ..dataset.mix import parse_dataset_mix, get_dataset_weights
from ..utils.encode_text import encode_text
from ..utils.vae_codec import get_vae_stats
from ..flow.paths import FlowMatchingOT, shift_timesteps
from ..evaluation.eval_loss import EvalLossProbe
from ..evaluation.prompt_grid import grid_due, run_prompt_grid_eval
from ..evaluation.kid_eval import run_kid_eval
from .config import load_config, flatten
from .stage_control import (
    CHECKPOINT_RECORD,
    stage_endpoint,
    validate_checkpoint,
    write_checkpoint_record,
    verify_restored_rng,
)
from .muon import build_param_groups as build_muon_param_groups
from .health import (
    ema_rel_distance,
    qk_gain_stats,
    snapshot_weights,
    update_weight_ratios,
)

# Suppress specific warning about RMSNorm dtype mismatch in mixed precision
warnings.filterwarnings("ignore", message="Mismatch dtype between input and weight")


@torch.no_grad()
def update_ema_model(
    ema_model: torch.nn.Module,
    current_model: torch.nn.Module,
    decay: float,
    *,
    foreach: bool = False,
) -> None:
    update_ema(ema_model, current_model, decay, foreach=foreach)


def encode_training_text(captions, text_encoder, tokenizer, *, exit_layer, bucket_hi):
    """Stop the frozen encoder at the selected layer and pad to the batch bucket."""
    txt, mask, pooled = encode_text(
        captions,
        text_encoder,
        tokenizer,
        pooling=True,
        exit_layer=exit_layer,
        exit_mode="stop_at_layer",
        fast_slice=True,
    )
    txt, mask = pad_text_to_hi(txt, mask, bucket_hi)
    return txt, mask, pooled


def set_caption_curriculum(
    sampler: RowLengthQueueBatchSampler,
    *,
    global_step: int,
    max_steps: int,
    curriculum_start: float,
    curriculum_end: float,
) -> None:
    """Use whole-run progress for new draws without changing queued captions."""
    progress = global_step / max(1, max_steps)
    sampler.set_stage(curriculum_start + (curriculum_end - curriculum_start) * progress)


def matching_sampler_sidecars(checkpoint: str, world_size: int) -> list[str]:
    """Only a complete, exact rank set can restore stride-sharded row order."""
    expected = [f"sampler_state_rank_{rank:05d}.pt" for rank in range(world_size)]
    present = {
        name
        for name in os.listdir(checkpoint)
        if re.fullmatch(r"sampler_state_rank_\d+\.pt", name)
        and os.path.isfile(os.path.join(checkpoint, name))
    }
    return (
        [os.path.join(checkpoint, name) for name in expected]
        if present == set(expected)
        else []
    )


def build_linear_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    min_learning_rate: float,
    base_learning_rate: float,
    start_learning_rate: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(num_warmup_steps, 0)
    total_steps = max(num_training_steps, warmup_steps + 1)
    if base_learning_rate <= 0:
        raise ValueError("Base learning rate must be positive for cosine scheduler")

    # Calculate ratios
    min_ratio = min(min_learning_rate, base_learning_rate) / base_learning_rate
    start_ratio = min(start_learning_rate, base_learning_rate) / base_learning_rate

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup from start_ratio to 1.0
            progress = current_step / max(1, warmup_steps)
            return start_ratio + (1.0 - start_ratio) * progress

        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def restore_scheduler_base_lrs(scheduler, base_lrs) -> None:
    """Apply configured rates before the first full-resume optimizer update."""
    scheduler.base_lrs = list(base_lrs)
    rates = scheduler.get_lr()
    for group, rate in zip(scheduler.optimizer.param_groups, rates):
        group["lr"] = rate
    scheduler._last_lr = rates


def _device_mem_allocated(device) -> float:
    """Current allocated memory in GiB for cuda/npu; 0.0 otherwise."""
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated(device) / 1024**3
    if device.type == "npu":
        return torch.npu.memory_allocated(device) / 1024**3
    return 0.0


def _device_peak_mem(device):
    """(peak_allocated_gib, peak_reserved_gib) for cuda/npu, else None."""
    if device.type == "cuda" and torch.cuda.is_available():
        return (
            torch.cuda.max_memory_allocated(device) / 1024**3,
            torch.cuda.max_memory_reserved(device) / 1024**3,
        )
    if device.type == "npu":
        return (
            torch.npu.max_memory_allocated(device) / 1024**3,
            torch.npu.max_memory_reserved(device) / 1024**3,
        )
    return None


def _device_reset_peak_mem(device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "npu":
        torch.npu.reset_peak_memory_stats(device)


def sample_logit_normal_timesteps(batch_size, device, mu=0.0, sigma=1.0):
    """Sample timesteps from a logit-normal distribution."""
    logit = mu + sigma * torch.randn(batch_size, device=device)
    t = torch.sigmoid(logit)
    return t


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ascend pretraining from one complete run config"
    )

    class OneConfig(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            if getattr(namespace, self.dest, None) is not None:
                parser.error("--config accepts exactly one file")
            setattr(namespace, self.dest, value)

    parser.add_argument(
        "--config",
        action=OneConfig,
        required=True,
        help="Complete run TOML; no overlays",
    )
    parser.add_argument("--stage", required=True, help="Stage name from the run config")
    parser.add_argument(
        "--resume",
        help="Restore the complete checkpoint, including optimizer, EMA, RNG and step",
    )
    parser.add_argument("--verify_resume_state", action="store_true")
    parser.add_argument("--step_breakdown", action="store_true")
    parser.add_argument("--cpu_wall_profile", action="store_true")
    parser.add_argument("--log_shapes", action="store_true")
    parser.add_argument("--infra_metrics", action="store_true")
    parser.add_argument("--trace_start", type=int, default=-1)
    parser.add_argument("--trace_steps", type=int, default=3)
    parser.add_argument("--record_identity", action="store_true")
    parser.add_argument(
        "--check_config",
        action="store_true",
        help="Validate and print the full resolved config without allocating devices",
    )
    return parser


def load_bucket_plan(spec: str, resolution_ids) -> BucketPlan:
    """Load a resolution-keyed bucket plan from a JSON file path or inline JSON."""
    if os.path.isfile(spec):
        with open(spec) as handle:
            raw = json.load(handle)
    else:
        try:
            raw = json.loads(spec)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "bucket_plan (config [data]) must be a JSON file or inline JSON object"
            ) from exc

    if not isinstance(raw, dict):
        raise ValueError("bucket_plan must be an object keyed by resolution ID")
    raw = raw.get("by_resolution", raw)
    if not isinstance(raw, dict):
        raise ValueError("bucket plan by_resolution must be a JSON object")

    by_resolution = {}
    for resolution_id, buckets in raw.items():
        if not isinstance(buckets, list) or not buckets:
            raise ValueError(f"bucket plan for resolution {resolution_id} is empty")
        normalized = []
        for bucket in buckets:
            if isinstance(bucket, dict):
                try:
                    normalized.append(
                        LenBucket(bucket["max_length"], bucket["batch_size"])
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"bucket for resolution {resolution_id} needs max_length and batch_size"
                    ) from exc
            elif isinstance(bucket, (list, tuple)) and len(bucket) == 2:
                normalized.append(LenBucket(bucket[0], bucket[1]))
            else:
                raise ValueError(
                    f"bucket for resolution {resolution_id} must be [max_length, batch_size]"
                )
        by_resolution[int(resolution_id)] = normalized

    resolution_ids = {int(resolution_id) for resolution_id in resolution_ids}
    missing = sorted(resolution_ids.difference(by_resolution))
    if missing:
        raise ValueError(f"bucket plan is missing metadata resolution IDs: {missing}")
    return BucketPlan(by_resolution)


def main():
    cli = parse_args().parse_args()
    config = load_config(cli.config)
    args = SimpleNamespace(
        **flatten(config, cli.stage),
        **{
            key: val for key, val in vars(cli).items() if key not in ("config", "stage")
        },
    )
    if cli.check_config:
        print(json.dumps(asdict(config), indent=2))
        return
    if args.verify_resume_state and not args.resume:
        raise ValueError("--verify_resume_state requires --resume")
    # Reject invalid stages/checkpoints before allocating the model or GPUs.
    resume_step = 0
    if args.resume and args.resume != "None":
        resume_step = validate_checkpoint(
            args.resume,
            max_steps=args.max_steps,
            stop_at_step=args.stop_at_step,
            require_record=True,
            use_ema=True,
            device_type="npu",
        )
    end_step = stage_endpoint(args.max_steps, args.stop_at_step, resume_step)
    if not args.stage_start <= resume_step <= end_step:
        raise ValueError(
            f"stage {cli.stage} requires a checkpoint at steps {args.stage_start}..{end_step}"
        )
    reset_sampler = resume_step == args.stage_start and args.stage_start > 0
    if args.resume:
        saved_config = Path(args.resume) / "run_config.json"
        if not saved_config.is_file() or json.loads(saved_config.read_text()) != asdict(
            config
        ):
            raise ValueError(
                "resume requires the same complete run config recorded in the checkpoint"
            )
    if not Path(args.bucket_plan).is_file():
        raise ValueError(
            f"stage bucket plan is missing: {args.bucket_plan}; qualify it before launch"
        )

    bucket_contents = json.loads(Path(args.bucket_plan).read_text())
    if args.resume and not reset_sampler:
        saved_plan = Path(args.resume) / "bucket_plan.json"
        if (
            not saved_plan.is_file()
            or json.loads(saved_plan.read_text()) != bucket_contents
        ):
            raise ValueError("same-stage resume requires the checkpoint's bucket plan")
    log_shapes = args.log_shapes

    # Micro-batches carry their own (resolution, length bucket) and local
    # sample count, so accumulation is explicit in the training loop: keep
    # Accelerate's factor at one micro-batch and track the optimizer-step
    # boundary there.
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps (config [train]) must be >= 1")

    # Accelerator
    project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs")
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        log_with="swanlab",
        project_config=project_config,
    )

    if accelerator.device.type != "npu":
        raise RuntimeError(
            "pretraining requires Ascend NPU; CPU/CUDA remain supported for model tests and inference"
        )
    device_api = torch.npu
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    run_dir = os.path.join(args.output_dir, args.run_name)
    runtime_path = os.path.join(run_dir, "runtime.json")

    # SwanLab run resumption: reuse the stored run id on crash-resume so curves
    # stay in one run instead of fragmenting across restarts.
    swanlab_kwargs = {"experiment_name": args.run_name}
    if args.resume and os.path.exists(runtime_path):
        try:
            with open(runtime_path) as f:
                stored_run_id = json.load(f).get("swanlab_run_id")
            if stored_run_id:
                swanlab_kwargs["id"] = stored_run_id
                swanlab_kwargs["resume"] = "must"
        except Exception as e:
            print(f"Could not read swanlab run id ({e}); starting a fresh run")

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.swanlab_project,
            config={
                "architecture": ArtFlow.ARCHITECTURE,
                "recipe": asdict(config),
                "stage": cli.stage,
                "active_bucket_plan": bucket_contents,
            },
            init_kwargs={"swanlab": swanlab_kwargs},
        )

    # Load Text Encoder (Frozen, on GPU)
    # Each rank loads its own copy on its assigned device for parallel text encoding
    accelerator.print(f"Loading Text Encoder: {args.text_encoder_path}")
    text_encoder = AutoModelForCausalLM.from_pretrained(
        args.text_encoder_path,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device},  # Pin to current rank's device
        local_files_only=True,
    )
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path)

    # Load VAE Stats
    accelerator.print(f"Loading VAE Stats from {args.vae_path}")
    vae_mean, vae_std = get_vae_stats(args.vae_path, device=accelerator.device)
    vae_mean, vae_std = vae_mean.to(torch.bfloat16), vae_std.to(torch.bfloat16)

    # Load Model
    accelerator.print("Initializing ArtFlow Model...")
    model = ArtFlow(**asdict(config.model))
    # Print model statistics
    accelerator.print(
        f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Optimizers: Muon (chunked orthogonalization) updates the 2D hidden
    # weights; an auxiliary AdamW updates every other parameter. An
    # AdamW-only variant was the historical comparison baseline and is no
    # longer an option.
    optimizers = build_muon_param_groups(
        model,
        muon_lr=args.muon_lr,
        muon_wd=args.muon_wd,
        adam_lr=args.learning_rate,
        adam_wd=args.adam_wd,
        adam_eps=args.adam_eps,
        adam_betas=tuple(args.adam_betas),
        muon_momentum=args.muon_momentum,
    )
    muon_params = sum(
        p.numel() for group in optimizers[0].param_groups for p in group["params"]
    )
    aux_params = sum(
        p.numel()
        for optimizer in optimizers[1:]
        for group in optimizer.param_groups
        for p in group["params"]
    )
    accelerator.print(
        f"Muon: {muon_params:,} params (lr={args.muon_lr}); "
        f"AdamW aux: {aux_params:,} params"
    )

    optimizer_base_lrs = [
        [float(param_group["lr"]) for param_group in optimizer.param_groups]
        for optimizer in optimizers
    ]
    schedulers = []
    for opt in optimizers:
        base_lr = opt.param_groups[0]["lr"]
        lr_ratio = base_lr / args.learning_rate
        schedulers.append(
            build_linear_cosine_scheduler(
                opt,
                num_warmup_steps=args.lr_warmup_steps,
                num_training_steps=args.max_steps,
                min_learning_rate=args.min_learning_rate * lr_ratio,
                base_learning_rate=base_lr,
                start_learning_rate=args.start_learning_rate * lr_ratio,
            )
        )

    ema_model = deepcopy(model)

    # The measured Ascend path runs eagerly. Keep an unwrapped reference for
    # EMA, checkpoints and internal probes after DDP wraps the live model.
    model_raw = model

    # Dataset (with optional mixing)
    accelerator.print(f"Parsing dataset mix: {args.dataset_mix}")
    dataset_entries = parse_dataset_mix(args.dataset_mix)
    dataset_weights = get_dataset_weights(dataset_entries)
    if not args.bucket_plan:
        raise ValueError(
            "a non-empty [data] bucket_plan is required: every dataset row is "
            "assigned to a (resolution, caption-length) bucket with its own "
            "local micro-batch size"
        )

    for entry in dataset_entries:
        accelerator.print(
            f"  - {entry.alias}: weight={entry.weight:.3f}, path={entry.path}"
        )

    is_multi_dataset = len(dataset_entries) > 1
    dataloader_worker_kwargs = {}
    if args.num_workers > 0:
        dataloader_worker_kwargs = {
            "num_workers": args.num_workers,
            "prefetch_factor": 4,
            "persistent_workers": True,
            # Fork can inherit a background monitor's transient subprocess
            # pipe, keeping its writer open and deadlocking tracker shutdown.
            # Spawn also avoids inheriting an initialized NPU runtime.
            "multiprocessing_context": "spawn",
        }

    # One length-metadata sidecar per mix entry, derived from the entry's own
    # dataset directory: loaded when present, generated there (as
    # length_metadata.npz) on first use. Imported here rather than at module
    # scope because it is only needed once training starts.
    from ..dataset.length_metadata import ensure_sidecar

    entry_metadata = [
        ensure_sidecar(entry.path, args.text_encoder_path) for entry in dataset_entries
    ]
    resolution_ids = {
        int(resolution_id)
        for metadata in entry_metadata
        for resolution_id in metadata.resolution_ids
    }
    bucket_plan = load_bucket_plan(args.bucket_plan, resolution_ids)
    entry_datasets = [load_from_disk(str(entry.path)) for entry in dataset_entries]
    for entry, dataset, metadata in zip(
        dataset_entries, entry_datasets, entry_metadata
    ):
        try:
            metadata.validate_against_dataset(dataset)
        except ValueError as exc:
            raise ValueError(f"invalid metadata for {entry.alias}: {exc}") from exc
        accelerator.print(
            f"  loaded {entry.alias}: {len(dataset)} rows, "
            f"{metadata.num_captions} caption lengths"
        )
    row_dataset = RowDescriptorDataset(entry_datasets, entry_metadata)
    caption_policy = CaptionPolicy(
        kind="beta",
        beta_start=args.caption_beta_start,
        beta_end=args.caption_beta_end,
        schedule="linear",
        early_at=0.5,
        short_reserve=args.caption_short_reserve,
        short_threshold=args.caption_short_threshold,
    )
    sampler = RowLengthQueueBatchSampler(
        metadata=entry_metadata,
        bucket_plan=bucket_plan,
        dataset_weights=dataset_weights,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=args.seed + accelerator.process_index,
        initial_stage=args.curriculum_start,
        caption_policy=caption_policy,
    )
    bucket_desc = "; ".join(
        f"res={resolution_id}:"
        + ",".join(f"{bucket.max_length}:{bucket.batch_size}" for bucket in buckets)
        for resolution_id, buckets in sorted(bucket_plan.by_resolution.items())
    )
    accelerator.print(f"  Length-bucketed sampler plan: {bucket_desc}")
    caption_telemetry = CaptionTelemetry()
    # Caption-length weighting forms a weighted mean over the entire update,
    # including every rank and accumulated micro-batch.
    caption_loss_weights = CaptionLossWeights(
        curve="log2",
        reference=args.caption_loss_weight_reference,
    )
    # NumPy payloads cross worker queues by value, avoiding /dev/shm tensor
    # storage on compute pods with a small shared-memory mount.
    collate_fn = functools.partial(row_length_collate_fn, return_numpy=True)
    dataloader = DataLoader(
        row_dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        # Iterator creation draws a worker base seed even with zero workers.
        # Keep that draw out of the training CPU RNG (caption dropout), so
        # recreating the iterator after full resume cannot advance it. Workers
        # only resolve preselected RowRefs; row/caption randomness belongs to
        # the separately checkpointed sampler, not this worker-seed generator.
        generator=torch.Generator().manual_seed(args.seed + accelerator.process_index),
        **dataloader_worker_kwargs,
    )

    # Accelerate does not persist a custom batch sampler that is intentionally
    # kept outside prepare().  Save one state file per rank so the queue tails
    # and row order remain resumable in distributed runs.
    sampler_state_name = "sampler_state_rank_{:05d}.pt".format(
        accelerator.process_index
    )

    # The sampler already assigns complete (resolution, length) micro-batches
    # to ranks; letting Accelerate shard its DataLoader a second time would
    # drop whole batches, so the dataloader stays outside prepare().
    #
    # Schedulers also stay outside prepare(): AcceleratedScheduler.step()
    # steps the inner scheduler num_processes times per call when
    # split_batches=False, which would compress the LR schedule by the GPU
    # count (warmup 500 ending at step 125 on 4 ranks, cosine hitting its
    # floor at max_steps/4). Stepping the bare schedulers below is exactly
    # one schedule step per optimizer step. Their state_dicts are saved and
    # restored by hand next to accelerator.save_state/load_state.
    prepared = accelerator.prepare(model, *optimizers)
    model = prepared[0]
    optimizers = list(prepared[1:])

    # Resume Logic
    resumed_step = 0
    if args.resume and args.resume != "None":
        accelerator.print(f"Resuming from checkpoint: {args.resume}")
        if args.stop_at_step > 0:
            validate_checkpoint(
                args.resume,
                max_steps=args.max_steps,
                stop_at_step=args.stop_at_step,
                require_record=True,
                scheduler_count=len(schedulers),
                use_ema=True,
                world_size=accelerator.num_processes,
            )
        accelerator.load_state(args.resume)
        device_api.set_rng_state(
            torch.load(
                Path(args.resume)
                / f"npu_rng_state_rank_{accelerator.process_index:05d}.pt",
                map_location="cpu",
                weights_only=True,
            ),
            accelerator.device,
        )
        if args.stop_at_step > 0:
            verify_restored_rng(
                args.resume,
                process_index=accelerator.process_index,
                device=accelerator.device,
            )

        resumed_step = resume_step
        # Resolution transitions start a new sampler. Within a stage, every
        # rank restores its own queues and RNG; world size was validated above.
        sidecars = (
            []
            if reset_sampler
            else matching_sampler_sidecars(args.resume, accelerator.num_processes)
        )
        if sidecars:
            sampler.load_state_dict(
                torch.load(
                    sidecars[accelerator.process_index],
                    map_location="cpu",
                    weights_only=False,
                )
            )
            accelerator.print(
                f"Restored sampler state from {sidecars[accelerator.process_index]}"
            )
        elif reset_sampler:
            accelerator.print("Resolution transition: starting the new stage's sampler")
        else:
            raise ValueError("same-stage resume requires every rank's sampler state")
        accelerator.print(f"Full resume at step {resumed_step}")
        # Schedulers are not prepared (see above), so load_state does not
        # restore them; load their state_dicts explicitly.
        for i, sch in enumerate(schedulers):
            sch_path = os.path.join(
                args.resume, "scheduler.bin" if i == 0 else f"scheduler_{i}.bin"
            )
            if not os.path.isfile(sch_path):
                raise ValueError(f"full resume requires scheduler state: {sch_path}")
            state = torch.load(sch_path, map_location="cpu", weights_only=False)
            if state.get("last_epoch") != resumed_step:
                raise ValueError(
                    f"scheduler step does not match checkpoint step: {sch_path}"
                )
            sch.load_state_dict(state)
            # Restore the next update's LR at the resumed schedule position.
            # Recipe equality was checked before loading optimizer state.
            restore_scheduler_base_lrs(sch, optimizer_base_lrs[i])

    if ema_model is not None:
        dtype = next(model_raw.parameters()).dtype
        ema_model.to(accelerator.device, dtype=dtype)
        ema_model.eval()
        ema_model.load_state_dict(model_raw.state_dict())
        if args.resume and args.resume != "None":
            ema_path = os.path.join(args.resume, "ema_weights.pt")
            if not os.path.isfile(ema_path):
                raise ValueError(f"full resume requires EMA state: {ema_path}")
            if os.path.exists(ema_path):
                ema_model.load_state_dict(
                    torch.load(ema_path, map_location="cpu", weights_only=False)
                )
                ema_model.to(accelerator.device, dtype=dtype)
                accelerator.print("Restored EMA weights from checkpoint")
        for param in ema_model.parameters():
            param.requires_grad_(False)

    if args.verify_resume_state:
        from .state_verification import verify_restored_training_state

        verification = verify_restored_training_state(
            args.resume, model_raw, optimizers, schedulers, ema_model
        )
        verify_restored_rng(
            args.resume,
            process_index=accelerator.process_index,
            device=accelerator.device,
        )
        verification.update(
            rng_exact=True,
            step=resumed_step,
            max_steps=args.max_steps,
            rank=accelerator.process_index,
        )
        # This runs before InfraRecorder/checkpoint creation. A fresh resume
        # branch therefore may not have its per-run output directory yet.
        os.makedirs(run_dir, exist_ok=True)
        with open(
            os.path.join(
                run_dir,
                f"restore_step_{resumed_step:06d}_"
                f"rank_{accelerator.process_index:05d}.json",
            ),
            "w",
        ) as handle:
            json.dump(verification, handle, indent=2)

    # Probability Path
    algorithm = FlowMatchingOT()

    # Per-dataset telemetry (for multi-dataset mode)
    dataset_aliases = [entry.alias for entry in dataset_entries]

    # Counted on the host: the micro-batch metadata never leaves the CPU, and
    # the counter only touches the device for the periodic cross-rank reduce.
    telemetry_counts = torch.zeros(len(dataset_aliases), dtype=torch.long)

    # Fixed eval-loss probe (built identically on every rank; main process logs)
    eval_probe = None
    if args.eval_loss_interval > 0:
        eval_probe = EvalLossProbe(
            args.eval_dataset_path,
            text_encoder,
            tokenizer,
            pooling=True,
            exit_layer=args.text_encoder_exit_layer,
            vae_mean=vae_mean,
            vae_std=vae_std,
            num_samples=args.eval_loss_samples,
            batch_size=args.eval_batch_size,
            device=accelerator.device,
        )
        accelerator.print(f"Eval-loss probe ready ({len(eval_probe.latents)} samples)")

    # Training Loop
    global_step = resumed_step

    def evaluate_grid():
        eval_model = (
            ema_model if ema_model is not None else accelerator.unwrap_model(model)
        )
        # Evaluation must not perturb the training timestep/noise RNG stream.
        with torch.random.fork_rng(
            devices=[accelerator.device.index], device_type="npu"
        ):
            run_prompt_grid_eval(
                accelerator=accelerator,
                model=eval_model,
                vae_path=args.vae_path,
                save_path=f"{args.output_dir}/{args.run_name}",
                current_step=global_step,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                pooling=True,
                exit_layer=args.text_encoder_exit_layer,
                prompts_path=args.prompts_file,
                eval_dataset_path=args.eval_dataset_path,
                batch_size=args.eval_batch_size,
                ode_steps=args.ode_steps,
                weights="ema" if ema_model is not None else "live",
            )

    # Only explicit points run before training: this captures the incoming
    # checkpoint at the new resolution without adding grids to every resume.
    if global_step in args.grid_steps:
        evaluate_grid()

    def evaluate_loss_metrics(eval_model):
        started = time.monotonic()
        metrics = eval_probe.evaluate(eval_model)
        if (
            args.stability_interval
            and ema_model is not None
            and accelerator.is_main_process
        ):
            live = eval_probe.evaluate(model_raw)
            metrics.update(
                {
                    key.replace("eval/", "eval_live/", 1): value
                    for key, value in live.items()
                }
            )
        # Includes both EMA and live-model passes when stability telemetry is
        # enabled. Each probe synchronizes when collecting per-sample losses.
        metrics["perf/eval_seconds"] = time.monotonic() - started
        if args.stability_interval and accelerator.is_main_process:
            record_metrics(run_dir, global_step, metrics)
        return metrics

    # Baseline probe read before the first optimizer step. For a resumed run
    # this measures the incoming checkpoint on this run's probe; for a fresh
    # run it records the random-init loss.
    if eval_probe is not None:
        eval_model = (
            ema_model if ema_model is not None else accelerator.unwrap_model(model)
        )
        probe_metrics = evaluate_loss_metrics(eval_model)
        if accelerator.is_main_process:
            accelerator.print(
                f"[eval-loss@{global_step}] "
                + " ".join(f"{key}={value:.5f}" for key, value in probe_metrics.items())
            )
            accelerator.log(probe_metrics, step=global_step)

    progress_bar = tqdm(
        total=args.max_steps,
        initial=resumed_step,
        disable=not accelerator.is_local_main_process,
    )

    # Full resume may intentionally have no sampler sidecar when changing
    # resolution. Set whole-run progress before iter(dataloader) can prefetch
    # any captions, and before initializing policy telemetry. For same-stage
    # resume, set_stage leaves restored queues, replay batches and RNG intact.
    set_caption_curriculum(
        sampler,
        global_step=global_step,
        max_steps=args.max_steps,
        curriculum_start=args.curriculum_start,
        curriculum_end=args.curriculum_end,
    )

    def current_policy_state() -> PolicyState:
        # The telemetry records the policy that produced the captions rather
        # than one recomputed later: the sampler's own stage is what the
        # within-row selector used, and for the length-preference selector the
        # length_preference_beta is beta at that position.
        return PolicyState(
            progress=global_step / max(1, args.max_steps),
            curriculum_position=sampler.stage,
            length_preference_beta=caption_policy.beta(sampler.stage),
        )

    policy_state = current_policy_state()
    # One optimizer step's weighted mean is built up here: the micro-batches
    # add their weighted losses and their weights, and the boundary below
    # divides once. See StepLossAccumulator for why the division is not moved
    # into the micro-batches.
    step_losses = StepLossAccumulator(accelerator.device)
    micro_count = 0
    step_global_loss = None
    step_global_samples = None
    device_api.synchronize(accelerator.device)
    last_step_time = time.monotonic()
    sps_ema = None
    train_wall_total = 0.0
    train_samples_total = 0
    train_steps_total = 0
    steady_wall_total = 0.0
    steady_samples_total = 0
    peak_mem_gb = 0.0
    peak_reserved_gb = 0.0

    stability_monitor = (
        StabilityMonitor(model_raw, optimizers, run_dir, autocast=accelerator.autocast)
        if args.stability_interval and accelerator.is_main_process
        else None
    )

    infra_recorder = None
    if args.infra_metrics:
        infra_recorder = InfraRecorder(
            run_dir,
            accelerator.process_index,
            device_type="npu",
            trace_start=args.trace_start,
            trace_steps=args.trace_steps,
            record_identity=args.record_identity,
        )

    train_iter = iter(dataloader) if global_step < end_step else iter(())

    model.train()

    # ---- Text encoding -----------------------------------------------------
    # The frozen Qwen forward is pure inference and runs inline on the compute
    # stream. (A variant that encoded one micro-batch ahead on a side CUDA
    # stream was tried and rejected on measurement.)
    latents_device_for_dropout = accelerator.device

    # Micro-batch fields that are only ever read on the host: telemetry ids,
    # caption lengths, the bucket upper bound, the batch id used for the
    # replay acknowledgement. Moving them to the device with the rest of the
    # batch and reading them back (``.tolist()``/``int()``) enqueues a
    # device-to-host copy behind all queued compute, so the CPU stalls until
    # the GPU drains and CPU and GPU work serialize every micro-batch
    # instead of overlapping. They are popped before the device transfer and
    # stay on the host.
    host_batch_keys = (
        "dataset_ids",
        "retained_lengths",
        "row_positions",
        "row_indices",
        "caption_indices",
        "bucket_hi",
        "batch_id",
        "resolution_bucket_ids",
    )

    def select_captions(batch):
        captions_list = batch["captions"]
        # Length-bucketed batches carry one pre-selected caption per item, so
        # only the classifier-free-guidance dropout is applied here.
        selected = list(captions_list)

        if args.caption_dropout_prob > 0:
            # Each sample drops independently, including a whole micro-batch.
            # Empty prompts retain chat suffix tokens in the text encoder.
            drop_mask = torch.rand(len(selected)) < args.caption_dropout_prob
            for i, drop in enumerate(drop_mask.tolist()):
                if drop:
                    selected[i] = ""
        else:
            drop_mask = torch.zeros(len(selected), dtype=torch.bool)
        # The mask is returned rather than recomputed so the telemetry reports
        # the dropout that actually happened instead of drawing a second one.
        return selected, drop_mask.tolist()

    def prepare_latents(batch):
        latents = batch["latents"]
        # Normalize latents: z_norm = (z - mean) / std
        latents = (latents - vae_mean) / vae_std
        t = sample_logit_normal_timesteps(
            latents.shape[0],
            latents.device,
            mu=args.logit_normal_mu,
            sigma=args.logit_normal_sigma,
        )
        return latents, shift_timesteps(t, latents)

    # Per-optimizer-step breakdown (--step_breakdown): record device-event
    # pairs per segment per micro-batch, settle (sync + elapsed) once per
    # optimizer step. Segments: text (frozen encoder), fwd (DiT fwd + loss),
    # bwd, opt (clip+step+zero), ema.
    if args.step_breakdown:
        _bd_event_mod = device_api
        bd_seg: dict = {"text": [], "fwd": [], "bwd": [], "opt": [], "ema": []}

        def bd_mark(seg: str):
            ev = _bd_event_mod.Event(enable_timing=True)
            ev.record()
            bd_seg[seg].append(ev)

        def bd_settle():
            _bd_event_mod.synchronize()
            out = {}
            for seg, evs in bd_seg.items():
                pairs = len(evs) // 2
                total_ms = sum(
                    evs[2 * i].elapsed_time(evs[2 * i + 1]) for i in range(pairs)
                )
                out[seg] = total_ms / pairs if pairs else 0.0
                bd_seg[seg] = []
            return out

    bd_cur = None

    # CPU wall-clock profile (--cpu_wall_profile): per-micro next/encode/micro
    # totals, settled per optimizer step. Locates CPU time outside the GPU
    # segments (post-compile bottleneck hunt).
    # Per-micro CPU sub-segments (wall-clock): prep (normalize/timestep/caption
    # select) / enc (encode_text) / pad (bucket-hi pad) / fwd (flow matching +
    # DiT fwd + loss) / bwd; syncopt (grad-div+clip+step+zero) and post
    # (ema+scheduler+telemetry) are per-step costs counted separately.
    cpu_acc = (
        {
            "next_ms": 0.0,
            "enc_ms": 0.0,
            "micro_ms": 0.0,
            "n": 0,
            "prep_ms": 0.0,
            "pad_ms": 0.0,
            "fwd_ms": 0.0,
            "bwd_ms": 0.0,
            "syncopt_ms": 0.0,
            "n_sync": 0,
            "post_ms": 0.0,
            "n_post": 0,
        }
        if args.cpu_wall_profile
        else None
    )

    def save_checkpoint(step: int) -> None:
        checkpoint_started = time.monotonic()
        save_path = os.path.join(
            args.output_dir,
            f"{args.run_name}/checkpoint_step_{step:06d}",
        )
        os.makedirs(save_path, exist_ok=True)
        if accelerator.is_main_process:
            marker = os.path.join(save_path, CHECKPOINT_RECORD)
            if os.path.exists(marker):
                os.remove(
                    marker
                )  # Invalidate an older completion record before overwriting.
        accelerator.wait_for_everyone()
        accelerator.save_state(save_path)
        # Save the rank-local NPU generator explicitly; its preservation must
        # not depend on Accelerate's device-extension integration.
        torch.save(
            device_api.get_rng_state(accelerator.device).cpu(),
            Path(save_path) / f"npu_rng_state_rank_{accelerator.process_index:05d}.pt",
        )
        accelerator.wait_for_everyone()
        # Scheduler state (schedulers are deliberately not prepared, so
        # save_state does not cover them; see the prepare call site).
        if accelerator.is_main_process:
            for i, sch in enumerate(schedulers):
                torch.save(
                    sch.state_dict(),
                    os.path.join(
                        save_path, "scheduler.bin" if i == 0 else f"scheduler_{i}.bin"
                    ),
                )
        sampler_path = os.path.join(save_path, sampler_state_name)
        sampler_tmp = sampler_path + ".tmp"
        torch.save(sampler.state_dict(), sampler_tmp)
        os.replace(sampler_tmp, sampler_path)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if ema_model is not None:
                torch.save(
                    ema_model.state_dict(),
                    os.path.join(save_path, "ema_weights.pt"),
                )
            Path(save_path, "bucket_plan.json").write_text(
                json.dumps(bucket_contents, indent=2) + "\n"
            )
            Path(save_path, "run_config.json").write_text(
                json.dumps(asdict(config), indent=2) + "\n"
            )
            Path(save_path, "transformer_config.json").write_text(
                json.dumps(model_raw.get_config(), indent=2) + "\n"
            )
            # Persist runtime state for crash-resume (step + swanlab run id)
            runtime = {"global_step": step}
            try:
                import swanlab

                runtime["swanlab_run_id"] = getattr(swanlab.get_run(), "id", None)
            except Exception:
                pass
            os.makedirs(run_dir, exist_ok=True)
            with open(runtime_path, "w") as f:
                json.dump(runtime, f)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            write_checkpoint_record(
                save_path,
                step=step,
                max_steps=args.max_steps,
                scheduler_count=len(schedulers),
                use_ema=ema_model is not None,
                world_size=accelerator.num_processes,
                device_type="npu",
            )
            if args.checkpoint_keep_last:
                removed = prune_checkpoints(
                    save_path,
                    keep_last=args.checkpoint_keep_last,
                    max_steps=args.max_steps,
                    scheduler_count=len(schedulers),
                    use_ema=ema_model is not None,
                    world_size=accelerator.num_processes,
                )
                if removed:
                    accelerator.print(
                        f"Pruned older complete checkpoints: {', '.join(removed)}"
                    )
        accelerator.wait_for_everyone()

        if infra_recorder is not None:
            infra_recorder.checkpoint(
                step=step, seconds=time.monotonic() - checkpoint_started
            )

    while global_step < end_step:
        if (
            infra_recorder is not None
            and micro_count % args.gradient_accumulation_steps == 0
        ):
            infra_recorder.begin_update(global_step)
        if cpu_acc is not None:
            _t_next0 = time.monotonic()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            batch = next(train_iter)
        if cpu_acc is not None:
            cpu_acc["next_ms"] += (time.monotonic() - _t_next0) * 1e3
            _t_micro0 = time.monotonic()
            _t_prep0 = _t_micro0

        optimizer_step_boundary = (
            micro_count + 1
        ) % args.gradient_accumulation_steps == 0
        # Micro-batches carry their own (resolution, length bucket) and local
        # batch size, so Accelerate's accumulation helper does not apply.
        # DDP reduction is deferred to the optimizer boundary; the accumulated
        # gradient is unchanged because
        # the reduction is linear.
        if not optimizer_step_boundary:
            accumulation_context = accelerator.no_sync(model)
        else:
            accumulation_context = contextlib.nullcontext()
        with accumulation_context:
            host_meta = {key: batch.pop(key) for key in host_batch_keys if key in batch}
            if infra_recorder is not None:
                identity = (
                    dict(
                        rows=host_meta["row_positions"],
                        captions=host_meta["caption_indices"].tolist(),
                        batch_id=host_meta["batch_id"],
                    )
                    if infra_recorder.record_identity
                    else None
                )
                infra_recorder.micro(
                    batch["latents"].shape,
                    host_meta["bucket_hi"],
                    sample_identity=identity,
                )

            def to_device(value):
                if torch.is_tensor(value):
                    return value.to(accelerator.device, non_blocking=True)
                if isinstance(value, np.ndarray):
                    # Restore tensors after the NumPy worker-queue transfer.
                    return torch.from_numpy(value).to(
                        accelerator.device, non_blocking=True
                    )
                return value

            batch = {key: to_device(value) for key, value in batch.items()}

            # Track per-dataset samples (multi-dataset mode)
            if "dataset_ids" in host_meta:
                telemetry_counts += torch.bincount(
                    torch.as_tensor(host_meta["dataset_ids"]),
                    minlength=len(dataset_aliases),
                )

            # 1. Prepare inputs (normalize latents, draw timesteps)
            # 2. Encode the pre-selected caption. Caption selection by the
            #    short-to-long curriculum happens inside the sampler, which is
            #    advanced with sampler.set_stage at the step boundary.
            if args.cpu_wall_profile:
                _t_enc0 = time.monotonic()
                cpu_acc["prep_ms"] += (_t_enc0 - _t_prep0) * 1e3
            if args.step_breakdown:
                bd_mark("text")
            latents, t = prepare_latents(batch)
            selected_captions, dropped_mask = select_captions(batch)
            # Caption lengths and the loss weights they imply are host values:
            # a curve evaluation is a host-side computation and the total the
            # optimizer step normalizes by is a sum of them, so neither needs a
            # device round-trip.
            retained_lengths = host_meta["retained_lengths"].tolist()
            micro_weights = caption_loss_weights.for_micro_batch(
                retained_lengths, dropped_mask, accelerator.device
            )
            caption_telemetry.record(
                retained_lengths,
                dropped_mask,
                int(host_meta["bucket_hi"]),
                row_positions=host_meta["row_positions"],
                policy=policy_state,
                loss_weights=micro_weights.values,
            )
            txt, txt_mask, txt_pooled = encode_training_text(
                selected_captions,
                text_encoder,
                tokenizer,
                exit_layer=args.text_encoder_exit_layer,
                bucket_hi=int(host_meta["bucket_hi"]),
            )
            if args.step_breakdown:
                bd_mark("text")
            if args.cpu_wall_profile:
                cpu_acc["enc_ms"] += (time.monotonic() - _t_enc0) * 1e3

            # Flow Matching
            if args.cpu_wall_profile:
                _t_fwd0 = time.monotonic()
            z1 = latents
            z0 = torch.randn_like(z1)
            z_t = algorithm.sample_zt(z0, z1, t)

            # Per-micro-batch shape log for out-of-memory post-mortems: with
            # --log_shapes enabled, every micro-batch prints the dimensions
            # that decide its activation memory (resolution, padded text
            # length, local batch size) together with the memory level the
            # forward starts from, so a later OOM can be attributed to the
            # exact bucket that caused it instead of reconstructed by hand.
            if log_shapes:
                accelerator.print(
                    f"[shape] step={global_step} micro={micro_count} "
                    f"res={int(host_meta['resolution_bucket_ids'][0])} "
                    f"txt_hi={int(host_meta['bucket_hi'])} "
                    f"txt_len={txt.shape[1]} "
                    f"B={int(latents.shape[0])} "
                    f"latent={z_t.shape[-2]}x{z_t.shape[-1]} "
                    f"mem_gb={_device_mem_allocated(accelerator.device):.2f}"
                )
            # Forward
            if args.step_breakdown:
                bd_mark("fwd")
            model_output = model(
                z_t,
                t,
                txt=txt,
                txt_pooled=txt_pooled,
                txt_mask=txt_mask,
            )

            # Loss
            #
            # A micro-batch whose samples carry different caption-length
            # weights is reduced to their weighted mean; one whose weights are
            # all equal is a plain rescale of the mean the algorithm already
            # computes, and takes that path instead. Either way the step
            # accumulates the weighted losses and the weights and divides once,
            # at the optimizer boundary, which is what makes the step the
            # weighted mean of its samples however it was split into
            # micro-batches.
            loss = algorithm.compute_loss(
                model_output, z0, z1, t, sample_weights=micro_weights.tensor
            )
            local_batch_size = int(latents.shape[0])
            # ``total`` is the micro-batch's share of the step's normalization:
            # the sum of its samples' weights, which is exactly the sample
            # count while the table is all ones. DDP averages each rank's
            # accumulated weighted-loss gradient across ranks; the optimizer
            # boundary divides by the global weight per rank to recover the
            # weighted mean over the actual samples.
            loss_for_backward = step_losses.add(
                loss, micro_weights.total, local_batch_size
            )
            if args.step_breakdown:
                bd_mark("fwd")
            if args.cpu_wall_profile:
                # fwd segment covers flow matching + DiT fwd + loss (CPU wall,
                # incl. any syncs the compiled region forces).
                cpu_acc["fwd_ms"] += (time.monotonic() - _t_fwd0) * 1e3

            if args.step_breakdown:
                bd_mark("bwd")
            _t_bwd0 = time.monotonic() if args.cpu_wall_profile else None
            accelerator.backward(loss_for_backward)
            # Keep an in-flight batch replayable until backward succeeds. If
            # the process dies before this acknowledgement, the next run
            # replays the batch instead of silently losing it.
            sampler.ack_batch(host_meta["batch_id"])
            micro_count += 1
            if cpu_acc is not None:
                cpu_acc["bwd_ms"] += (time.monotonic() - _t_bwd0) * 1e3
            if args.step_breakdown:
                bd_mark("bwd")

            should_optimizer_step = optimizer_step_boundary
            grad_norm = None

            if should_optimizer_step:
                if args.cpu_wall_profile:
                    _t_sync0 = time.monotonic()
                # Sample count and total loss weight travel in one collective:
                # the count says how much data the step used, the weight is
                # what the gradient is normalized by. While the weight table is
                # all ones the two are the same number.
                local_step_totals = torch.tensor(
                    [float(step_losses.sample_count), step_losses.weight_sum],
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                step_totals = accelerator.reduce(local_step_totals, reduction="sum")
                global_sample_count, global_loss_weight = step_totals.tolist()
                if global_sample_count <= 0:
                    raise RuntimeError("optimizer step has no samples")
                grad_divisor = global_loss_weight / accelerator.num_processes
                divide_gradients(model.parameters(), grad_divisor, foreach=True)
                global_loss_sum = accelerator.reduce(
                    step_losses.loss_sum(), reduction="sum"
                )
                # The step's weighted mean, divided in the dtype the reduction
                # produced it in.
                step_global_loss = weighted_mean(global_loss_sum, step_totals[1]).item()
                step_global_samples = int(global_sample_count)
                if args.step_breakdown:
                    bd_mark("opt")
                stability_due = stability_monitor is not None and (
                    global_step == resumed_step
                    or (global_step + 1) % args.stability_interval == 0
                )
                if stability_due:
                    stability_monitor.ensure_panel(z0, z1, txt, txt_pooled, txt_mask)
                    stability_monitor.before_update()
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                require_finite_update(step_global_loss, grad_norm)

                health_snapshot = None
                if (
                    args.health_interval
                    and (global_step + 1) % args.health_interval == 0
                ):
                    health_snapshot = snapshot_weights(optimizers, device="cpu")

                for opt in optimizers:
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                stability_metrics = None
                if stability_due:
                    stability_metrics = stability_monitor.after_update(
                        step=global_step + 1, applied=True
                    )
                    accelerator.print(
                        f"[stability@{global_step + 1}] "
                        f"c_update={stability_metrics['stability/conditioning/update_rms']:.6g} "
                        f"c_gain={stability_metrics['stability/response/conditioning_gain']:.6g} "
                        f"loss_delta_t050={stability_metrics['stability/response/t050/conditioning_loss_delta']:.6g} "
                        f"probe_s={stability_metrics['stability/probe_seconds']:.2f}",
                        flush=True,
                    )

                health_metrics = None
                if health_snapshot is not None:
                    ratios = update_weight_ratios(health_snapshot)
                    # A GPU snapshot must not survive into the next forward's
                    # activation peak. Ratios contain only Python scalars.
                    health_snapshot = None
                    health_metrics = {
                        "health/update_weight_ratio_muon": ratios[0],
                    }
                    if len(ratios) > 1:
                        health_metrics["health/update_weight_ratio_aux"] = ratios[-1]
                if args.step_breakdown:
                    bd_mark("opt")
                if args.cpu_wall_profile:
                    cpu_acc["syncopt_ms"] += (time.monotonic() - _t_sync0) * 1e3
                    cpu_acc["n_sync"] += 1

        if cpu_acc is not None:
            cpu_acc["micro_ms"] += (time.monotonic() - _t_micro0) * 1e3
            cpu_acc["n"] += 1
            _t_post0 = time.monotonic()

        if should_optimizer_step:
            for sch in schedulers:
                sch.step()
            global_step += 1

            step_samples = step_global_samples
            step_loss = step_global_loss
            step_losses.reset()
            set_caption_curriculum(
                sampler,
                global_step=global_step,
                max_steps=args.max_steps,
                curriculum_start=args.curriculum_start,
                curriculum_end=args.curriculum_end,
            )
            policy_state = current_policy_state()

            if ema_model is not None and global_step % args.ema_update_interval == 0:
                if args.step_breakdown:
                    bd_mark("ema")
                update_ema_model(
                    ema_model,
                    model_raw,
                    ema_decay_at(global_step, args.ema_decay, warmup=True),
                    foreach=True,
                )
                if args.step_breakdown:
                    bd_mark("ema")

            if health_metrics is not None:
                gains = qk_gain_stats(model_raw)
                if gains is not None:
                    health_metrics["health/attn_qk_gain_max"] = gains[0]
                    health_metrics["health/attn_qk_gain_mean"] = gains[1]
                if ema_model is not None:
                    health_metrics["health/ema_rel_distance"] = ema_rel_distance(
                        ema_model, model_raw
                    )

            progress_bar.update(1)

            if args.step_breakdown:
                bd_cur = bd_settle()

            if cpu_acc is not None:
                cpu_acc["post_ms"] += (time.monotonic() - _t_post0) * 1e3
                cpu_acc["n_post"] += 1

            if args.cpu_wall_profile and global_step % args.telemetry_log_interval == 0:
                _n = max(cpu_acc["n"], 1)
                _ns = max(cpu_acc["n_sync"], 1)
                _np = max(cpu_acc["n_post"], 1)
                _seg_sum = (
                    cpu_acc["prep_ms"]
                    + cpu_acc["enc_ms"]
                    + cpu_acc["pad_ms"]
                    + cpu_acc["fwd_ms"]
                    + cpu_acc["bwd_ms"]
                )
                # micro_ms starts after next(train_iter); subtracting next_ms
                # again would hide unaccounted host time. Host call durations
                # can wait for earlier asynchronous CUDA work; they are not
                # GPU segment durations or a CPU/GPU critical-path partition.
                _resid = cpu_acc["micro_ms"] - _seg_sum - cpu_acc["syncopt_ms"]
                accelerator.print(
                    f"[cpu-wall@{global_step}] micro={cpu_acc['micro_ms']/_n:.1f}ms "
                    f"(next={cpu_acc['next_ms']/_n:.1f} prep={cpu_acc['prep_ms']/_n:.1f} "
                    f"enc={cpu_acc['enc_ms']/_n:.1f} pad={cpu_acc['pad_ms']/_n:.1f} "
                    f"fwd={cpu_acc['fwd_ms']/_n:.1f} bwd={cpu_acc['bwd_ms']/_n:.1f}) "
                    f"residual={max(0.0, _resid)/_n:.1f}ms; "
                    f"per-step: syncopt={cpu_acc['syncopt_ms']/_ns:.1f}ms "
                    f"post={cpu_acc['post_ms']/_np:.1f}ms",
                )
                for _k in cpu_acc:
                    cpu_acc[_k] = 0.0

            # Per-dataset telemetry (log periodically)
            synchronized_counts = None
            if is_multi_dataset and global_step % args.telemetry_log_interval == 0:
                synchronized_counts = {}
                total_samples = 0

                synced = accelerator.reduce(
                    telemetry_counts.to(accelerator.device), reduction="sum"
                ).tolist()
                for alias, count in zip(dataset_aliases, synced):
                    synchronized_counts[alias] = count
                    total_samples += count
                telemetry_counts.zero_()

            # Caption telemetry: every rank joins the window reduction and
            # closes its own window; only the main process logs below.
            caption_window = None
            if global_step % args.telemetry_log_interval == 0:
                caption_telemetry.reduce(
                    accelerator.device, world_size=accelerator.num_processes
                )
                caption_window = caption_telemetry.snapshot(reset=True)

            # CUDA cache clearing (independent of telemetry)

            # Live telemetry includes completed optimizer and EMA GPU work.
            # The final summary below additionally includes logging overhead.
            device_api.synchronize(accelerator.device)
            step_dt = time.monotonic() - last_step_time
            infra_memory = None
            if infra_recorder is not None:
                infra_memory = (
                    device_api.max_memory_allocated(accelerator.device),
                    device_api.max_memory_reserved(accelerator.device),
                )
            if step_dt > 0:
                sps = step_samples / step_dt
                sps_ema = sps if sps_ema is None else 0.95 * sps_ema + 0.05 * sps

            if accelerator.is_main_process:
                log_dict = {
                    "train/loss": step_loss,
                    "train/lr": optimizers[0].param_groups[0]["lr"],
                    "train/update_applied": 1.0,
                    "train/consecutive_skips": 0,
                }
                if len(optimizers) > 1:
                    log_dict["train/lr_aux"] = optimizers[-1].param_groups[0]["lr"]
                if grad_norm is not None:
                    log_dict["train/grad_norm"] = float(grad_norm)
                if sps_ema is not None:
                    log_dict["train/samples_per_sec"] = sps_ema
                peak = _device_peak_mem(accelerator.device)
                if peak is not None:
                    # Memory telemetry: catch transient spikes (long-caption batches,
                    # ckpt saves, eval overlaps) instead of guessing post-OOM.
                    step_peak_gb, step_reserved_gb = peak
                    peak_mem_gb = max(peak_mem_gb, step_peak_gb)
                    # Reserved (the allocator's own ceiling) is what decides whether
                    # a larger batch fits, so a batch-size screen needs it next to
                    # the allocated figure.
                    peak_reserved_gb = max(peak_reserved_gb, step_reserved_gb)
                    log_dict["train/mem_peak_gb"] = step_peak_gb
                    _device_reset_peak_mem(accelerator.device)
                if caption_window:
                    log_dict.update(caption_window)

                if bd_cur is not None:
                    for seg, ms in bd_cur.items():
                        log_dict[f"train/ms_{seg}"] = ms

                if synchronized_counts is not None:
                    total_samples = sum(synchronized_counts.values())
                    if total_samples > 0:
                        for alias, count in synchronized_counts.items():
                            log_dict[f"data/{alias}_ratio"] = count / total_samples

                if health_metrics is not None:
                    log_dict.update(health_metrics)
                if stability_metrics is not None:
                    log_dict.update(stability_metrics)
                if stability_monitor is not None:
                    record_metrics(run_dir, global_step, log_dict)

                accelerator.log(log_dict, step=global_step)

            # End the measured training interval only after optimizer, EMA,
            # telemetry and logging have completed. CUDA launches are async:
            # counting host dispatch alone would omit their unfinished work.
            device_api.synchronize(accelerator.device)
            training_end = time.monotonic()
            measured_step_dt = training_end - last_step_time
            train_wall_total += measured_step_dt
            train_samples_total += step_samples
            train_steps_total += 1
            if train_steps_total > args.steady_state_skip_steps:
                steady_wall_total += measured_step_dt
                steady_samples_total += step_samples

            if infra_recorder is not None:
                infra_recorder.end_update(
                    step=global_step,
                    seconds=measured_step_dt,
                    global_samples=step_samples,
                    loss=step_loss,
                    progress=sampler.stage,
                    peak_allocated=infra_memory[0],
                    peak_reserved=infra_memory[1],
                    breakdown=bd_cur,
                )

            # Always persist an endpoint before potentially expensive/failing eval.
            if global_step % args.checkpoint_interval == 0 or global_step == end_step:
                save_checkpoint(global_step)

            # Fixed eval-loss probe (primary cross-run comparison metric)
            if eval_probe is not None and global_step % args.eval_loss_interval == 0:
                eval_model = (
                    ema_model
                    if ema_model is not None
                    else accelerator.unwrap_model(model)
                )
                probe_metrics = evaluate_loss_metrics(eval_model)
                if accelerator.is_main_process:
                    accelerator.print(
                        f"[eval-loss@{global_step}] "
                        + " ".join(
                            f"{key}={value:.5f}" for key, value in probe_metrics.items()
                        )
                    )
                    accelerator.log(probe_metrics, step=global_step)

            # Fixed-prompt sample grids
            if grid_due(global_step, args.eval_interval, args.grid_steps):
                evaluate_grid()

            # Exclude only evaluation/checkpoint work from the next interval.
            device_api.synchronize(accelerator.device)
            last_step_time = time.monotonic()

            if global_step == end_step:
                accelerator.print(
                    f"[stop] reached stage endpoint {end_step}; global T={args.max_steps}"
                )

    # End-of-training KID (fixed fakes vs full held-out real); only for a run
    # that actually completed its schedule, not an early stop_at_step exit.
    if args.kid_num_fake > 0 and global_step >= args.max_steps:
        eval_model = (
            ema_model if ema_model is not None else accelerator.unwrap_model(model)
        )
        run_kid_eval(
            accelerator=accelerator,
            model=eval_model,
            vae_path=args.vae_path,
            save_path=f"{args.output_dir}/{args.run_name}",
            current_step=global_step,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            pooling=True,
            exit_layer=args.text_encoder_exit_layer,
            dataset_path=args.eval_dataset_path,
            num_fake=args.kid_num_fake,
            batch_size=args.eval_batch_size,
            ode_steps=args.ode_steps,
        )

    if accelerator.is_main_process and train_wall_total > 0:
        # Self-reported throughput. step_dt excludes eval/checkpoint time (the
        # clock is reset after those blocks). The steady_* figures drop the
        # initial allocator/kernel warmup steps set in the recipe. This is
        # the primary comparator between runs.
        steady = ""
        if steady_wall_total > 0:
            steady = (
                f"steady_steps={train_steps_total - args.steady_state_skip_steps} "
                f"samples_per_sec_steady="
                f"{steady_samples_total / steady_wall_total:.2f} "
            )
        accelerator.print(
            f"[throughput-summary] steps={train_steps_total} "
            f"samples={train_samples_total} "
            f"train_wall_s={train_wall_total:.1f} "
            f"samples_per_sec={train_samples_total / train_wall_total:.2f} "
            f"samples_per_step={train_samples_total / max(train_steps_total, 1):.1f} "
            f"{steady}"
            f"peak_mem_gb={peak_mem_gb:.1f} "
            f"peak_mem_reserved_gb={peak_reserved_gb:.1f}"
        )

    if infra_recorder is not None:
        infra_recorder.close()
    # end_training destroys the process group; finish collective work first.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.print("[training-complete]", flush=True)


if __name__ == "__main__":
    main()

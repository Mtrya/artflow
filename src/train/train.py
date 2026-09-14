"""Train the ArtFlow conditional rectified-flow generator.

The experiment definition — data mix, (resolution, caption-length) bucket
plan, model shape, optimizer and schedule, eval cadence — is loaded from one
or more TOML config files (``--config``, layered in order). The command line
carries only run-time control (which configs, run name, resume, profiling)
and the diagnostic fast-path switches.
"""

import os
import gc
import argparse
import contextlib
import json
import re
import time
import warnings
import math
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_scheduler
from datasets import load_from_disk

from ..models.artflow import ArtFlow
from ..models.dit_blocks import set_sdpa_backends
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
from ..dataset.mix import parse_dataset_mix, get_dataset_weights
from ..utils.encode_text import encode_text
from ..utils.vae_codec import get_vae_stats
from ..flow.paths import FlowMatchingOT, shift_timesteps
from ..evaluation.eval_loss import EvalLossProbe
from ..evaluation.prompt_grid import grid_due, run_prompt_grid_eval
from ..evaluation.kid_eval import run_kid_eval
from .config import load_config, flatten
from .stage_control import (
    CHECKPOINT_RECORD, stage_endpoint, validate_checkpoint, write_checkpoint_record,
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
    ema_model: torch.nn.Module, current_model: torch.nn.Module, decay: float
) -> None:
    for ema_param, param in zip(ema_model.parameters(), current_model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), current_model.buffers()):
        ema_buffer.copy_(buffer)


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
    sampler.set_stage(
        curriculum_start + (curriculum_end - curriculum_start) * progress
    )


def matching_sampler_sidecars(checkpoint: str, world_size: int) -> list[str]:
    """Only a complete, exact rank set can restore stride-sharded row order."""
    expected = [f"sampler_state_rank_{rank:05d}.pt" for rank in range(world_size)]
    present = {name for name in os.listdir(checkpoint)
               if re.fullmatch(r"sampler_state_rank_\d+\.pt", name)
               and os.path.isfile(os.path.join(checkpoint, name))}
    return [os.path.join(checkpoint, name) for name in expected] if present == set(expected) else []


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

def sample_logit_normal_timesteps(batch_size, device, mu=0.0, sigma=1.0):
    """Sample timesteps from a logit-normal distribution."""
    logit = mu + sigma * torch.randn(batch_size, device=device)
    t = torch.sigmoid(logit)
    return t


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the ArtFlow conditional generator. The experiment "
        "definition (data mix, bucket plan, model shape, optimizer, schedule, "
        "eval cadence) is loaded from TOML config files; these flags carry "
        "only run-time control and the diagnostic fast-path switches.",
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        metavar="PATH",
        help="TOML config file; repeat to layer, later files override earlier ones",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="artflow_run",
        help="Run name; output subdirectory under the config's output_dir and "
        "the SwanLab run name",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint directory to resume weights+optimizer from; the step "
        "schedule restarts at zero unless --resume_full is also given",
    )
    parser.add_argument(
        "--resume_full",
        action="store_true",
        help="Crash-resume: also restore global_step, scheduler, EMA, batch "
        "sampler and RNG state from the checkpoint",
    )
    parser.add_argument(
        "--resume_ema",
        action="store_true",
        help="With a plain --resume, also restore EMA weights from the "
        "checkpoint (eval-only probes; without it EMA re-initializes from "
        "the resumed model weights)",
    )
    parser.add_argument(
        "--reset_sampler", action="store_true",
        help="With --resume_full, start fresh resolution-specific sampler queues; "
             "keep the global step, scheduler, optimizer, EMA and RNG state",
    )
    parser.add_argument(
        "--step_breakdown",
        action="store_true",
        help="Measure per-optimizer-step GPU time of the text-encode / "
        "forward / backward / optimizer / EMA segments with CUDA events and "
        "log train/ms_* every telemetry_log_interval steps (adds one device "
        "sync per step; for profiling only)",
    )
    parser.add_argument(
        "--cpu_wall_profile",
        action="store_true",
        help="Measure per-micro-batch CPU wall-clock of next(batch) / "
        "encode_text / the whole micro, printed every telemetry_log_interval "
        "steps (locates the CPU time left after torch.compile)",
    )
    # Diagnostic fast-path switches. Each one mirrors the validated throughput
    # stack and defaults to on; pass --no-<name> to restore the baseline path
    # for regression or profiling comparisons.
    parser.add_argument(
        "--text_encoder_exit_mode",
        choices=("full_forward_slice", "stop_at_layer"),
        default="stop_at_layer",
        help="Frozen text forward implementation; full_forward_slice restores "
        "the baseline for throughput comparisons",
    )
    parser.add_argument(
        "--fast_caption_dropout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the caption-dropout mask on the CPU so the step does not "
        "synchronize on a device-side reduction (identical dropout "
        "distribution)",
    )
    parser.add_argument(
        "--fast_telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accumulate per-dataset sample counts with one bincount instead "
        "of one device kernel per sample",
    )
    parser.add_argument(
        "--fast_text_slice",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Slice the padded text hidden tensor instead of "
        "gathering/repacking per sequence (removes a host sync; identical "
        "conditioning features; requires right padding)",
    )
    parser.add_argument(
        "--attn_bias_hoist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hoist RoPE frequencies and the padded-text attention bias out "
        "of the DiT block loop (identical math, fewer per-layer ops)",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the DiT before DDP wrapping (the EMA copy is "
        "taken pre-compile). Eager Python dispatch dominates wall time, so "
        "this is the largest single lever",
    )
    parser.add_argument(
        "--compile_blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --compile: compile each DiT block separately. All blocks "
        "share one graph per bucket shape, so compile time is paid once "
        "instead of once per shape for a 24-layer graph",
    )
    parser.add_argument(
        "--muon_batched_ns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Muon's Newton-Schulz iterations as one batched matmul per "
        "matrix shape instead of a serial chain of small GEMMs (same "
        "per-matrix math)",
    )
    parser.add_argument(
        "--ddp_boundary_sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reduce DDP gradients once per optimizer step instead of once "
        "per micro-batch: micro-batches inside a step accumulate locally "
        "under no_sync. The gradient is identical because the reduction is "
        "linear, and the per-micro straggler wait across ranks disappears",
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
        raise ValueError(
            f"bucket plan is missing metadata resolution IDs: {missing}"
        )
    return BucketPlan(by_resolution)


def main():
    parser = parse_args()
    cli = parser.parse_args()
    config = load_config(cli.config)
    args = SimpleNamespace(**flatten(config))
    for name in (
        "run_name",
        "resume",
        "resume_full",
        "resume_ema",
        "reset_sampler",
        "step_breakdown",
        "cpu_wall_profile",
        "fast_caption_dropout",
        "fast_telemetry",
        "fast_text_slice",
        "attn_bias_hoist",
        "compile",
        "compile_blocks",
        "muon_batched_ns",
        "ddp_boundary_sync",
        "text_encoder_exit_mode",
    ):
        setattr(args, name, getattr(cli, name))

    if args.reset_sampler and not (args.resume_full and args.resume):
        raise ValueError("--reset_sampler requires --resume and --resume_full")
    # Reject invalid stages/checkpoints before allocating the model or GPUs.
    resume_step = 0
    if args.resume_full and args.resume and args.resume != "None":
        resume_step = validate_checkpoint(
            args.resume, max_steps=args.max_steps, stop_at_step=args.stop_at_step,
            require_record=args.stop_at_step > 0, use_ema=args.use_ema,
        )
    end_step = stage_endpoint(args.max_steps, args.stop_at_step, resume_step)

    # The masked (padded-text) attention path uses the memory-efficient SDPA
    # backend; the cuDNN backend was measured slower for these shapes.
    set_sdpa_backends(["EFFICIENT_ATTENTION", "MATH"])

    # Allocation-history capture for out-of-memory post-mortems, enabled by
    # setting ARTFLOW_MEM_SNAPSHOT to a file prefix: the recorder runs from
    # startup and the snapshot (which allocation stacks hold what) is dumped
    # at interpreter exit, including after an unhandled OOM.  Recording slows
    # the run slightly, so it stays off unless the variable is set.
    mem_snapshot_prefix = os.environ.get("ARTFLOW_MEM_SNAPSHOT")
    if mem_snapshot_prefix and torch.cuda.is_available():
        import atexit

        torch.cuda.memory._record_memory_history(max_entries=100000)
        atexit.register(torch.cuda.memory._dump_snapshot,
                        f"{mem_snapshot_prefix}.pickle")

    # Per-micro-batch shape logging, the cheap half of an OOM post-mortem (see
    # the [shape] line ahead of the DiT forward).
    log_shapes = bool(os.environ.get("ARTFLOW_LOG_SHAPES"))

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

    set_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    run_dir = os.path.join(args.output_dir, args.run_name)
    runtime_path = os.path.join(run_dir, "runtime.json")

    # SwanLab run resumption: reuse the stored run id on crash-resume so curves
    # stay in one run instead of fragmenting across restarts.
    swanlab_kwargs = {"experiment_name": args.run_name}
    if args.resume_full and os.path.exists(runtime_path):
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
            config=vars(args),
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
    model = ArtFlow(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        double_stream_depth=args.double_stream_depth,
        single_stream_depth=args.single_stream_depth,
        mlp_ratio=args.mlp_ratio,
        conditioning_scheme=args.conditioning_scheme,
        qkv_bias=args.qkv_bias,
        double_stream_modulation=args.double_stream_modulation,
        single_stream_modulation=args.single_stream_modulation,
        ffn_type=args.ffn_type,
        rope_centered_grid=args.rope_centered_grid,
        # Default params
        patch_size=2,
        in_channels=16,
        txt_in_features=1024,
    )
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
        muon_momentum=args.muon_momentum,
        batched_ns=args.muon_batched_ns,
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
        if args.lr_scheduler_type == "linear_cosine":
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
        else:
            schedulers.append(
                get_scheduler(
                    args.lr_scheduler_type,
                    optimizer=opt,
                    num_warmup_steps=args.lr_warmup_steps,
                    num_training_steps=args.max_steps,
                )
            )

    ema_model = deepcopy(model) if args.use_ema else None

    # torch.compile BEFORE accelerator.prepare (DDP wrapping): the compiled
    # graph must not capture distributed collectives. EMA was deep-copied from
    # the raw model above, so it stays an eager copy (eval-only).
    # model_raw keeps an unprefixed reference: the compiled wrapper's
    # state_dict keys carry an `_orig_mod.` prefix, so every EMA/state_dict
    # path must read from model_raw (parameters are SHARED with the wrapper).
    model_raw = model
    if args.compile and args.compile_blocks:
        # Per-block compilation: all 24 blocks share the same shapes, so one
        # graph per (resolution, length, batch) is compiled once and reused,
        # instead of one 24-layer graph per shape. Same fusion opportunities on
        # the elementwise/norm/modulation ops, ~20x less compile time.
        # Dynamo specializes on shape AND stride, so every (length bucket,
        # Per-block compilation shares one dynamo cache per code object, and
        # every (resolution, length bucket) pair - hence every (sequence
        # length, local batch) pair - is a separate graph.  The bucket plan
        # multiplies length buckets by resolution ids, and dynamo spends more
        # than one cache entry per shape while it specializes, so a small
        # limit is exhausted mid-run; a shape compiled after the limit runs
        # eagerly, and an eager forward keeps roughly twice the activation
        # memory of its compiled graph (observed: 38 GB vs 20 GB at batch 16
        # with a ~1250-token text, enough to OOM a 48 GB card by itself).
        # Keep the floor well above any plausible plan's shape count.
        try:
            import torch._dynamo as _dynamo

            limit = max(int(getattr(_dynamo.config, "recompile_limit", 8)), 512)
            _dynamo.config.recompile_limit = limit
            if hasattr(_dynamo.config, "cache_size_limit"):
                _dynamo.config.cache_size_limit = limit
        except Exception as exc:  # pragma: no cover - version guard
            accelerator.print(f"could not raise the dynamo recompile limit ({exc})")
        accelerator.print(
            "Compiling each DiT block (mode=\"default\", "
            f"{len(model_raw.blocks)} blocks sharing one graph per shape, "
            f"recompile_limit={getattr(_dynamo.config, 'recompile_limit', '?')})..."
        )
        t0 = time.time()
        for block in model_raw.blocks:
            block.forward = torch.compile(
                block.forward, mode="default", dynamic=False
            )
        accelerator.print(
            f"per-block torch.compile wrappers ready ({time.time() - t0:.1f}s; "
            "first step per shape triggers graph compilation)"
        )
    elif args.compile:
        accelerator.print("Compiling DiT with torch.compile (mode=\"default\")...")
        t0 = time.time()
        model = torch.compile(model, mode="default")
        accelerator.print(
            f"torch.compile wrapper ready ({time.time() - t0:.1f}s; first step "
            "triggers graph compilation)"
        )

    # Dataset (with optional mixing)
    accelerator.print(f"Parsing dataset mix: {args.dataset_mix}")
    dataset_entries = parse_dataset_mix(args.dataset_mix)
    dataset_weights = get_dataset_weights(dataset_entries)
    if args.stage_sync_interval < 1:
        raise ValueError("stage_sync_interval (config [data]) must be >= 1")
    if not args.bucket_plan:
        raise ValueError(
            "a non-empty [data] bucket_plan is required: every dataset row is "
            "assigned to a (resolution, caption-length) bucket with its own "
            "local micro-batch size"
        )

    for entry in dataset_entries:
        accelerator.print(f"  - {entry.alias}: weight={entry.weight:.3f}, path={entry.path}")

    is_multi_dataset = len(dataset_entries) > 1
    dataloader_worker_kwargs = {}
    if args.num_workers > 0:
        dataloader_worker_kwargs = {
            "num_workers": args.num_workers,
            "prefetch_factor": 4,
            "persistent_workers": True,
        }

    # One length-metadata sidecar per mix entry, derived from the entry's own
    # dataset directory: loaded when present, generated there (as
    # length_metadata.npz) on first use. Imported here rather than at module
    # scope because it is only needed once training starts.
    from ..dataset.length_metadata import ensure_sidecar

    entry_metadata = [
        ensure_sidecar(entry.path, args.text_encoder_path)
        for entry in dataset_entries
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
        kind=args.caption_policy,
        beta_start=args.caption_beta_start,
        beta_end=args.caption_beta_end,
        schedule=args.caption_schedule,
        early_at=args.caption_early_at,
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
        f"res={resolution_id}:" + ",".join(
            f"{bucket.max_length}:{bucket.batch_size}"
            for bucket in buckets
        )
        for resolution_id, buckets in sorted(bucket_plan.by_resolution.items())
    )
    accelerator.print(f"  Length-bucketed sampler plan: {bucket_desc}")
    caption_telemetry = CaptionTelemetry(
        short_threshold=caption_policy.short_threshold,
        log_every=args.telemetry_log_interval,
    )
    # Per-sample loss weights as a curve over retained caption length. The
    # "none" curve (the default) leaves the loss and its normalization exactly
    # as they were, so a run is only reweighted when its config says so.
    caption_loss_weights = CaptionLossWeights(
        curve=args.caption_loss_weight_curve,
        reference=args.caption_loss_weight_reference,
    )
    dataloader = DataLoader(
        row_dataset,
        batch_sampler=sampler,
        collate_fn=row_length_collate_fn,
        pin_memory=True,
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
        accelerator.load_state(args.resume)

        if args.resume_full:
            # Crash-resume restores model/optimizer state and step metadata.
            # The batch sampler state is restored from its rank-local sidecar.
            resumed_step = resume_step
            # Each rank walks a stride-sharded row order under its own RNG
            # stream, so a sampler state only fits the world size that wrote
            # it. Restore only when the exact rank set is present; a resume
            # that changes the GPU count starts a fresh sampler instead.
            sidecars = ([] if args.reset_sampler else
                        matching_sampler_sidecars(args.resume, accelerator.num_processes))
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
            else:
                accelerator.print(
                    "Sampler reset requested or sidecars do not cover this world size; "
                    "starting a fresh sampler (data order restarts)."
                )
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
                    raise ValueError(f"scheduler step does not match checkpoint step: {sch_path}")
                sch.load_state_dict(state)
        else:
            # A plain --resume carries weights+optimizer but restarts the
            # schedule from step 0: the schedulers are fresh (not restored by
            # load_state), so only the optimizer LRs restored from the
            # checkpoint need resetting until the first scheduler step.
            for optimizer, base_lrs in zip(optimizers, optimizer_base_lrs):
                for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
                    param_group["lr"] = args.start_learning_rate * (
                        base_lr / args.learning_rate
                    )
            accelerator.print(
                "Resumed model/optimizer state. Resetting global_step and scheduler."
            )

    if ema_model is not None:
        dtype = next(model_raw.parameters()).dtype
        ema_model.to(accelerator.device, dtype=dtype)
        ema_model.eval()
        ema_model.load_state_dict(model_raw.state_dict())
        if (args.resume_full or args.resume_ema) and args.resume and args.resume != "None":
            ema_path = os.path.join(args.resume, "ema_weights.pt")
            if args.resume_full and not os.path.isfile(ema_path):
                raise ValueError(f"full resume requires EMA state: {ema_path}")
            if os.path.exists(ema_path):
                ema_model.load_state_dict(
                    torch.load(ema_path, map_location="cpu", weights_only=False)
                )
                ema_model.to(accelerator.device, dtype=dtype)
                accelerator.print("Restored EMA weights from checkpoint")
        for param in ema_model.parameters():
            param.requires_grad_(False)

    # Probability Path
    algorithm = FlowMatchingOT()

    # Per-dataset telemetry (for multi-dataset mode)
    dataset_aliases = [entry.alias for entry in dataset_entries]

    # Initialize telemetry tensors for distributed synchronization
    telemetry_tensors = {
        alias: torch.zeros(1, dtype=torch.long, device=accelerator.device)
        for alias in dataset_aliases
    }
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
            pooling=(args.conditioning_scheme == "fused"),
            exit_layer=args.text_encoder_exit_layer,
            vae_mean=vae_mean,
            vae_std=vae_std,
            num_samples=args.eval_loss_samples,
            device=accelerator.device,
        )
        accelerator.print(f"Eval-loss probe ready ({len(eval_probe.latents)} samples)")

    # Training Loop
    global_step = resumed_step

    def evaluate_grid():
        eval_model = ema_model if ema_model is not None else accelerator.unwrap_model(model)
        # Evaluation must not perturb the training timestep/noise RNG stream.
        devices = [accelerator.device] if accelerator.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            run_prompt_grid_eval(
                accelerator=accelerator, model=eval_model, vae_path=args.vae_path,
                save_path=f"{args.output_dir}/{args.run_name}", current_step=global_step,
                text_encoder=text_encoder, tokenizer=tokenizer,
                pooling=(args.conditioning_scheme == "fused"),
                exit_layer=args.text_encoder_exit_layer, prompts_path=args.prompts_file,
                eval_dataset_path=args.eval_dataset_path, batch_size=args.eval_batch_size,
                ode_steps=args.ode_steps,
                weights="ema" if ema_model is not None else "live",
            )

    # Only explicit points run before training: this captures the incoming
    # checkpoint at the new resolution without adding grids to every resume.
    if global_step in args.grid_steps:
        evaluate_grid()

    # Baseline probe read before the first optimizer step. For a resumed run
    # this measures the incoming checkpoint on this run's probe; for a fresh
    # run it records the random-init loss.
    if eval_probe is not None:
        eval_model = (
            ema_model if ema_model is not None else accelerator.unwrap_model(model)
        )
        probe_metrics = eval_probe.evaluate(eval_model)
        if accelerator.is_main_process:
            accelerator.print(
                f"[eval-loss@{global_step}] "
                + " ".join(
                    f"{key}={value:.5f}" for key, value in probe_metrics.items()
                )
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
            length_preference_beta=(
                caption_policy.beta(sampler.stage)
                if caption_policy.kind == "beta"
                else sampler.stage
            ),
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
    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)
    last_step_time = time.monotonic()
    sps_ema = None
    train_wall_total = 0.0
    train_samples_total = 0
    train_steps_total = 0
    steady_wall_total = 0.0
    steady_samples_total = 0
    peak_mem_gb = 0.0
    peak_reserved_gb = 0.0

    train_iter = iter(dataloader) if global_step < end_step else iter(())

    model.train()

    fast_flags = [
        name
        for name, enabled in (
            ("fast_caption_dropout", args.fast_caption_dropout),
            ("fast_telemetry", args.fast_telemetry),
            ("fast_text_slice", args.fast_text_slice),
            ("attn_bias_hoist", args.attn_bias_hoist),
            ("muon_batched_ns", args.muon_batched_ns),
            ("ddp_boundary_sync", args.ddp_boundary_sync),
            ("compile", args.compile),
            ("compile_blocks", args.compile_blocks),
        )
        if enabled
    ]
    accelerator.print(
        "Fast-path flags enabled (--no-<flag> disables): "
        + (", ".join(fast_flags) if fast_flags else "none")
    )

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
            if args.fast_caption_dropout:
                # Host-side draw: same distribution, no device reduction.
                drop_mask = torch.rand(len(selected)) < args.caption_dropout_prob
                if bool(drop_mask.all()):
                    drop_mask[torch.randint(len(selected), (1,)).item()] = False
            else:
                drop_mask = (
                    torch.rand(
                        len(selected), device=latents_device_for_dropout
                    )
                    < args.caption_dropout_prob
                )
                if drop_mask.all():
                    keep_idx = torch.randint(
                        len(selected), (1,), device=latents_device_for_dropout
                    )
                    drop_mask[keep_idx] = False
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
        if args.use_logit_normal_sampling:
            t = sample_logit_normal_timesteps(
                latents.shape[0],
                latents.device,
                mu=args.logit_normal_mu,
                sigma=args.logit_normal_sigma,
            )
        else:
            t = torch.rand(latents.shape[0], device=latents.device)
        return latents, shift_timesteps(t, latents)

    def encode_micro(batch, selected_captions, bucket_hi):
        # The frozen encoder stops after text_encoder_exit_layer: stopping the
        # forward at that layer was verified feature-identical to slicing a
        # full forward's hidden states, and it skips the remaining layers.
        txt, txt_mask, txt_pooled = encode_text(
            selected_captions,
            text_encoder,
            tokenizer,
            pooling=(args.conditioning_scheme == "fused"),
            exit_layer=args.text_encoder_exit_layer,
            exit_mode=args.text_encoder_exit_mode,
            fast_slice=args.fast_text_slice,
        )
        txt, txt_mask = pad_text_to_hi(txt, txt_mask, bucket_hi)
        return txt, txt_mask, txt_pooled

    # Per-optimizer-step breakdown (--step_breakdown): record CUDA-event
    # pairs per segment per micro-batch, settle (sync + elapsed) once per
    # optimizer step. Segments: text (frozen encoder), fwd (DiT fwd + loss),
    # bwd, opt (clip+step+zero), ema.
    if args.step_breakdown:
        bd_seg: dict = {"text": [], "fwd": [], "bwd": [], "opt": [], "ema": []}

        def bd_mark(seg: str):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            bd_seg[seg].append(ev)

        def bd_settle():
            torch.cuda.synchronize()
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
    cpu_acc = {
        "next_ms": 0.0, "enc_ms": 0.0, "micro_ms": 0.0, "n": 0,
        "prep_ms": 0.0, "pad_ms": 0.0, "fwd_ms": 0.0, "bwd_ms": 0.0,
        "syncopt_ms": 0.0, "n_sync": 0, "post_ms": 0.0, "n_post": 0,
    } if args.cpu_wall_profile else None

    def save_checkpoint(step: int) -> None:
        save_path = os.path.join(
            args.output_dir,
            f"{args.run_name}/checkpoint_step_{step:06d}",
        )
        os.makedirs(save_path, exist_ok=True)
        if accelerator.is_main_process:
            marker = os.path.join(save_path, CHECKPOINT_RECORD)
            if os.path.exists(marker):
                os.remove(marker)  # Invalidate an older completion record before overwriting.
        accelerator.wait_for_everyone()
        accelerator.save_state(save_path)
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
                save_path, step=step, max_steps=args.max_steps,
                scheduler_count=len(schedulers), use_ema=ema_model is not None,
                world_size=accelerator.num_processes,
            )
        accelerator.wait_for_everyone()

    while global_step < end_step:
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
            (micro_count + 1) % args.gradient_accumulation_steps == 0
        )
        # Micro-batches carry their own (resolution, length bucket) and local
        # batch size, so Accelerate's accumulation helper does not apply. With
        # ddp_boundary_sync (default) the DDP reduction is deferred to the
        # optimizer boundary; the accumulated gradient is unchanged because
        # the reduction is linear.
        if args.ddp_boundary_sync and not optimizer_step_boundary:
            accumulation_context = accelerator.no_sync(model)
        else:
            accumulation_context = contextlib.nullcontext()
        with accumulation_context:
            host_meta = {
                key: batch.pop(key) for key in host_batch_keys if key in batch
            }
            batch = {
                key: value.to(accelerator.device, non_blocking=True)
                if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            # Track per-dataset samples (multi-dataset mode)
            if "dataset_ids" in host_meta:
                if args.fast_telemetry:
                    telemetry_counts += torch.bincount(
                        host_meta["dataset_ids"], minlength=len(dataset_aliases)
                    )
                else:
                    for ds_id in host_meta["dataset_ids"].tolist():
                        alias = dataset_aliases[ds_id]
                        telemetry_tensors[alias] += 1

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
                retained_lengths, dropped_mask, accelerator.device)
            caption_telemetry.record(
                retained_lengths, dropped_mask,
                int(host_meta["bucket_hi"]),
                row_positions=host_meta["row_positions"],
                policy=policy_state,
                loss_weights=micro_weights.values)
            txt, txt_mask, txt_pooled = encode_micro(
                batch, selected_captions, int(host_meta["bucket_hi"]))
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
            # ARTFLOW_LOG_SHAPES set, every micro-batch prints the dimensions
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
                    f"mem_gb={torch.cuda.memory_allocated() / 1024**3:.2f}"
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
                fast_attn=args.attn_bias_hoist,
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
                model_output, z0, z1, t, sample_weights=micro_weights.tensor)
            local_batch_size = int(latents.shape[0])
            # ``total`` is the micro-batch's share of the step's normalization:
            # the sum of its samples' weights, which is exactly the sample
            # count while the table is all ones. DDP averages each rank's
            # accumulated weighted-loss gradient across ranks; the optimizer
            # boundary divides by the global weight per rank to recover the
            # weighted mean over the actual samples.
            loss_for_backward = step_losses.add(
                loss, micro_weights.total, local_batch_size)
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
                step_totals = accelerator.reduce(
                    local_step_totals, reduction="sum"
                )
                global_sample_count, global_loss_weight = step_totals.tolist()
                if global_sample_count <= 0:
                    raise RuntimeError("optimizer step has no samples")
                grad_divisor = global_loss_weight / accelerator.num_processes
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(grad_divisor)
                global_loss_sum = accelerator.reduce(
                    step_losses.loss_sum(), reduction="sum"
                )
                # The step's weighted mean, divided in the dtype the reduction
                # produced it in.
                step_global_loss = weighted_mean(
                    global_loss_sum, step_totals[1]
                ).item()
                step_global_samples = int(global_sample_count)
                if args.step_breakdown:
                    bd_mark("opt")
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )

                health_snapshot = None
                if (
                    args.health_interval
                    and (global_step + 1) % args.health_interval == 0
                ):
                    health_snapshot = snapshot_weights(optimizers)

                for opt in optimizers:
                    opt.step()
                    opt.zero_grad()

                health_metrics = None
                if health_snapshot is not None:
                    ratios = update_weight_ratios(health_snapshot)
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
            if global_step % args.stage_sync_interval == 0:
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
                    ema_model, model_raw, args.ema_decay
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
                    cpu_acc["prep_ms"] + cpu_acc["enc_ms"] + cpu_acc["pad_ms"]
                    + cpu_acc["fwd_ms"] + cpu_acc["bwd_ms"]
                )
                _resid = cpu_acc["micro_ms"] - cpu_acc["next_ms"] - _seg_sum - cpu_acc["syncopt_ms"]
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

                if args.fast_telemetry:
                    synced = accelerator.reduce(
                        telemetry_counts.to(accelerator.device), reduction="sum"
                    ).tolist()
                    for alias, count in zip(dataset_aliases, synced):
                        synchronized_counts[alias] = count
                        total_samples += count
                    telemetry_counts.zero_()
                else:
                    for alias in dataset_aliases:
                        # Reduce (sum) across all processes to avoid deadlocks
                        synced_tensor = accelerator.reduce(
                            telemetry_tensors[alias], reduction="sum"
                        )
                        count = synced_tensor.item()
                        synchronized_counts[alias] = count
                        total_samples += count

                    # Reset telemetry counters on every rank for the next interval
                    for alias in dataset_aliases:
                        telemetry_tensors[alias].zero_()

            # Caption telemetry: every rank joins the window reduction and
            # closes its own window; only the main process logs below.
            caption_window = None
            if global_step % args.telemetry_log_interval == 0:
                caption_telemetry.reduce(accelerator.device,
                                         world_size=accelerator.num_processes)
                caption_window = caption_telemetry.snapshot(reset=True)

            # CUDA cache clearing (independent of telemetry)
            if global_step % args.cache_clear_interval == 0:
                gc.collect()
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()

            # Live telemetry includes completed optimizer and EMA GPU work.
            # The final summary below additionally includes logging overhead.
            if torch.cuda.is_available():
                torch.cuda.synchronize(accelerator.device)
            step_dt = time.monotonic() - last_step_time
            if step_dt > 0:
                sps = step_samples / step_dt
                sps_ema = sps if sps_ema is None else 0.95 * sps_ema + 0.05 * sps

            if accelerator.is_main_process:
                log_dict = {
                    "train/loss": step_loss,
                    "train/lr": optimizers[0].param_groups[0]["lr"],
                }
                if len(optimizers) > 1:
                    log_dict["train/lr_aux"] = optimizers[-1].param_groups[0]["lr"]
                if grad_norm is not None:
                    log_dict["train/grad_norm"] = float(grad_norm)
                if sps_ema is not None:
                    log_dict["train/samples_per_sec"] = sps_ema
                if torch.cuda.is_available():
                    # Memory telemetry: catch transient spikes (long-caption batches,
                    # ckpt saves, eval overlaps) instead of guessing post-OOM.
                    step_peak_gb = torch.cuda.max_memory_allocated() / 1024**3
                    peak_mem_gb = max(peak_mem_gb, step_peak_gb)
                    # Reserved (the allocator's own ceiling) is what decides whether
                    # a larger batch fits, so a batch-size screen needs it next to
                    # the allocated figure.
                    peak_reserved_gb = max(
                        peak_reserved_gb, torch.cuda.max_memory_reserved() / 1024**3
                    )
                    log_dict["train/mem_peak_gb"] = step_peak_gb
                    torch.cuda.reset_peak_memory_stats()
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

                accelerator.log(log_dict, step=global_step)

            # End the measured training interval only after optimizer, EMA,
            # telemetry and logging have completed. CUDA launches are async:
            # counting host dispatch alone would omit their unfinished work.
            if torch.cuda.is_available():
                torch.cuda.synchronize(accelerator.device)
            training_end = time.monotonic()
            measured_step_dt = training_end - last_step_time
            train_wall_total += measured_step_dt
            train_samples_total += step_samples
            train_steps_total += 1
            if train_steps_total > args.steady_state_skip_steps:
                steady_wall_total += measured_step_dt
                steady_samples_total += step_samples

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
                probe_metrics = eval_probe.evaluate(eval_model)
                if accelerator.is_main_process:
                    accelerator.print(
                        f"[eval-loss@{global_step}] "
                        + " ".join(
                            f"{key}={value:.5f}"
                            for key, value in probe_metrics.items()
                        )
                    )
                    accelerator.log(probe_metrics, step=global_step)

            # Fixed-prompt sample grids
            if grid_due(global_step, args.eval_interval, args.grid_steps):
                evaluate_grid()

            # Exclude only evaluation/checkpoint work from the next interval.
            if torch.cuda.is_available():
                torch.cuda.synchronize(accelerator.device)
            last_step_time = time.monotonic()

            if global_step == end_step:
                accelerator.print(
                    f"[stop] reached stage endpoint {end_step}; global T={args.max_steps}"
                )

    # End-of-training KID (fixed fakes vs full held-out real); only for a run
    # that actually completed its schedule, not an early stop_at_step exit.
    if args.kid_eval_at_end and global_step >= args.max_steps:
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
            pooling=(args.conditioning_scheme == "fused"),
            exit_layer=args.text_encoder_exit_layer,
            dataset_path=args.eval_dataset_path,
            num_fake=args.kid_num_fake,
            batch_size=args.eval_batch_size,
        )

    if accelerator.is_main_process and train_wall_total > 0:
        # Self-reported throughput. step_dt excludes eval/checkpoint time (the
        # clock is reset after those blocks) but does include one-time
        # torch.compile stalls; the steady_* figures drop the first
        # steady_state_skip_steps steps (config [train]) so two runs can be
        # compared without a compile-warmup term. The steady-state figure is
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

    accelerator.end_training()
    print("Training finished.")


if __name__ == "__main__":
    main()

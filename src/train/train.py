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
from ..dataset.mix import parse_dataset_mix, get_dataset_weights
from ..utils.encode_text import encode_text
from ..utils.vae_codec import get_vae_stats
from ..flow.paths import FlowMatchingOT, shift_timesteps
from ..evaluation.eval_loss import EvalLossProbe
from ..evaluation.prompt_grid import run_prompt_grid_eval
from ..evaluation.kid_eval import run_kid_eval
from .config import load_config, flatten
from .muon import build_param_groups as build_muon_param_groups

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
    ):
        setattr(args, name, getattr(cli, name))

    # The masked (padded-text) attention path uses the memory-efficient SDPA
    # backend; the cuDNN backend was measured slower for these shapes.
    set_sdpa_backends(["EFFICIENT_ATTENTION", "MATH"])

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
        # local batch) pair is a separate graph; raise the per-code-object
        # recompile limit above the bucket count so it never silently falls
        # back to eager.
        try:
            import torch._dynamo as _dynamo

            limit = max(int(getattr(_dynamo.config, "recompile_limit", 8)), 64)
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
    sampler = RowLengthQueueBatchSampler(
        metadata=entry_metadata,
        bucket_plan=bucket_plan,
        dataset_weights=dataset_weights,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=args.seed + accelerator.process_index,
        initial_stage=args.curriculum_start,
    )
    bucket_desc = "; ".join(
        f"res={resolution_id}:" + ",".join(
            f"{bucket.max_length}:{bucket.batch_size}"
            for bucket in buckets
        )
        for resolution_id, buckets in sorted(bucket_plan.by_resolution.items())
    )
    accelerator.print(f"  Length-bucketed sampler plan: {bucket_desc}")
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
    prepared = accelerator.prepare(model, *optimizers, *schedulers)
    model = prepared[0]
    optimizers = list(prepared[1 : 1 + len(optimizers)])
    schedulers = list(prepared[1 + len(optimizers) :])

    # Resume Logic
    resumed_step = 0
    if args.resume and args.resume != "None":
        accelerator.print(f"Resuming from checkpoint: {args.resume}")
        accelerator.load_state(args.resume)

        if args.resume_full:
            # Crash-resume restores model/optimizer state and step metadata.
            # The batch sampler state is restored from its rank-local sidecar.
            m = re.search(r"checkpoint_step_(\d+)", args.resume.rstrip("/"))
            if m is None:
                raise ValueError(
                    f"--resume_full needs a checkpoint_step_* path, got {args.resume}"
                )
            resumed_step = int(m.group(1))
            resume_sampler_path = os.path.join(args.resume, sampler_state_name)
            if not os.path.exists(resume_sampler_path):
                raise ValueError(
                    "full resume requires the rank-local sampler state: "
                    f"{resume_sampler_path}"
                )
            sampler.load_state_dict(
                torch.load(resume_sampler_path, map_location="cpu", weights_only=False)
            )
            accelerator.print(f"Restored sampler state from {resume_sampler_path}")
            accelerator.print(f"Full resume at step {resumed_step}")
        else:
            # A plain --resume carries weights+optimizer but restarts the
            # schedule from step 0.
            for sch in schedulers:
                if hasattr(sch, "last_epoch"):
                    sch.last_epoch = -1
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
        if args.resume_full and args.resume and args.resume != "None":
            ema_path = os.path.join(args.resume, "ema_weights.pt")
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
    telemetry_counts = torch.zeros(
        len(dataset_aliases), dtype=torch.long, device=accelerator.device
    )

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
    progress_bar = tqdm(
        total=args.max_steps,
        initial=resumed_step,
        disable=not accelerator.is_local_main_process,
    )

    stage = args.curriculum_start
    micro_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    micro_sample_count = 0
    micro_count = 0
    step_global_loss = None
    step_global_samples = None
    last_step_time = time.time()
    sps_ema = None
    train_wall_total = 0.0
    train_samples_total = 0
    train_steps_total = 0
    steady_wall_total = 0.0
    steady_samples_total = 0
    peak_mem_gb = 0.0

    train_iter = iter(dataloader)

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
        return selected

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

    def encode_micro(batch, selected_captions):
        # The frozen encoder stops after text_encoder_exit_layer: stopping the
        # forward at that layer was verified feature-identical to slicing a
        # full forward's hidden states, and it skips the remaining layers.
        txt, txt_mask, txt_pooled = encode_text(
            selected_captions,
            text_encoder,
            tokenizer,
            pooling=(args.conditioning_scheme == "fused"),
            exit_layer=args.text_encoder_exit_layer,
            exit_mode="stop_at_layer",
            fast_slice=args.fast_text_slice,
        )
        txt, txt_mask = pad_text_to_hi(txt, txt_mask, batch["bucket_hi"])
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

    while global_step < args.max_steps:
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
            batch = {
                key: value.to(accelerator.device, non_blocking=True)
                if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            # Track per-dataset samples (multi-dataset mode)
            if "dataset_ids" in batch:
                if args.fast_telemetry:
                    telemetry_counts += torch.bincount(
                        batch["dataset_ids"], minlength=len(dataset_aliases)
                    )
                else:
                    for ds_id in batch["dataset_ids"].tolist():
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
            selected_captions = select_captions(batch)
            txt, txt_mask, txt_pooled = encode_micro(batch, selected_captions)
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
            loss = algorithm.compute_loss(model_output, z0, z1, t)
            local_batch_size = int(latents.shape[0])
            # Micro-batches may hold different local sample counts, so each
            # micro's loss is weighted by its own sample count before it is
            # summed for the optimizer step. DDP averages each per-micro
            # sample-weighted gradient sum across ranks; the optimizer
            # boundary divides by global samples / num_ranks to recover the
            # mean over the actual samples.
            micro_sample_count += local_batch_size
            micro_loss_sum += loss.detach().float() * local_batch_size
            loss_for_backward = loss * local_batch_size
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
            sampler.ack_batch(batch["batch_id"])
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
                local_sample_count = torch.tensor(
                    float(micro_sample_count),
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                global_sample_count = accelerator.reduce(
                    local_sample_count, reduction="sum"
                )
                if global_sample_count.item() <= 0:
                    raise RuntimeError("optimizer step has no samples")
                grad_divisor = global_sample_count.item() / accelerator.num_processes
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(grad_divisor)
                global_loss_sum = accelerator.reduce(
                    micro_loss_sum, reduction="sum"
                )
                step_global_loss = (
                    global_loss_sum / global_sample_count
                ).item()
                step_global_samples = int(global_sample_count.item())
                if args.step_breakdown:
                    bd_mark("opt")
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )

                for opt in optimizers:
                    opt.step()
                    opt.zero_grad()
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
            micro_sample_count = 0
            micro_loss_sum.zero_()
            progress = global_step / args.max_steps
            stage = args.curriculum_start + (
                args.curriculum_end - args.curriculum_start
            ) * progress
            if global_step % args.stage_sync_interval == 0:
                sampler.set_stage(stage)

            now = time.time()
            step_dt = now - last_step_time
            last_step_time = now
            train_wall_total += step_dt
            train_samples_total += step_samples
            train_steps_total += 1
            if train_steps_total > args.steady_state_skip_steps:
                steady_wall_total += step_dt
                steady_samples_total += step_samples
            if step_dt > 0:
                sps = step_samples / step_dt
                sps_ema = sps if sps_ema is None else 0.95 * sps_ema + 0.05 * sps

            if ema_model is not None and global_step % args.ema_update_interval == 0:
                if args.step_breakdown:
                    bd_mark("ema")
                update_ema_model(
                    ema_model, model_raw, args.ema_decay
                )
                if args.step_breakdown:
                    bd_mark("ema")

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
                        telemetry_counts, reduction="sum"
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

            # CUDA cache clearing (independent of telemetry)
            if global_step % args.cache_clear_interval == 0:
                gc.collect()
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()

            if accelerator.is_main_process:
                log_dict = {
                    "train/loss": step_loss,
                    "train/stage": stage,
                    "train/lr": optimizers[0].param_groups[0]["lr"],
                    "train/global_samples": step_samples,
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
                    log_dict["train/mem_peak_gb"] = step_peak_gb
                    log_dict["train/mem_alloc_gb"] = (
                        torch.cuda.memory_allocated() / 1024**3
                    )
                    torch.cuda.reset_peak_memory_stats()
                log_dict["train/txt_seq_len"] = float(txt.shape[1])

                if bd_cur is not None:
                    for seg, ms in bd_cur.items():
                        log_dict[f"train/ms_{seg}"] = ms

                if synchronized_counts is not None:
                    total_samples = sum(synchronized_counts.values())
                    if total_samples > 0:
                        for alias, count in synchronized_counts.items():
                            ratio = count / total_samples
                            log_dict[f"data/{alias}_ratio"] = ratio
                            log_dict[f"data/{alias}_count"] = count

                accelerator.log(log_dict, step=global_step)

            # Checkpointing
            if global_step % args.checkpoint_interval == 0:
                save_path = os.path.join(
                    args.output_dir,
                    f"{args.run_name}/checkpoint_step_{global_step:06d}",
                )
                os.makedirs(save_path, exist_ok=True)
                accelerator.save_state(save_path)
                accelerator.wait_for_everyone()
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
                    runtime = {"global_step": global_step}
                    try:
                        import swanlab

                        runtime["swanlab_run_id"] = getattr(swanlab.get_run(), "id", None)
                    except Exception:
                        pass
                    os.makedirs(run_dir, exist_ok=True)
                    with open(runtime_path, "w") as f:
                        json.dump(runtime, f)

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
            if global_step % args.eval_interval == 0:
                eval_model = (
                    ema_model
                    if ema_model is not None
                    else accelerator.unwrap_model(model)
                )
                run_prompt_grid_eval(
                    accelerator=accelerator,
                    model=eval_model,
                    vae_path=args.vae_path,
                    save_path=f"{args.output_dir}/{args.run_name}",
                    current_step=global_step,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    pooling=(args.conditioning_scheme == "fused"),
                    exit_layer=args.text_encoder_exit_layer,
                    prompts_path=args.prompts_file,
                    eval_dataset_path=args.eval_dataset_path,
                    batch_size=args.eval_batch_size,
                )
                model.train()

            # Eval/ckpt blocks above run inside this optimizer step's wall time;
            # reset the clock so train/samples_per_sec reflects training only.
            last_step_time = time.time()

    # End-of-training KID (fixed fakes vs full held-out real)
    if args.kid_eval_at_end:
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
            f"peak_mem_gb={peak_mem_gb:.1f}"
        )

    accelerator.end_training()
    print("Training finished.")


if __name__ == "__main__":
    main()

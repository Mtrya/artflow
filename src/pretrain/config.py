"""Training configuration: TOML files plus run-time overrides.

An experiment is defined by a config file (see ``configs/base.toml`` for the
shipped recipe), optionally layered with more config files that override it.
The command line carries only run-time control: which configs to load, where
to write, whether to resume, and whether to profile.

``load_config`` merges the files in order and validates the result.
``flatten`` turns the nested dataclass into the flat attribute names the
training loop reads, so the loop itself does not care how the values were
grouped for readability.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .caption_loss_weights import CaptionLossWeights
from .stage_control import stage_endpoint


@dataclass(frozen=True)
class DataConfig:
    """What is trained on, and how rows are drawn from it."""

    mix: str = ""
    bucket_plan: str = ""
    caption_dropout_prob: float = 0.1
    # Which caption inside a drawn row is used. "legacy" keeps the original
    # heuristic short-to-long curriculum; "beta" selects on exact retained
    # lengths with a length preference and a short-caption reserve.
    caption_policy: str = "legacy"
    curriculum_start: float = 0.0
    curriculum_end: float = 1.0
    caption_beta_start: float = -1.0
    caption_beta_end: float = 1.0
    # "linear" ramps beta over the run, "early" reaches beta_end earlier and
    # holds it, "stationary" uses each row's schedule-averaged probabilities.
    caption_schedule: str = "linear"
    caption_early_at: float = 0.5
    caption_short_reserve: float = 0.20
    caption_short_threshold: int = 256
    stage_sync_interval: int = 1


@dataclass(frozen=True)
class ModelConfig:
    """Transformer shape and conditioning."""

    hidden_size: int = 1152
    num_heads: int = 16
    double_stream_depth: int = 0
    single_stream_depth: int = 24
    mlp_ratio: float = 2.67
    conditioning_scheme: str = "fused"
    double_stream_modulation: str = "none"
    single_stream_modulation: str = "layer"
    ffn_type: str = "gated"
    qkv_bias: bool = True
    rope_centered_grid: bool = True
    # Ablation flags from the 2026-09-24 spike investigation; both default off,
    # i.e. off is the shipped recipe. See notes/muon_weight_growth_0924.md.
    branch_norm: bool = False
    cond_norm: bool = False


@dataclass(frozen=True)
class TextEncoderConfig:
    """Frozen encoder used for conditioning features."""

    path: str = ""
    exit_layer: int = 20


@dataclass(frozen=True)
class OptimConfig:
    """Learning rates and the Muon plus auxiliary AdamW schedule."""

    learning_rate: float = 3e-4
    start_learning_rate: float = 1e-5
    min_learning_rate: float = 0.5e-4
    lr_scheduler_type: str = "linear_cosine"
    lr_warmup_steps: int = 500
    max_grad_norm: float = 1.0
    # Skip the optimizer step entirely when the pre-clip gradient norm
    # exceeds this value (0 disables). This guard does not diagnose the
    # cause or establish stability; repeated skips can freeze learning.
    grad_spike_skip: float = 0.0
    muon_lr: float = 0.02
    muon_wd: float = 0.01
    muon_momentum: float = 0.95


@dataclass(frozen=True)
class TrainLoopConfig:
    """How long the loop runs, and how it accumulates, weights and averages."""

    max_steps: int = 50000
    # Clean-exit step boundary for a stage that must stop before the schedule
    # horizon: the loop checkpoints and exits at stop_at_step, while the LR
    # and caption-length schedules keep max_steps as their horizon so a
    # follow-up stage resumes mid-schedule. 0 disables it.
    stop_at_step: int = 0
    seed: int = 42
    gradient_accumulation_steps: int = 16
    num_workers: int = 8
    use_ema: bool = True
    ema_decay: float = 0.999
    # Bias-corrected EMA schedule: decay_t = min(ema_decay, (1+t)/(10+t))
    # (ADM/EDM warmup). Early steps average over a short window so the EMA
    # tracks the live weights instead of initialization residue; pure function
    # of the step, so crash-resume reproduces it exactly.
    ema_decay_warmup: bool = False
    ema_update_interval: int = 1
    use_logit_normal_sampling: bool = True
    logit_normal_mu: float = 0.0
    logit_normal_sigma: float = 1.0
    checkpoint_interval: int = 500
    # Zero keeps all checkpoints. Positive values retain this many complete
    # checkpoints per run; completed resolution endpoints stay in their runs.
    checkpoint_keep_last: int = 0
    eval_interval: int = 100000
    steady_state_skip_steps: int = 50
    # Per-sample loss weight as a function of the caption's retained length:
    # "none" (every weight 1.0, the default) or "log2" for
    # max(1, log2(L / caption_loss_weight_reference)).  A curve the trainer
    # does not know is a configuration error, not something to repair on the
    # way in.  See src/pretrain/caption_loss_weights.py for what the weights act
    # on, what they are normalized by, and why.
    caption_loss_weight_curve: str = "none"
    caption_loss_weight_reference: int = 128


@dataclass(frozen=True)
class EvalConfig:
    """Probes used to compare runs."""

    dataset_path: str = ""
    batch_size: int = 8
    loss_interval: int = 50
    loss_samples: int = 512
    prompts_file: str = "assets/eval/prompts_v1.jsonl"
    ode_steps: int = 50
    # Absolute global steps: stage baseline, +2k transition check, endpoint.
    grid_steps: list[int] = field(default_factory=list)
    # End of global max_steps, not an intermediate stop_at_step boundary.
    kid_at_end: bool = False
    kid_num_fake: int = 2000


@dataclass(frozen=True)
class PathConfig:
    """Model and output locations."""

    vae: str = ""
    output_dir: str = "output"


@dataclass(frozen=True)
class TelemetryConfig:
    """Logging and allocator housekeeping."""

    log_interval: int = 25
    # Zero disables periodic garbage collection / CUDA cache release.
    cache_clear_interval: int = 100
    swanlab_project: str = "artflow"
    # Steps between model-internal health reads (update/weight ratio, QK
    # gains, EMA distance). 0 disables the probe.
    health_interval: int = 250
    # Fixed-panel conditioning/update probe on rank zero. 0 disables it.
    # Samples four examples at three timesteps; also writes stability.jsonl.
    stability_interval: int = 0


@dataclass(frozen=True)
class TrainConfig:
    """The complete experiment definition."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)


# Config keys whose flat name differs from the field name, because the flat
# name is what the training loop reads. Everything else keeps its field name.
_RENAMES = {
    ("data", "mix"): "dataset_mix",
    ("text_encoder", "path"): "text_encoder_path",
    ("text_encoder", "exit_layer"): "text_encoder_exit_layer",
    ("eval", "dataset_path"): "eval_dataset_path",
    ("eval", "batch_size"): "eval_batch_size",
    ("eval", "loss_interval"): "eval_loss_interval",
    ("eval", "loss_samples"): "eval_loss_samples",
    ("eval", "kid_at_end"): "kid_eval_at_end",
    ("paths", "vae"): "vae_path",
    ("telemetry", "log_interval"): "telemetry_log_interval",
}


def _merge(base: Dict[str, Any], override: Mapping[str, Any], origin: str) -> Dict[str, Any]:
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"{origin}: unknown config section [{key}]")
        if not isinstance(value, Mapping):
            raise ValueError(f"{origin}: [{key}] must be a table")
        for sub_key, sub_value in value.items():
            if sub_key not in base[key]:
                raise ValueError(f"{origin}: unknown key [{key}].{sub_key}")
            if isinstance(base[key][sub_key], bool) != isinstance(sub_value, bool):
                raise ValueError(
                    f"{origin}: [{key}].{sub_key} must be "
                    f"{'a boolean' if isinstance(base[key][sub_key], bool) else 'a number or string'}"
                )
            base[key][sub_key] = sub_value
    return base


def load_config(paths: Sequence[str]) -> TrainConfig:
    """Load and merge config files in order; later files override earlier ones."""
    if not paths:
        raise ValueError("at least one config file is required")

    merged = asdict(TrainConfig())
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            raise ValueError(f"config file not found: {path}")
        with file_path.open("rb") as handle:
            payload = tomllib.load(handle)
        merged = _merge(merged, payload, str(file_path))

    config = TrainConfig(
        data=DataConfig(**merged["data"]),
        model=ModelConfig(**merged["model"]),
        text_encoder=TextEncoderConfig(**merged["text_encoder"]),
        optim=OptimConfig(**merged["optim"]),
        train=TrainLoopConfig(**merged["train"]),
        eval=EvalConfig(**merged["eval"]),
        paths=PathConfig(**merged["paths"]),
        telemetry=TelemetryConfig(**merged["telemetry"]),
    )
    _validate(config)
    return config


def _validate(config: TrainConfig) -> None:
    if type(config.telemetry.stability_interval) is not int or config.telemetry.stability_interval < 0:
        raise ValueError("[telemetry].stability_interval must be a nonnegative integer")
    if type(config.train.checkpoint_keep_last) is not int or config.train.checkpoint_keep_last < 0:
        raise ValueError("[train].checkpoint_keep_last must be a nonnegative integer")
    if type(config.telemetry.cache_clear_interval) is not int or config.telemetry.cache_clear_interval < 0:
        raise ValueError("[telemetry].cache_clear_interval must be a nonnegative integer")
    if type(config.eval.ode_steps) is not int or config.eval.ode_steps < 1:
        raise ValueError("[eval].ode_steps must be a positive integer")
    if not isinstance(config.eval.grid_steps, list) or any(
        type(step) is not int or step < 0 for step in config.eval.grid_steps
    ):
        raise ValueError("[eval].grid_steps must be a list of nonnegative integer global steps")
    if config.train.gradient_accumulation_steps < 1:
        raise ValueError("[train].gradient_accumulation_steps must be >= 1")
    stage_endpoint(config.train.max_steps, config.train.stop_at_step)
    # A loss-weight curve that is not usable is a configuration error, not
    # something to repair on the way in.
    CaptionLossWeights(
        curve=config.train.caption_loss_weight_curve,
        reference=config.train.caption_loss_weight_reference,
    )
    for name in ("curriculum_start", "curriculum_end"):
        value = getattr(config.data, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"[data].{name} must be within [0, 1]")
    if config.data.caption_policy not in ("legacy", "beta"):
        raise ValueError('[data].caption_policy must be "legacy" or "beta"')
    if config.data.caption_schedule not in ("linear", "stationary", "early"):
        raise ValueError('[data].caption_schedule must be "linear", "stationary" or "early"')
    if not 0.0 < config.data.caption_early_at <= 1.0:
        raise ValueError("[data].caption_early_at must be within (0, 1]")
    if not 0.0 <= config.data.caption_short_reserve <= 1.0:
        raise ValueError("[data].caption_short_reserve must be within [0, 1]")
    if config.data.caption_short_threshold < 1:
        raise ValueError("[data].caption_short_threshold must be positive")
    if not 0.0 <= config.data.caption_dropout_prob <= 1.0:
        raise ValueError("[data].caption_dropout_prob must be within [0, 1]")
    if config.model.hidden_size % config.model.num_heads:
        raise ValueError("[model].hidden_size must be divisible by num_heads")
    if not 0.0 <= config.train.ema_decay < 1.0:
        raise ValueError("[train].ema_decay must be within [0, 1)")
    if config.model.double_stream_depth + config.model.single_stream_depth < 1:
        raise ValueError("the model needs at least one transformer block")
    if config.text_encoder.exit_layer < 1:
        raise ValueError("[text_encoder].exit_layer must be >= 1")


def flatten(config: TrainConfig) -> Dict[str, Any]:
    """Return {flat name: value} for every config field.

    Section names are dropped; the handful of fields whose flat name differs
    are listed in ``_RENAMES``. The training loop reads these names directly.
    """
    flat: Dict[str, Any] = {}
    for section in fields(config):
        section_name = section.name
        section_value = getattr(config, section_name)
        for item in fields(section_value):
            flat[_RENAMES.get((section_name, item.name), item.name)] = getattr(
                section_value, item.name
            )
    return flat

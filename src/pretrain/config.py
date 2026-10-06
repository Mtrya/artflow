"""One explicit TOML recipe for a complete resolution curriculum.

Every dataclass field is required. There are no inherited configs, environment
substitutions, or hyperparameter overrides. Selecting a stage only projects its
inputs into the trainer; the optimizer and caption schedules retain the final
stage's endpoint as their horizon.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints


@dataclass(frozen=True)
class DataConfig:
    caption_dropout_prob: float
    curriculum_start: float
    curriculum_end: float
    caption_beta_start: float
    caption_beta_end: float
    caption_short_reserve: float
    caption_short_threshold: int


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_heads: int
    double_stream_depth: int
    single_stream_depth: int
    mlp_ratio: float


@dataclass(frozen=True)
class TextEncoderConfig:
    path: str
    exit_layer: int


@dataclass(frozen=True)
class OptimConfig:
    learning_rate: float
    start_learning_rate: float
    min_learning_rate: float
    lr_warmup_steps: int
    max_grad_norm: float
    adam_wd: float
    adam_conditioning_wd: float
    adam_eps: float
    adam_betas: list[float]
    muon_lr: float
    muon_wd: float
    muon_momentum: float


@dataclass(frozen=True)
class TrainLoopConfig:
    run_name: str
    seed: int
    num_workers: int
    ema_decay: float
    ema_update_interval: int
    logit_normal_mu: float
    logit_normal_sigma: float
    checkpoint_interval: int
    checkpoint_keep_last: int
    eval_interval: int
    steady_state_skip_steps: int
    caption_loss_weight_reference: int


@dataclass(frozen=True)
class EvalConfig:
    batch_size: int
    loss_interval: int
    loss_samples: int
    prompts_file: str
    ode_steps: int
    kid_num_fake: int


@dataclass(frozen=True)
class PathConfig:
    storage_root: str
    vae: str
    output_dir: str


@dataclass(frozen=True)
class TelemetryConfig:
    log_interval: int
    swanlab_project: str
    health_interval: int
    stability_interval: int


@dataclass(frozen=True)
class DatasetConfig:
    path: str
    weight: float


@dataclass(frozen=True)
class StageConfig:
    name: str
    end_step: int
    gradient_accumulation_steps: int
    bucket_plan: str
    eval_dataset_path: str
    grid_steps: list[int]
    datasets: list[DatasetConfig]


@dataclass(frozen=True)
class TrainConfig:
    data: DataConfig
    model: ModelConfig
    text_encoder: TextEncoderConfig
    optim: OptimConfig
    train: TrainLoopConfig
    eval: EvalConfig
    paths: PathConfig
    telemetry: TelemetryConfig
    stages: list[StageConfig]

    @property
    def max_steps(self) -> int:
        return self.stages[-1].end_step

    def stage(self, name: str) -> StageConfig:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise ValueError(
            f"unknown stage {name!r}; choose from {[s.name for s in self.stages]}"
        )

    def stage_start(self, name: str) -> int:
        index = self.stages.index(self.stage(name))
        return self.stages[index - 1].end_step if index else 0


def stage_caption_policy(config: TrainConfig, stage_name: str):
    """Use the trainer's global-step curriculum for this stage's exposure estimate."""
    from ..dataset.captions import CaptionPolicy

    data = config.data
    policy = CaptionPolicy(
        beta_start=data.caption_beta_start, beta_end=data.caption_beta_end,
        short_reserve=data.caption_short_reserve,
        short_threshold=data.caption_short_threshold,
    )
    interval = tuple(
        data.curriculum_start + (data.curriculum_end - data.curriculum_start)
        * step / config.max_steps
        for step in (config.stage_start(stage_name), config.stage(stage_name).end_step)
    )
    return policy, interval


def _decode(kind, value, location):
    if is_dataclass(kind):
        if not isinstance(value, dict):
            raise ValueError(f"{location}: expected a table")
        types = get_type_hints(kind)
        unknown, missing = value.keys() - types.keys(), types.keys() - value.keys()
        if unknown:
            raise ValueError(f"{location}: unknown fields {sorted(unknown)}")
        if missing:
            raise ValueError(f"{location}: missing required fields {sorted(missing)}")
        return kind(
            **{
                key: _decode(typ, value[key], f"{location}.{key}")
                for key, typ in types.items()
            }
        )
    if get_origin(kind) is list:
        if not isinstance(value, list):
            raise ValueError(f"{location}: expected a list")
        return [
            _decode(get_args(kind)[0], item, f"{location}[{i}]")
            for i, item in enumerate(value)
        ]
    if kind is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{location}: expected a finite number")
        return float(value)
    if type(value) is not kind:
        raise ValueError(f"{location}: expected {kind.__name__}")
    if kind is str and (not value.strip() or "${" in value):
        raise ValueError(
            f"{location}: use a nonempty explicit value, not an environment template"
        )
    return value


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def repository_config_path(value: str) -> Path:
    """Resolve tracked configuration assets from the repository, never the CWD."""
    path = Path(value)
    resolved = (REPOSITORY_ROOT / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(REPOSITORY_ROOT / "configs"):
        raise ValueError(f"tracked asset must be repository-relative under configs/: {value}")
    return resolved


def load_config(path: str | Path, *, storage_root: str | Path) -> TrainConfig:
    """Resolve heavy artifacts under the explicit root and tracked assets in configs/."""
    if not isinstance(path, (str, Path)):
        raise ValueError(
            "exactly one config path is required; config layering is unsupported"
        )
    path = Path(path).resolve()
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
            if "storage_root" in payload.get("paths", {}):
                raise ValueError("supply storage_root with --storage-root, not in the recipe")
            payload.setdefault("paths", {})["storage_root"] = str(Path(storage_root).resolve())
            config = _decode(TrainConfig, payload, str(path))
    except OSError as exc:
        raise ValueError(f"cannot read config {path}: {exc}") from exc
    _validate(config)
    # Path resolution changes locations only, never fills in missing settings.
    payload = asdict(config)
    storage = Path(config.paths.storage_root)
    if storage.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("storage root must be external to the repository")

    def heavy_path(value):
        resolved = storage / value
        if Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(f"heavy artifact must be relative to storage root: {value}")
        return str(resolved)

    payload["paths"]["storage_root"] = str(storage)
    for key in ("vae", "output_dir"):
        payload["paths"][key] = heavy_path(payload["paths"][key])
    payload["text_encoder"]["path"] = heavy_path(config.text_encoder.path)
    payload["eval"]["prompts_file"] = str(
        repository_config_path(config.eval.prompts_file)
    )
    for stage in payload["stages"]:
        stage["bucket_plan"] = str(repository_config_path(stage["bucket_plan"]))
        stage["eval_dataset_path"] = heavy_path(stage["eval_dataset_path"])
        for dataset in stage["datasets"]:
            dataset["path"] = heavy_path(dataset["path"])
    return _decode(TrainConfig, payload, str(path))


def _validate(config: TrainConfig) -> None:
    def require(test, message):
        if not test:
            raise ValueError(message)

    m, o, t, d, e = config.model, config.optim, config.train, config.data, config.eval
    require(m.hidden_size > 0 and m.num_heads > 0, "model dimensions must be positive")
    require(
        m.hidden_size % (4 * m.num_heads) == 0,
        "model.hidden_size must be divisible by 4 * num_heads for two RoPE axes",
    )
    require(
        m.double_stream_depth >= 0
        and m.single_stream_depth >= 0
        and m.double_stream_depth + m.single_stream_depth > 0,
        "invalid model depths",
    )
    require(m.mlp_ratio > 0, "model.mlp_ratio must be positive")
    require(
        config.text_encoder.exit_layer > 0, "text_encoder.exit_layer must be positive"
    )
    require(bool(config.stages), "at least one stage is required")
    names = [s.name for s in config.stages]
    require(len(names) == len(set(names)), "stage names must be unique")
    previous = 0
    for stage in config.stages:
        require(
            stage.name.replace("-", "").replace("_", "").isalnum(), "invalid stage name"
        )
        require(
            stage.end_step > previous, "stage end_step values must strictly increase"
        )
        require(
            stage.gradient_accumulation_steps > 0,
            "gradient_accumulation_steps must be positive",
        )
        require(
            bool(stage.datasets) and all(d.weight > 0 for d in stage.datasets),
            f"{stage.name}: datasets must have positive weights",
        )
        require(
            len({d.path for d in stage.datasets}) == len(stage.datasets),
            f"{stage.name}: duplicate dataset paths",
        )
        require(
            all(previous <= step <= stage.end_step for step in stage.grid_steps),
            f"{stage.name}: grid_steps must lie within this stage",
        )
        previous = stage.end_step
    require(
        o.learning_rate > 0 and o.muon_lr > 0,
        "optimizer learning rates must be positive",
    )
    require(
        0 <= o.start_learning_rate <= o.learning_rate
        and 0 <= o.min_learning_rate <= o.learning_rate,
        "invalid learning rate schedule bounds",
    )
    require(
        0 <= o.lr_warmup_steps < config.max_steps, "warmup must be shorter than the run"
    )
    require(
        o.max_grad_norm > 0
        and min(o.muon_wd, o.adam_wd, o.adam_conditioning_wd) >= 0
        and o.adam_eps > 0,
        "invalid clipping, decay or Adam epsilon",
    )
    require(
        len(o.adam_betas) == 2 and all(0 <= b < 1 for b in o.adam_betas),
        "invalid adam_betas",
    )
    require(0 <= o.muon_momentum < 1, "muon_momentum must be within [0, 1)")
    for key in (
        "caption_dropout_prob",
        "curriculum_start",
        "curriculum_end",
        "caption_short_reserve",
    ):
        require(0 <= getattr(d, key) <= 1, f"data.{key} must be within [0, 1]")
    require(d.caption_short_threshold > 0, "caption_short_threshold must be positive")
    require(0 <= t.ema_decay < 1, "ema_decay must be within [0, 1)")
    require(0 <= t.seed < 2**32, "seed must fit an unsigned 32-bit integer")
    require(t.logit_normal_sigma > 0, "logit_normal_sigma must be positive")
    require(
        t.caption_loss_weight_reference >= 2,
        "caption_loss_weight_reference must be >= 2",
    )
    require(
        Path(t.run_name).name == t.run_name and t.run_name not in (".", ".."),
        "invalid run_name",
    )
    for section, keys in (
        (
            t,
            (
                "num_workers",
                "checkpoint_keep_last",
                "eval_interval",
                "steady_state_skip_steps",
            ),
        ),
        (e, ("loss_interval", "kid_num_fake")),
        (config.telemetry, ("health_interval", "stability_interval")),
    ):
        for key in keys:
            require(getattr(section, key) >= 0, f"{key} must be nonnegative")
    for section, keys in (
        (t, ("checkpoint_interval", "ema_update_interval")),
        (e, ("batch_size", "loss_samples", "ode_steps")),
        (config.telemetry, ("log_interval",)),
    ):
        for key in keys:
            require(getattr(section, key) > 0, f"{key} must be positive")


_RENAMES = {
    ("text_encoder", "path"): "text_encoder_path",
    ("text_encoder", "exit_layer"): "text_encoder_exit_layer",
    ("eval", "batch_size"): "eval_batch_size",
    ("eval", "loss_interval"): "eval_loss_interval",
    ("eval", "loss_samples"): "eval_loss_samples",
    ("paths", "vae"): "vae_path",
    ("telemetry", "log_interval"): "telemetry_log_interval",
}


def flatten(config: TrainConfig, stage_name: str) -> dict[str, Any]:
    """Project the selected stage for the loop without overriding any values."""
    flat = {}
    for section in fields(config):
        if section.name == "stages":
            continue
        for key, value in asdict(getattr(config, section.name)).items():
            flat[_RENAMES.get((section.name, key), key)] = value
    stage = config.stage(stage_name)
    flat.update(asdict(stage))
    flat.pop("datasets")
    flat.pop("name")
    flat.pop("end_step")
    # parse_dataset_mix is also used by offline tooling. Keep its string
    # representation at this boundary; the run file stores structured entries.
    import shlex

    flat["dataset_mix"] = shlex.join(f"{d.path}:{d.weight}" for d in stage.datasets)
    # One run directory and one SwanLab experiment span every stage; stage
    # transitions resume the checkpoint-owned SwanLab identity.
    flat["run_name"] = config.train.run_name
    flat["max_steps"] = config.max_steps
    flat["stop_at_step"] = stage.end_step
    flat["stage_start"] = config.stage_start(stage_name)
    return flat

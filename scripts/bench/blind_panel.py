"""Blind side-by-side review of checkpoints from several training arms.

An *arm* is one training configuration.  This tool turns the checkpoints of
several arms into the material for a human preference review without revealing
which arm made which image:

1. ``generate`` samples the fixed prompt suite from one checkpoint, one PNG per
   prompt.  Sampling follows the training-time prompt grid
   (``src.evaluation.prompt_grid``) exactly - per-prompt seed, Euler solver at
   50 steps, the VAE's own decoding statistics - so an image matches the grid
   image of the same checkpoint and prompt.
2. ``panel`` collects the generated directories of several arms and writes one
   side-by-side image per prompt with the panels numbered 1..N (no arm names),
   an ``answer_key.json`` that maps each number back to its arm, a
   ``ballot.json`` template to fill in, and an ``index.md`` that shows the
   prompt text next to each comparison.
3. ``tally`` reads the completed ballot and reports how many prompts each arm
   won, with ties counted separately.

The panel numbering is shuffled independently per prompt, seeded from
``--seed`` and the prompt id, so the same inputs and seed always produce the
same blinding; keep ``answer_key.json`` closed until the ballot is complete.

Usage:
    python scripts/bench/blind_panel.py generate \
        --checkpoint <run>/checkpoint_step_000500 --out review/arm-a \
        --config configs/base.toml --config <run>.toml
    python scripts/bench/blind_panel.py panel \
        --arm a=review/arm-a --arm b=review/arm-b --out review/panel --seed 7
    python scripts/bench/blind_panel.py tally \
        --answer-key review/panel/answer_key.json --ballot review/panel/ballot.json
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Importable both as ``python -m scripts.bench.blind_panel`` and as a plain
# ``python scripts/bench/blind_panel.py``: the latter puts the script's own
# directory on sys.path, which does not expose ``src.*``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_PROMPTS = "assets/eval/prompts_v1.jsonl"
TIE = "tie"

_CHECKPOINT_STEP = re.compile(r"checkpoint_step_(\d+)")


# ---------------------------------------------------------------------------
# Review material (pure CPU: no torch, no GPU)
# ---------------------------------------------------------------------------


def parse_arms(specs: List[str]) -> List[Tuple[str, Path]]:
    """Parse repeated ``--arm name=directory`` into ``(name, path)`` pairs."""
    arms: List[Tuple[str, Path]] = []
    names = set()
    for spec in specs:
        name, separator, directory = spec.partition("=")
        name = name.strip()
        if not separator or not name or not directory:
            raise ValueError(f"--arm must be name=dir, got {spec!r}")
        if name in names:
            raise ValueError(f"duplicate arm name {name!r}")
        names.add(name)
        arms.append((name, Path(directory)))
    if len(arms) < 2:
        raise ValueError("a blind comparison needs at least two arms")
    return arms


def prompt_ids_for_arms(
    arms: List[Tuple[str, Path]],
) -> Tuple[List[str], Dict[str, Dict[str, Path]]]:
    """The prompt ids shared by the arms and each arm's image for every id.

    Every arm must hold one PNG per prompt id; a prompt missing from any arm is
    an error rather than a silently dropped comparison.
    """
    per_arm: List[Tuple[str, Dict[str, Path]]] = []
    for name, directory in arms:
        if not directory.is_dir():
            raise ValueError(f"arm {name!r}: {directory} is not a directory")
        per_arm.append(
            (name, {path.stem: path for path in sorted(directory.glob("*.png"))})
        )

    prompt_ids = sorted({stem for _, images in per_arm for stem in images})
    if not prompt_ids:
        raise ValueError(
            "no PNGs found in " + ", ".join(str(directory) for _, directory in arms)
        )
    for name, images in per_arm:
        missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in images]
        if missing:
            preview = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
            raise ValueError(
                f"arm {name!r} is missing {len(missing)} prompt(s): {preview}"
            )
    return prompt_ids, dict(per_arm)


def shuffled_arms(
    arms: List[Tuple[str, Path]], prompt_id: str, seed: int
) -> List[str]:
    """The arm names in the order they appear as labels 1..N for one prompt.

    Seeding from ``(seed, prompt_id)`` makes each prompt's shuffle independent
    of the other prompts and reproducible across runs, whatever order the
    prompts are processed in.
    """
    order = [name for name, _ in arms]
    random.Random(f"{seed}:{prompt_id}").shuffle(order)
    return order


def _label_font(size: int):
    for candidate in (
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def compose_panel(image_paths: List[Path], labels: List[str], out_path: Path) -> None:
    """Write one side-by-side image: panels left to right, each numbered.

    Panels are pasted at their own size on a white canvas whose slots are as
    large as the largest panel, so an arm that produced a different resolution
    stays visible as such instead of being hidden by a resize.
    """
    panels = [Image.open(path).convert("RGB") for path in image_paths]
    slot_width = max(panel.width for panel in panels)
    slot_height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (slot_width * len(panels), slot_height), "white")
    draw = ImageDraw.Draw(canvas)

    size = max(16, min(slot_width, slot_height) // 12)
    font = _label_font(size)
    padding = max(4, size // 3)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        left = index * slot_width + (slot_width - panel.width) // 2
        canvas.paste(panel, (left, 0))

        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        x0 = index * slot_width + padding
        draw.rectangle(
            [x0, padding, x0 + text_width + 2 * padding, padding + text_height + 2 * padding],
            fill="black",
        )
        draw.text(
            (x0 + padding, padding + padding - box[1]),
            label,
            fill="white",
            font=font,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    for panel in panels:
        panel.close()


def load_prompt_records(path: str) -> Dict[str, Dict[str, Any]]:
    """``{prompt id: record}`` for the fixed prompt suite (JSONL)."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"prompt suite not found: {source}")

    from src.evaluation.prompt_grid import load_prompt_suite

    return {record["id"]: record for record in load_prompt_suite(str(source))}


def panel_choices(arm_count: int) -> List[str]:
    """The panel numbers shown to the reviewer, as strings."""
    return [str(number) for number in range(1, arm_count + 1)]


def cmd_panel(args: argparse.Namespace) -> int:
    arms = parse_arms(args.arm)
    prompt_ids, images_by_arm = prompt_ids_for_arms(arms)

    suite = load_prompt_records(args.prompts)
    missing_text = [prompt_id for prompt_id in prompt_ids if prompt_id not in suite]
    if missing_text:
        preview = ", ".join(missing_text[:5]) + ("..." if len(missing_text) > 5 else "")
        raise ValueError(
            f"{args.prompts} has no text for {len(missing_text)} generated "
            f"prompt(s): {preview}"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    choices = panel_choices(len(arms))
    answer_key: Dict[str, Dict[str, str]] = {}
    ballot: Dict[str, Optional[str]] = {}
    sections: List[str] = []
    for prompt_id in prompt_ids:
        order = shuffled_arms(arms, prompt_id, args.seed)
        compose_panel(
            [images_by_arm[name][prompt_id] for name in order],
            choices,
            out_dir / f"{prompt_id}.png",
        )
        answer_key[prompt_id] = dict(zip(choices, order))
        ballot[prompt_id] = None

        record = suite[prompt_id]
        lines = [
            f"## {prompt_id}",
            "",
            f"![{prompt_id}]({prompt_id}.png)",
            "",
            "**Prompt:** " + " ".join(str(record.get("text", "")).split()),
        ]
        metadata = []
        if record.get("tags"):
            metadata.append(", ".join(str(tag) for tag in record["tags"]))
        if record.get("domain"):
            metadata.append(str(record["domain"]))
        if record.get("aspect"):
            metadata.append(f"aspect {record['aspect']}")
        if metadata:
            lines += ["", "**Metadata:** " + " · ".join(metadata)]
        lines += [
            "",
            "Choose one in `ballot.json`: " + " / ".join(choices) + f" / {TIE}",
            "",
        ]
        lines += [f"- {choice}:" for choice in choices + [TIE]]
        lines.append("")
        sections.append("\n".join(lines))

    header = [
        "# Blind prompt panel",
        "",
        "Each image shows the same prompt generated by every arm, panels numbered",
        "left to right. The numbering is shuffled independently per prompt, so the",
        "same number does not mean the same arm in different prompts.",
        "",
        "Record one choice per prompt in `ballot.json` (a panel number or `tie`),",
        "then run `tally`. Do not read `answer_key.json` before the ballot is",
        "complete.",
        "",
    ]
    (out_dir / "answer_key.json").write_text(
        json.dumps(answer_key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "ballot.json").write_text(
        json.dumps(ballot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "index.md").write_text("\n".join(header + sections), encoding="utf-8")

    print(f"Wrote {len(prompt_ids)} comparisons for {len(arms)} arms to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Counting the completed ballot
# ---------------------------------------------------------------------------


def tally_votes(
    answer_key: Dict[str, Dict[str, str]], ballot: Dict[str, Any]
) -> Dict[str, Any]:
    """Count each arm's wins and the ties from a completed ballot.

    ``answer_key`` maps a prompt id to ``{panel number: arm name}``; ``ballot``
    maps the same prompt ids to the reviewer's choice, a panel number or
    ``"tie"``.  A prompt that is missing from the ballot, still unfilled, or
    carrying a choice the answer key does not know is an error: a partially
    counted review would silently misstate the result.
    """
    missing = sorted(set(answer_key) - set(ballot))
    if missing:
        preview = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        raise ValueError(f"ballot has no entry for {len(missing)} prompt(s): {preview}")
    unknown = sorted(set(ballot) - set(answer_key))
    if unknown:
        preview = ", ".join(unknown[:5]) + ("..." if len(unknown) > 5 else "")
        raise ValueError(f"ballot names {len(unknown)} unknown prompt(s): {preview}")

    arms: List[str] = []
    for mapping in answer_key.values():
        for arm in mapping.values():
            if arm not in arms:
                arms.append(arm)
    wins = {arm: 0 for arm in arms}
    ties = 0
    per_prompt: Dict[str, str] = {}
    for prompt_id, mapping in answer_key.items():
        choice = ballot[prompt_id]
        if choice is None or str(choice).strip() == "":
            raise ValueError(f"ballot entry for {prompt_id!r} is empty")
        choice = str(choice).strip()
        if choice == TIE:
            ties += 1
            per_prompt[prompt_id] = TIE
        elif choice in mapping:
            wins[mapping[choice]] += 1
            per_prompt[prompt_id] = mapping[choice]
        else:
            raise ValueError(
                f"ballot entry for {prompt_id!r} is {choice!r}, expected one of "
                f"{', '.join(sorted(mapping))} or {TIE!r}"
            )
    return {"wins": wins, "ties": ties, "votes": len(answer_key), "per_prompt": per_prompt}


def cmd_tally(args: argparse.Namespace) -> int:
    answer_key = json.loads(Path(args.answer_key).read_text(encoding="utf-8"))
    ballot = json.loads(Path(args.ballot).read_text(encoding="utf-8"))
    result = tally_votes(answer_key, ballot)

    print("arm\twins")
    for arm, wins in result["wins"].items():
        print(f"{arm}\t{wins}")
    print(f"{TIE}\t{result['ties']}")
    print(f"votes\t{result['votes']}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


# ---------------------------------------------------------------------------
# GPU generation, on the training-time sampling path
# ---------------------------------------------------------------------------


def resolve_checkpoint(path: str) -> Path:
    """The weights file behind ``--checkpoint``: a file, or a dir's EMA weights."""
    checkpoint = Path(path)
    if checkpoint.is_dir():
        weights = checkpoint / "ema_weights.pt"
        if not weights.is_file():
            raise FileNotFoundError(
                f"{checkpoint} has no ema_weights.pt; pass the weights file directly"
            )
        return weights
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    return checkpoint


def checkpoint_step(path: str) -> Optional[int]:
    """Training step parsed from a ``checkpoint_step_<N>`` directory name."""
    match = _CHECKPOINT_STEP.search(str(path))
    return int(match.group(1)) if match else None


def cmd_generate(args: argparse.Namespace) -> int:
    import functools
    from datetime import datetime

    import torch
    from diffusers import AutoencoderKLQwenImage
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.evaluation.prompt_grid import (
        load_prompt_plan,
        prompt_seed,
        sample_prompt_images,
    )
    from src.models.artflow import ArtFlow
    from src.train.config import flatten, load_config
    from src.utils.vae_codec import get_vae_stats

    config = flatten(load_config(args.config))
    model_config = {
        "hidden_size": config["hidden_size"],
        "num_heads": config["num_heads"],
        "double_stream_depth": config["double_stream_depth"],
        "single_stream_depth": config["single_stream_depth"],
        "mlp_ratio": config["mlp_ratio"],
        "conditioning_scheme": config["conditioning_scheme"],
        "qkv_bias": config["qkv_bias"],
        "double_stream_modulation": config["double_stream_modulation"],
        "single_stream_modulation": config["single_stream_modulation"],
        "ffn_type": config["ffn_type"],
        "rope_centered_grid": config["rope_centered_grid"],
        # Fixed architecture constants, as in the training entry point.
        "patch_size": 2,
        "in_channels": 16,
        "txt_in_features": 1024,
    }
    vae_path = config["vae_path"]
    text_encoder_path = config["text_encoder_path"]
    exit_layer = config["text_encoder_exit_layer"]
    pooling = config["conditioning_scheme"] == "fused"
    dataset_path = args.dataset or config["eval_dataset_path"]
    if not dataset_path:
        raise ValueError(
            "no eval dataset: pass --dataset or set [eval].dataset_path in --config"
        )
    batch_size = args.batch_size or config["eval_batch_size"]

    weights = resolve_checkpoint(args.checkpoint)
    step = args.step if args.step is not None else checkpoint_step(weights)
    device = torch.device(args.device)

    print(f"Loading checkpoint {weights} (step {step}) onto {device}...")
    state_dict = torch.load(str(weights), map_location="cpu")
    if isinstance(state_dict, dict) and "module" in state_dict:
        state_dict = state_dict["module"]
    model = ArtFlow(**model_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    vae = AutoencoderKLQwenImage.from_pretrained(
        vae_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    vae_mean, vae_std = get_vae_stats(vae_path, device=device)
    vae_mean = vae_mean.to(dtype=torch.bfloat16)
    vae_std = vae_std.to(dtype=torch.bfloat16)

    text_encoder = AutoModelForCausalLM.from_pretrained(
        text_encoder_path,
        torch_dtype=torch.bfloat16,
        device_map=str(device),
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    text_encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)

    prompts, shapes = load_prompt_plan(args.prompts, dataset_path)
    if not prompts:
        raise ValueError(f"prompt suite is empty: {args.prompts}")

    amp_context = None
    if device.type == "cuda":
        amp_context = functools.partial(
            torch.autocast, device_type="cuda", dtype=torch.bfloat16
        )
    print(
        f"Sampling {len(prompts)} prompts ({args.ode_steps} solver steps, "
        f"batch size {batch_size})..."
    )
    results = sample_prompt_images(
        model,
        vae,
        vae_mean,
        vae_std,
        text_encoder,
        tokenizer,
        prompts,
        shapes,
        batch_size=batch_size,
        ode_steps=args.ode_steps,
        pooling=pooling,
        device=device,
        exit_layer=exit_layer,
        amp_context=amp_context,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for prompt, image in results:
        array = (
            image.permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0
        ).round().astype("uint8")
        Image.fromarray(array).save(out_dir / f"{prompt['id']}.png")

    manifest = {
        "checkpoint": str(weights),
        "step": step,
        "prompts_file": str(args.prompts),
        "eval_dataset": dataset_path,
        "solver": {
            "name": "euler",
            "ode_steps": args.ode_steps,
            "t_start": 0.0,
            "t_end": 1.0,
            "time_shift": "resolution-dependent, applied by sample_ode as in training",
        },
        "seed_rule": "int(md5(prompt_id)[:8], 16) -> torch.Generator(device='cpu')",
        "seeds": {prompt["id"]: prompt_seed(prompt["id"]) for prompt, _ in results},
        "prompts": {
            prompt["id"]: {
                "bucket": prompt["_bucket"],
                "latents": list(shapes[prompt["_bucket"]]),
            }
            for prompt, _ in results
        },
        "pooling": pooling,
        "text_encoder": {"path": text_encoder_path, "exit_layer": exit_layer},
        "vae": vae_path,
        "model_config": model_config,
        "batch_size": batch_size,
        "device": str(device),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(results)} PNGs and manifest.json to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blind_panel.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser(
        "generate",
        help="sample the fixed prompt suite from one checkpoint",
        description=(
            "Sample one PNG per prompt from a checkpoint, on the same sampling "
            "path as the training-time prompt grid."
        ),
    )
    generate.add_argument(
        "--checkpoint",
        required=True,
        help="checkpoint directory (uses its ema_weights.pt) or weights file",
    )
    generate.add_argument(
        "--out", required=True, help="output directory for <prompt_id>.png"
    )
    generate.add_argument(
        "--config",
        action="append",
        default=None,
        metavar="TOML",
        help="TOML config chain, later files win (default: configs/base.toml); "
        "pass the same chain the arm was trained with",
    )
    generate.add_argument(
        "--prompts",
        default=DEFAULT_PROMPTS,
        help=f"fixed prompt suite JSONL (default: {DEFAULT_PROMPTS})",
    )
    generate.add_argument(
        "--dataset",
        default=None,
        help="precomputed eval dataset whose latents define the resolutions "
        "(default: [eval].dataset_path from the config)",
    )
    generate.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="prompts per sampling batch (default: [eval].batch_size)",
    )
    generate.add_argument(
        "--ode-steps", type=int, default=50, help="solver steps (default: 50)"
    )
    generate.add_argument(
        "--step",
        type=int,
        default=None,
        help="training step recorded in the manifest (default: parsed from the path)",
    )
    generate.add_argument("--device", default="cuda:0")
    generate.set_defaults(func=cmd_generate)

    panel = sub.add_parser(
        "panel",
        help="build unlabelled side-by-side comparisons from several arms",
        description=(
            "Gather the generate outputs of several arms, shuffle the arms "
            "independently per prompt, and write the comparison images plus the "
            "answer key, ballot template and index."
        ),
    )
    panel.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="one arm's generate output directory; repeat per arm",
    )
    panel.add_argument("--out", required=True, help="output directory for the panel")
    panel.add_argument(
        "--seed", type=int, default=0, help="seed for the per-prompt shuffle"
    )
    panel.add_argument(
        "--prompts",
        default=DEFAULT_PROMPTS,
        help=f"prompt suite JSONL used for the prompt text (default: {DEFAULT_PROMPTS})",
    )
    panel.set_defaults(func=cmd_panel)

    tally = sub.add_parser(
        "tally",
        help="count wins from a completed ballot",
        description="Count each arm's wins and the ties from the completed ballot.",
    )
    tally.add_argument(
        "--answer-key", required=True, help="answer_key.json written by panel"
    )
    tally.add_argument("--ballot", required=True, help="completed ballot.json")
    tally.add_argument(
        "--out", default=None, help="optional JSON file for the count result"
    )
    tally.set_defaults(func=cmd_tally)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate" and args.config is None:
        args.config = ["configs/base.toml"]
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

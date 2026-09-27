"""Build blinded image comparisons and tally human preferences.

``panel`` reads <prompt_id>.png files from each named arm, shuffles their
positions per prompt using --seed, and writes comparison images, an answer
key, a ballot and an index. Keep the answer key closed while voting.
``tally`` reports wins and ties from the completed ballot.

Run from the repository root:
    python -m scripts.eval.blind_panel panel \
        --arm a=review/arm-a --arm b=review/arm-b --out review/panel --seed 7
    python -m scripts.eval.blind_panel tally \
        --answer-key review/panel/answer_key.json --ballot review/panel/ballot.json
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

DEFAULT_PROMPTS = "assets/eval/hero_monitor_v1.jsonl"
TIE = "tie"



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
# Command-line interface
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blind_panel.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    panel = sub.add_parser(
        "panel",
        help="build unlabelled side-by-side comparisons from several arms",
        description=(
            "Gather the image directories of several arms, shuffle the arms "
            "independently per prompt, and write the comparison images plus the "
            "answer key, ballot template and index."
        ),
    )
    panel.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="directory of <prompt_id>.png images; repeat per arm",
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
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

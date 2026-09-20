"""Resolve versioned hero input paths without regenerating weights or choosing T.

Templates preserve the exact shard weights used in the infra comparisons. Only
the explicit ARTFLOW_ROOT placeholder is substituted; no shell or arbitrary
environment expansion runs. Output must be a new path. The training config
loader itself deliberately does not interpret templates.
"""

import argparse
import json
from pathlib import Path
import tomllib


STAGES = ("256p", "640p", "896p")
REPO = Path(__file__).resolve().parents[2]


def render(stage, root):
    if stage not in STAGES:
        raise ValueError("unsupported hero resolution")
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("hero artifact root must be absolute")
    if any(char.isspace() for char in str(root)):
        raise ValueError("hero artifact root cannot contain whitespace in the dataset-mix syntax")
    text = (REPO / "configs" / "hero" / f"{stage}.toml.in").read_text()
    # The placeholder appears inside TOML basic strings; escape the substituted
    # path as a string fragment, not executable TOML or shell syntax.
    fragment = json.dumps(str(root), ensure_ascii=False)[1:-1]
    resolved = text.replace("${ARTFLOW_ROOT}", fragment)
    tomllib.loads(resolved)
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = render(args.stage, args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()

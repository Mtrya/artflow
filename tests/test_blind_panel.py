"""The blind panel's review material: shuffling, answer key and counting.

``generate`` is GPU code and is not exercised here (its sampling path lives in
``src.evaluation.prompt_grid``).  What is tested is everything around the
review: that a re-run with the same seed reproduces the same blinding, that the
answer key describes the order actually pasted into the comparison image, that
a ballot is counted to the right arm with ties counted separately, and that a
prompt missing from an arm or from the ballot is refused rather than silently
dropped.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.bench.blind_panel import (
    TIE,
    compose_panel,
    main,
    parse_arms,
    prompt_ids_for_arms,
    shuffled_arms,
    tally_votes,
)

ARM_COLORS = {"a": (255, 0, 0), "b": (0, 128, 0), "c": (0, 0, 255)}
IMAGE_SIZE = (64, 48)


def write_images(directory: Path, prompt_ids, color, size=IMAGE_SIZE) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for prompt_id in prompt_ids:
        Image.new("RGB", size, color).save(directory / f"{prompt_id}.png")


def write_suite(path: Path, prompt_ids) -> Path:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": prompt_id,
                    "text": f"prompt text for {prompt_id}",
                    "tags": ["style"],
                    "domain": "test",
                    "aspect": "4:3",
                }
            )
            + "\n"
            for prompt_id in prompt_ids
        ),
        encoding="utf-8",
    )
    return path


def write_arms(root: Path, prompt_ids, names="abc"):
    for name in names:
        write_images(root / f"arm-{name}", prompt_ids, ARM_COLORS[name])
    return root


def run_panel(root: Path, prompt_ids, out: Path, seed=3, names="abc") -> int:
    write_arms(root, prompt_ids, names)
    suite = write_suite(root / "prompts.jsonl", prompt_ids)
    argv = ["panel"]
    for name in names:
        argv += ["--arm", f"{name}={root / f'arm-{name}'}"]
    argv += ["--out", str(out), "--seed", str(seed), "--prompts", str(suite)]
    return main(argv)


# ---------------------------------------------------------------------------
# Shuffling
# ---------------------------------------------------------------------------


def test_arm_specs_parse_and_reject_malformed_or_duplicate(tmp_path):
    arms = parse_arms([f"a={tmp_path / 'a'}", f"b={tmp_path / 'b'}"])

    assert arms == [("a", tmp_path / "a"), ("b", tmp_path / "b")]
    with pytest.raises(ValueError, match="name=dir"):
        parse_arms(["just-a-name"])
    with pytest.raises(ValueError, match="duplicate"):
        parse_arms(["a=x", "a=y"])
    with pytest.raises(ValueError, match="at least two"):
        parse_arms(["a=x"])


def test_shuffle_is_a_reproducible_permutation_per_prompt():
    arms = [("a", Path("a")), ("b", Path("b")), ("c", Path("c"))]
    prompt_ids = [f"p{index}" for index in range(20)]

    first = [shuffled_arms(arms, prompt_id, seed=7) for prompt_id in prompt_ids]
    again = [shuffled_arms(arms, prompt_id, seed=7) for prompt_id in prompt_ids]
    other_seed = [shuffled_arms(arms, prompt_id, seed=8) for prompt_id in prompt_ids]

    assert first == again, "the same seed must reproduce the same blinding"
    for order in first:
        assert sorted(order) == ["a", "b", "c"]
    assert len({tuple(order) for order in first}) > 1, "per-prompt shuffles are independent"
    assert any(one != two for one, two in zip(first, other_seed))


# ---------------------------------------------------------------------------
# Building the material
# ---------------------------------------------------------------------------


def test_prompt_ids_require_every_arm_to_hold_every_prompt(tmp_path):
    write_images(tmp_path / "a", ["p1", "p2"], ARM_COLORS["a"])
    write_images(tmp_path / "b", ["p1"], ARM_COLORS["b"])
    arms = parse_arms([f"a={tmp_path / 'a'}", f"b={tmp_path / 'b'}"])

    with pytest.raises(ValueError, match="missing 1 prompt"):
        prompt_ids_for_arms(arms)

    write_images(tmp_path / "b", ["p2"], ARM_COLORS["b"])
    prompt_ids, per_arm = prompt_ids_for_arms(arms)

    assert prompt_ids == ["p1", "p2"]
    assert per_arm["a"]["p1"] == tmp_path / "a" / "p1.png"


def test_panel_writes_comparisons_answer_key_and_ballot(tmp_path):
    prompt_ids = ["scene-1", "scene-2", "scene-3"]

    assert run_panel(tmp_path, prompt_ids, tmp_path / "panel") == 0

    out = tmp_path / "panel"
    answer_key = json.loads((out / "answer_key.json").read_text(encoding="utf-8"))
    ballot = json.loads((out / "ballot.json").read_text(encoding="utf-8"))

    assert set(answer_key) == set(prompt_ids) == set(ballot)
    for prompt_id, mapping in answer_key.items():
        assert sorted(mapping) == ["1", "2", "3"]
        assert sorted(mapping.values()) == ["a", "b", "c"]
        assert ballot[prompt_id] is None
        with Image.open(out / f"{prompt_id}.png") as panel:
            assert panel.size == (3 * IMAGE_SIZE[0], IMAGE_SIZE[1])

    index = (out / "index.md").read_text(encoding="utf-8")
    for prompt_id in prompt_ids:
        assert f"## {prompt_id}" in index
        assert f"prompt text for {prompt_id}" in index
        assert f"]({prompt_id}.png)" in index
    assert "tie:" in index


def test_the_answer_key_describes_the_pasted_panel_order(tmp_path):
    """Each numbered panel must actually show the arm the answer key names."""
    prompt_ids = [f"p{index}" for index in range(5)]

    assert run_panel(tmp_path, prompt_ids, tmp_path / "panel") == 0

    answer_key = json.loads(
        (tmp_path / "panel" / "answer_key.json").read_text(encoding="utf-8")
    )
    color_to_arm = {color: name for name, color in ARM_COLORS.items()}
    for prompt_id in prompt_ids:
        with Image.open(tmp_path / "panel" / f"{prompt_id}.png") as panel:
            for label, arm in answer_key[prompt_id].items():
                slot = int(label) - 1
                center = (slot * IMAGE_SIZE[0] + IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2)
                assert panel.getpixel(center) == ARM_COLORS[arm]
                assert color_to_arm[panel.getpixel(center)] == arm


def test_the_same_seed_reproduces_the_same_blinding(tmp_path):
    prompt_ids = [f"p{index}" for index in range(6)]

    keys = []
    for run in ("first", "second"):
        out = tmp_path / run
        assert run_panel(tmp_path, prompt_ids, out, seed=11) == 0
        keys.append(json.loads((out / "answer_key.json").read_text(encoding="utf-8")))

    assert keys[0] == keys[1]


def test_panel_refuses_a_prompt_missing_from_an_arm(tmp_path, capsys):
    write_images(tmp_path / "arm-a", ["p1", "p2"], ARM_COLORS["a"])
    write_images(tmp_path / "arm-b", ["p1"], ARM_COLORS["b"])
    suite = write_suite(tmp_path / "prompts.jsonl", ["p1", "p2"])

    result = main([
        "panel",
        "--arm", f"a={tmp_path / 'arm-a'}",
        "--arm", f"b={tmp_path / 'arm-b'}",
        "--out", str(tmp_path / "panel"),
        "--prompts", str(suite),
    ])

    assert result == 2
    assert "missing 1 prompt" in capsys.readouterr().err
    assert not (tmp_path / "panel" / "answer_key.json").exists()


def test_panel_refuses_a_generated_prompt_without_suite_text(tmp_path, capsys):
    write_images(tmp_path / "arm-a", ["p1", "p2"], ARM_COLORS["a"])
    write_images(tmp_path / "arm-b", ["p1", "p2"], ARM_COLORS["b"])
    suite = write_suite(tmp_path / "prompts.jsonl", ["p1"])

    result = main([
        "panel",
        "--arm", f"a={tmp_path / 'arm-a'}",
        "--arm", f"b={tmp_path / 'arm-b'}",
        "--out", str(tmp_path / "panel"),
        "--prompts", str(suite),
    ])

    assert result == 2
    assert "no text" in capsys.readouterr().err


def test_compose_panel_numbers_each_slot(tmp_path):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    Image.new("RGB", (40, 30), "white").save(left)
    Image.new("RGB", (20, 20), "black").save(right)

    out = tmp_path / "panel.png"
    compose_panel([left, right], ["1", "2"], out)

    with Image.open(out) as panel:
        assert panel.size == (80, 30)
        # The narrower panel is centred in its slot, not stretched: white
        # margin at 42, black panel from 50 to 69, white again at 70.
        assert panel.getpixel((42, 15)) == (255, 255, 255)
        assert panel.getpixel((66, 15)) == (0, 0, 0)
        assert panel.getpixel((69, 15)) == (0, 0, 0)
        assert panel.getpixel((70, 15)) == (255, 255, 255)
        # A number badge marks the top-left corner of each slot.
        assert panel.getpixel((6, 6)) == (0, 0, 0)
        assert panel.getpixel((40 + 6, 6)) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Counting the ballot
# ---------------------------------------------------------------------------


def test_tally_counts_wins_and_ties():
    answer_key = {
        "p1": {"1": "a", "2": "b", "3": "c"},
        "p2": {"1": "b", "2": "a", "3": "c"},
        "p3": {"1": "c", "2": "b", "3": "a"},
        "p4": {"1": "a", "2": "c", "3": "b"},
    }
    ballot = {"p1": "1", "p2": "1", "p3": TIE, "p4": "1"}

    result = tally_votes(answer_key, ballot)

    assert result["wins"] == {"a": 2, "b": 1, "c": 0}
    assert result["ties"] == 1
    assert result["votes"] == 4
    assert result["per_prompt"] == {"p1": "a", "p2": "b", "p3": TIE, "p4": "a"}


def test_tally_refuses_missing_unfilled_unknown_and_foreign_votes():
    answer_key = {"p1": {"1": "a", "2": "b"}, "p2": {"1": "b", "2": "a"}}

    with pytest.raises(ValueError, match="no entry"):
        tally_votes(answer_key, {"p1": "1"})
    with pytest.raises(ValueError, match="empty"):
        tally_votes(answer_key, {"p1": "1", "p2": None})
    with pytest.raises(ValueError, match="expected one of"):
        tally_votes(answer_key, {"p1": "9", "p2": TIE})
    with pytest.raises(ValueError, match="unknown prompt"):
        tally_votes(answer_key, {"p1": "1", "p2": "1", "p3": "1"})


def test_tally_reports_through_the_command_line(tmp_path, capsys):
    answer_key = {"p1": {"1": "a", "2": "b"}, "p2": {"1": "b", "2": "a"}}
    ballot = {"p1": "2", "p2": TIE}
    key_path = tmp_path / "answer_key.json"
    ballot_path = tmp_path / "ballot.json"
    key_path.write_text(json.dumps(answer_key), encoding="utf-8")
    ballot_path.write_text(json.dumps(ballot), encoding="utf-8")

    out = tmp_path / "result.json"
    assert main([
        "tally",
        "--answer-key", str(key_path),
        "--ballot", str(ballot_path),
        "--out", str(out),
    ]) == 0

    stdout = capsys.readouterr().out
    assert "a\t0" in stdout
    assert "b\t1" in stdout
    assert f"{TIE}\t1" in stdout
    assert json.loads(out.read_text(encoding="utf-8"))["wins"] == {"a": 0, "b": 1}

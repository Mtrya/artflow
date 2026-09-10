from collections import Counter, defaultdict

from scripts.data.synth_assign_models import STRATA, assign

MODELS = ["ernie-image-turbo", "qwen-image"]


def make_rows(count):
    """Grid rows built the way the real one is: the frame shape cycles with the
    row number, everything else looks random."""
    rows = []
    languages = ("zh", "en")
    families = ("photograph", "painting")
    groups = ("hanfu", "minority", "neighbour", "modern")
    aspects = ("1x1", "3x2", "2x3", "16x9", "9x16")
    for index in range(count):
        rows.append({
            "prompt_id": f"syn-{index:06d}",
            "language": languages[(index // 7) % 2],
            "family": families[(index // 13) % 2],
            "subject_group": groups[(index // 3) % 4],
            "aspect": aspects[index % 5],
        })
    return rows


def shares(counter, models):
    total = sum(counter.values())
    return [counter[name] / total for name in models]


def test_every_row_gets_one_of_the_generators():
    rows = assign(make_rows(1000), MODELS)
    assert {row["model"] for row in rows} == set(MODELS)


def test_split_is_even_inside_every_grid_cell():
    rows = assign(make_rows(1000), MODELS)
    cells = defaultdict(Counter)
    for row in rows:
        cells[tuple(row[key] for key in STRATA)][row["model"]] += 1
    for key, counter in cells.items():
        per_model = [counter[name] for name in MODELS]
        assert max(per_model) - min(per_model) <= 1, (key, per_model)


def test_no_slot_is_tied_to_a_generator():
    # The frame shape cycles with the row number, so handing out generators by
    # row parity would give one of them most of the square frames.  Every slot
    # value has to come out within a couple of points of an even split.
    rows = assign(make_rows(2000), MODELS)
    for key in STRATA:
        for value in {row[key] for row in rows}:
            counter = Counter(row["model"] for row in rows if row[key] == value)
            assert len(counter) == len(MODELS), (key, value)
            for share in shares(counter, MODELS):
                assert 0.45 <= share <= 0.55, (key, value, share, dict(counter))


def test_set_of_rows_is_split_evenly():
    rows = assign(make_rows(2000), MODELS)
    counter = Counter(row["model"] for row in rows)
    assert max(counter.values()) - min(counter.values()) <= 0.01 * len(rows)


def test_three_generators_take_equal_turns():
    models = ["a", "b", "c"]
    rows = assign(make_rows(600), models)
    counter = Counter(row["model"] for row in rows)
    for share in shares(counter, models):
        assert abs(share - 1 / 3) <= 0.03

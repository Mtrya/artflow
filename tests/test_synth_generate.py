from scripts.data.synth_generate import RECIPES, select, snap_resolution

SUPPORTED = [(1024, 1024), (1264, 848), (848, 1264), (1376, 768), (768, 1376)]


def grid(count):
    return [{"prompt_id": f"syn-{index:06d}"} for index in range(count)]


def test_render_frame_is_pulled_to_the_closest_trained_aspect():
    assert snap_resolution(1024, 1024, SUPPORTED) == (1024, 1024)
    assert snap_resolution(896, 1152, SUPPORTED) == (848, 1264)
    assert snap_resolution(1152, 896, SUPPORTED) == (1264, 848)
    assert snap_resolution(768, 1344, SUPPORTED) == (768, 1376)
    assert snap_resolution(1344, 768, SUPPORTED) == (1376, 768)


def test_every_recipe_names_a_pipeline_and_a_schedule():
    for name, recipe in RECIPES.items():
        assert recipe["pipeline"], name
        assert recipe["steps"] > 0, name
        assert isinstance(recipe.get("extra", {}), dict), name


def test_shards_cover_the_grid_exactly_once():
    rows = grid(1000)
    picked = [row["prompt_id"] for shard in range(4)
              for row in select(rows, shard, 4, 0, 0)]
    assert len(picked) == 1000
    assert len(set(picked)) == 1000


def test_different_shards_get_different_rows():
    # Taking the stride from the offset alone gives every worker the same rows.
    rows = grid(100)
    first = [row["prompt_id"] for row in select(rows, 0, 4, 0, 0)]
    second = [row["prompt_id"] for row in select(rows, 1, 4, 0, 0)]
    assert first and second
    assert set(first) & set(second) == set()


def test_offset_moves_the_stretch_and_still_splits_it():
    rows = grid(100)
    for shard in range(4):
        picked = {row["prompt_id"] for row in select(rows, shard, 4, 50, 0)}
        assert picked
        assert all(int(name.split("-")[1]) >= 50 for name in picked)
    everything = [row["prompt_id"] for shard in range(4)
                  for row in select(rows, shard, 4, 50, 0)]
    assert len(everything) == len(set(everything)) == 50


def test_one_shard_takes_everything_after_the_offset():
    rows = grid(10)
    picked = [row["prompt_id"] for row in select(rows, 0, 1, 3, 0)]
    assert picked == [f"syn-{index:06d}" for index in range(3, 10)]


def test_limit_caps_what_one_worker_takes():
    rows = grid(1000)
    assert len(select(rows, 0, 4, 0, 7)) == 7

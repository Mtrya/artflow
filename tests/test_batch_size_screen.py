"""The batch-size screen must pick a size per bucket, and say what it could not.

The screen is the only thing that turns the planner's placeholder batch sizes
into measured ones, and it runs on a GPU that these tests do not have.  What is
tested here is therefore everything around the measurement: that a rewritten
plan is a plan the trainer loads, that the choice is the lowest per-sample time
rather than the largest size that fit, that out-of-memory candidates are
dropped, that a run really can be attributed to one bucket, that the cache key
separates runs whose numbers are not interchangeable, and that a bucket which
was not screened is reported with the fallback size the plan actually carries.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.bench.batch_size_screen import (
    Bucket,
    CandidateStats,
    Combination,
    DatasetLengths,
    ReportContext,
    RunRecord,
    SamplingProbe,
    apply_sizes,
    build_run_env,
    build_run_command,
    bucket_draw_mass_by_resolution,
    cache_append,
    cache_records,
    candidate_stats,
    choose_batch_size,
    combinations,
    detect_out_of_memory,
    distribution_digest,
    execute,
    load_cache,
    load_plan_for_trainer,
    parse_bucket_batch_sizes,
    parse_memory_probe,
    parse_throughput_summary,
    plan_spec,
    probe_sampling,
    read_plan,
    record_from_outcome,
    render_report,
    RunOutcome,
    screen_config_text,
    screening_plan,
    settings_digest,
    sink_batch_size,
    store_cache,
    write_plan,
)
from src.dataset.captions import CaptionPolicy
from src.dataset.length_metadata import RowLengthMetadata
from src.train.train import load_bucket_plan


def metadata(lengths):
    """A one-resolution sidecar whose rows carry the given caption lengths."""
    offsets = [0]
    flat = []
    for row in lengths:
        flat.extend(row)
        offsets.append(offsets[-1] + len(row))
    return RowLengthMetadata(
        resolution_ids=np.ones(len(lengths), dtype=np.int64),
        caption_offsets=np.array(offsets, dtype=np.int64),
        prompt_lengths=np.array(flat, dtype=np.int64),
    )


def lengths_dataset(alias, lengths, weight=1.0):
    return DatasetLengths(alias=alias, weight=weight, rows=len(lengths),
                          metadata=metadata(lengths))


def plan_with(bounds, sizes=None):
    sizes = sizes or [16] * len(bounds)
    return {1: [Bucket(bound, size) for bound, size in zip(bounds, sizes)]}


def write_input_plan(path, resolutions):
    write_plan(str(path), resolutions)
    return str(path)


# ---------------------------------------------------------------------------
# The plan the screen reads and writes.
# ---------------------------------------------------------------------------


def test_rewritten_plan_loads_in_the_trainer(tmp_path):
    """The one contract that matters: what this writes, training can read."""
    path = tmp_path / "plan.json"
    plan = write_input_plan(path, {
        1: [Bucket(64, 16), Bucket(256, 16), Bucket(1280, 16)],
        2: [Bucket(128, 16), Bucket(1280, 16)],
    })

    read_back = read_plan(plan)
    assert [bucket.max_length for bucket in read_back[1]] == [64, 256, 1280]

    rewritten = apply_sizes(read_back, {(1, 1): 24, (2, 0): 6})
    out = tmp_path / "plan.screened.json"
    write_plan(str(out), rewritten)

    loaded = load_bucket_plan(str(out), resolution_ids=[1, 2])
    assert [bucket.batch_size for bucket in loaded.buckets_for(1)] == [16, 24, 16]
    assert [bucket.batch_size for bucket in loaded.buckets_for(2)] == [6, 16]
    # The boundaries are the planner's and are never touched by the screen.
    assert [bucket.max_length for bucket in loaded.buckets_for(1)] == [64, 256, 1280]
    assert [bucket.max_length for bucket in loaded.buckets_for(2)] == [128, 1280]
    # No extra top-level keys: the loader reads every one of them as a resolution.
    assert set(json.loads(out.read_text(encoding="utf-8"))) == {"1", "2"}
    # ... and the same text is a valid inline plan, which is what a run gets.
    load_plan_for_trainer(plan_spec(rewritten), [1, 2])


# ---------------------------------------------------------------------------
# Isolating one bucket inside the sampler.
# ---------------------------------------------------------------------------


def test_screening_plan_sinks_every_bucket_but_the_target():
    plan = plan_with([64, 256, 1280])
    sink = sink_batch_size(steps=30, accumulation=16, candidates=[4, 8, 16, 32])

    screened = screening_plan(plan, resolution_id=1, bucket_index=1, candidate=8, sink=sink)

    assert [bucket.max_length for bucket in screened[1]] == [64, 256, 1280]
    assert [bucket.batch_size for bucket in screened[1]] == [sink, 8, sink]
    assert sink > 30 * 16 * 32
    # A sink bucket never emits, so the plan still has to be loadable as-is.
    load_plan_for_trainer(plan_spec(screened), [1])


def test_the_sampler_emits_only_the_bucket_under_test():
    """The isolation is a property of the sampler, so check it against it."""
    rows = [[8, 40, 300], [90, 200, 700], [20, 150, 1000], [300, 600, 1200]] * 8
    records = [lengths_dataset("only", rows)]
    plan = plan_with([64, 256, 640, 1280])
    sink = sink_batch_size(steps=10, accumulation=1, candidates=[8])

    combination = Combination(resolution_id=1, bucket_index=1, max_length=256,
                              lower_bound=64, draw_share=0.3, captions=32)
    probe = probe_sampling(records, plan, combination, candidate=8, sink=sink,
                           seed=7, policy=CaptionPolicy(), initial_stage=0.5,
                           batches=4, max_seconds=30.0)

    assert probe.unexpected_batches == 0
    assert probe.emitted_batches == 4
    assert probe.emitted_samples == 4 * 8
    # The other buckets' rows were drawn and dropped: that is the sink's cost.
    assert probe.draws > probe.emitted_samples
    # and it is what the report states as the price of the isolation.
    assert probe.draws_per_sample > 1.0
    assert probe.ms_per_sample > 0.0


# ---------------------------------------------------------------------------
# Choosing between candidates.
# ---------------------------------------------------------------------------


def records_for(batch_size, times, status="ok", accumulation=1):
    out = []
    for repeat, ms in enumerate(times):
        out.append(RunRecord(
            resolution_id=1, bucket_index=0, batch_size=batch_size, repeat=repeat,
            status=status, isolated=True, steps=20, steady_steps=10, samples=200,
            samples_per_step=float(accumulation * batch_size),
            ms_per_sample=ms, ms_per_sample_all=ms * 1.5,
            peak_allocated_gb=10.0 + batch_size, peak_reserved_gb=12.0 + batch_size,
        ))
    return out


def test_selection_takes_the_lowest_per_sample_time_not_the_largest_size():
    # A clear optimum in the middle, an OOM at the top, and a spread on every
    # candidate that is smaller than the gaps between them.
    records = (
        records_for(4, [2.00, 2.04])
        + records_for(8, [1.60, 1.62])
        + records_for(16, [1.20, 1.23])
        + records_for(32, [1.55, 1.58])
        + records_for(64, [0.0], status="oom")
    )
    stats = candidate_stats(records, [4, 8, 16, 32, 64])

    chosen, reason, candidate, runner_up, at_edge = choose_batch_size(stats)

    assert chosen == 16, "the lowest per-sample time wins, not the largest that fits"
    assert candidate.ok_runs == 2
    assert runner_up.batch_size == 32
    assert not at_edge
    assert "lowest per-sample time" in reason
    # The OOM candidate is excluded rather than treated as free.
    assert next(item for item in stats if item.batch_size == 64).ok_runs == 0
    assert next(item for item in stats if item.batch_size == 64).status == "oom"


def test_a_tie_is_broken_towards_the_smaller_size():
    # Batch 16 is the measured minimum, but batch 8 is inside batch 16's own
    # spread, so the two are not distinguishable and the cheaper one wins.
    records = records_for(8, [1.29, 1.31]) + records_for(16, [1.26, 1.30])
    stats = candidate_stats(records, [8, 16])

    chosen, reason, candidate, _, _ = choose_batch_size(stats)

    assert chosen == 8, "inside the measured spread the smaller, cheaper size wins"
    assert candidate.ms_per_sample == pytest.approx(1.30, abs=0.01)
    assert "within the measurement spread" in reason
    assert "(1.280 ms at batch 16)" in reason


def test_a_minimum_at_the_edge_is_reported_as_such():
    records = records_for(4, [1.0]) + records_for(8, [1.4]) + records_for(16, [1.9])
    stats = candidate_stats(records, [4, 8, 16])

    chosen, reason, _, _, at_edge = choose_batch_size(stats)

    assert chosen == 4 and at_edge
    assert "the optimum may lie outside the screened range" in reason


def test_a_larger_size_that_ran_out_of_memory_is_not_an_edge():
    """The screened range ended there for a stated reason, not by accident."""
    records = (records_for(4, [2.0]) + records_for(8, [1.4])
               + records_for(16, [0.0], status="oom"))
    stats = candidate_stats(records, [4, 8, 16])

    chosen, reason, _, _, at_edge = choose_batch_size(stats)

    assert chosen == 8
    assert not at_edge
    assert "edge" not in reason


def test_a_bucket_where_every_candidate_failed_is_not_given_a_size():
    records = records_for(4, [0.0], status="oom") + records_for(8, [0.0], status="oom")
    stats = candidate_stats(records, [4, 8])

    chosen, reason, _, _, _ = choose_batch_size(stats)

    assert chosen is None
    assert "no candidate completed a run" in reason


def test_a_run_that_emitted_another_bucket_is_not_a_measurement():
    combination = Combination(resolution_id=1, bucket_index=0, max_length=64,
                              lower_bound=0, draw_share=0.5, captions=10)
    # Two micro-batches of 12 samples would report 24 samples per step; this run
    # reported 32, so something other than the bucket under test emitted.
    summary = {"steps": 30.0, "steady_steps": 20.0, "samples": 640.0,
               "samples_per_step": 32.0, "samples_per_sec": 100.0,
               "samples_per_sec_steady": 120.0, "peak_mem_gb": 9.0}
    outcome = RunOutcome(status="ok", returncode=0, log_path="log", command=[],
                         summary=summary, mem_probe={"peak_reserved_gb": 11.5})

    record = record_from_outcome(combination, batch_size=12, repeat=0, outcome=outcome,
                                 accumulation=2)

    assert record.samples_per_step == 32.0
    assert record.isolated is False
    assert record.status == "missed"
    assert "another bucket emitted" in record.note
    assert record.ms_per_sample == pytest.approx(1000.0 / 120.0)
    assert record.peak_reserved_gb == 11.5


def test_a_run_of_the_bucket_alone_is_accepted():
    combination = Combination(resolution_id=2, bucket_index=1, max_length=256,
                              lower_bound=64, draw_share=0.5, captions=10)
    summary = {"steps": 30.0, "steady_steps": 20.0, "samples": 768.0,
               "samples_per_step": 24.0, "samples_per_sec": 90.0,
               "samples_per_sec_steady": 96.0, "peak_mem_gb": 8.0}
    outcome = RunOutcome(status="ok", returncode=0, log_path="log", command=[],
                         summary=summary, mem_probe={})

    record = record_from_outcome(combination, batch_size=12, repeat=1, outcome=outcome,
                                 accumulation=2)

    assert record.isolated is True and record.status == "ok"
    assert record.ms_per_sample == pytest.approx(1000.0 / 96.0)
    assert record.ms_per_sample_all == pytest.approx(1000.0 / 90.0)
    assert record.steady_steps == 20 and record.steps == 30


# ---------------------------------------------------------------------------
# The cache.
# ---------------------------------------------------------------------------


def settings(tmp_path):
    return {
        "profile": "batch-size-screen",
        "model": {"hidden_size": 1152, "single_stream_depth": 24,
                  "text_encoder": "encoder-a", "vae": "vae-a"},
        "gpu": {"name": "GPU-A", "total_memory_gb": 48.0, "count": 1},
        "software": {"torch": "2.9.0", "cuda": "13.0"},
        "execution": {"steps": 30, "accumulation": 16, "trainer_args": ["--no-compile"]},
        "distribution": {"mix": "a:0.5 b:0.5", "data_digest": "abcd",
                         "bucket_bounds": {"1": [64, 256, 1280]}},
    }


def test_cache_key_changes_with_model_gpu_settings_and_distribution(tmp_path):
    base = settings(tmp_path)
    key = settings_digest(base)

    assert settings_digest(settings(tmp_path)) == key, "the same settings reuse the same key"

    def changed(path, value):
        variant = json.loads(json.dumps(base))
        node = variant
        for step in path[:-1]:
            node = node[step]
        node[path[-1]] = value
        return settings_digest(variant)

    assert changed(["model", "hidden_size"], 768) != key
    assert changed(["model", "text_encoder"], "encoder-b") != key
    assert changed(["gpu", "name"], "GPU-B") != key
    assert changed(["software", "torch"], "2.10.0") != key
    assert changed(["execution", "trainer_args"], []) != key
    assert changed(["execution", "accumulation"], 1) != key
    assert changed(["distribution", "data_digest"], "ef01") != key
    assert changed(["distribution", "bucket_bounds"], {"1": [64, 512, 1280]}) != key


def test_cached_runs_are_returned_by_key(tmp_path):
    path = str(tmp_path / "cache.json")
    cache = load_cache(path)
    record = RunRecord(resolution_id=1, bucket_index=0, batch_size=8, repeat=0,
                       status="ok", isolated=True, ms_per_sample=1.5)

    cache_append(cache, "key-a", {"steps": 30}, [record])
    store_cache(path, cache)

    reloaded = load_cache(path)
    assert [item.batch_size for item in cache_records(reloaded, "key-a")] == [8]
    assert cache_records(reloaded, "key-b") == []
    assert reloaded["entries"]["key-a"]["settings"] == {"steps": 30}


def test_distribution_digest_follows_the_lengths_not_the_name():
    short = [lengths_dataset("same-name", [[10, 20], [30, 40]])]
    longer = [lengths_dataset("same-name", [[10, 20], [30, 900]])]

    assert distribution_digest(short) == distribution_digest(
        [lengths_dataset("same-name", [[10, 20], [30, 40]])])
    assert distribution_digest(short) != distribution_digest(longer)


# ---------------------------------------------------------------------------
# Parsing a run's output.
# ---------------------------------------------------------------------------


def test_summary_and_memory_lines_are_parsed():
    output = "\n".join([
        "Loading Text Encoder",
        "[mem-probe] peak_allocated_gb=9.100 peak_reserved_gb=11.400",
        "[throughput-summary] steps=30 samples=7680 train_wall_s=90.0 samples_per_sec=85.33 "
        "samples_per_step=256.0 steady_steps=20 samples_per_sec_steady=96.00 peak_mem_gb=9.1",
        "[mem-probe] peak_allocated_gb=9.300 peak_reserved_gb=10.900",
    ])

    summary = parse_throughput_summary(output)
    probe = parse_memory_probe(output)

    assert summary["samples_per_sec_steady"] == 96.00
    assert summary["steady_steps"] == 20
    assert summary["samples_per_step"] == 256.0
    assert summary["peak_mem_gb"] == 9.1
    # The largest peak of the run, not the last one printed.
    assert probe["peak_allocated_gb"] == 9.3
    assert probe["peak_reserved_gb"] == 11.4
    assert detect_out_of_memory(output) is False
    assert detect_out_of_memory("torch.cuda.OutOfMemoryError: CUDA out of memory. "
                               "Tried to allocate 2.00 GiB") is True


def test_execute_streams_the_log_to_a_file(tmp_path):
    log = tmp_path / "run.log"
    result = execute([sys.executable, "-c", "print('[throughput-summary] steps=3')"],
                     str(log), env=build_run_env(str(tmp_path), None, {}), timeout=60)

    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert "[throughput-summary]" in result["output"]
    assert "[throughput-summary]" in log.read_text(encoding="utf-8")


def test_execute_kills_a_run_that_hangs(tmp_path):
    """A bucket that can never fill leaves the sampler drawing forever."""
    log = tmp_path / "hang.log"
    result = execute([sys.executable, "-c", "import time; time.sleep(60)"],
                     str(log), env={}, timeout=1.0)

    assert result["timed_out"] is True
    assert result["returncode"] != 0


def test_the_planted_memory_probe_is_inert_until_training_initializes_cuda(tmp_path):
    """It has to sample peaks without changing how the process starts."""
    import scripts.bench.batch_size_screen as module

    probe_dir = module.write_probe(str(tmp_path / "probe"))
    assert Path(probe_dir, "sitecustomize.py").is_file()

    env = build_run_env("/repo", probe_dir, {})
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; print('torch' in sys.modules); print('sitecustomize' in sys.modules)"],
        env=env, capture_output=True, text=True, cwd=str(tmp_path))
    imported = result.stdout.split()
    assert imported[0] == "False", "the probe must not import torch on its own"
    assert imported[1] == "True", "but it must be the module Python picks up"


def test_the_probe_refuses_to_shadow_an_existing_sitecustomize(tmp_path):
    import scripts.bench.batch_size_screen as module

    (tmp_path / "somewhere").mkdir()
    (tmp_path / "somewhere" / "sitecustomize.py").write_text("", encoding="utf-8")

    assert module.probe_conflicts([str(tmp_path / "empty"), str(tmp_path / "somewhere")]) == [
        str(tmp_path / "somewhere" / "sitecustomize.py")]
    assert module.probe_conflicts([str(tmp_path / "empty")]) == []


def test_the_probe_reports_peaks_the_parser_reads_back(tmp_path):
    """The two halves of the reserved-memory record have to agree.

    The probe prints from inside a training process that has CUDA initialized,
    which cannot happen here; the stub stands in for that process so the probe's
    own loop and the parser are checked against each other.
    """
    import scripts.bench.batch_size_screen as module

    probe_dir = module.write_probe(str(tmp_path / "probe"))
    stub = (
        "import sys, time, types\n"
        "class Cuda:\n"
        "    def is_initialized(self): return True\n"
        "    def max_memory_allocated(self): return 12 * 1024 ** 3\n"
        "    def max_memory_reserved(self): return 15 * 1024 ** 3\n"
        "sys.modules['torch'] = types.SimpleNamespace(cuda=Cuda())\n"
        "time.sleep(1.0)\n"
    )
    result = subprocess.run([sys.executable, "-c", stub],
                            env=build_run_env(str(tmp_path), probe_dir, {}),
                            capture_output=True, text=True, cwd=str(tmp_path))

    peaks = parse_memory_probe(result.stdout)
    assert peaks["peak_allocated_gb"] == pytest.approx(12.0)
    assert peaks["peak_reserved_gb"] == pytest.approx(15.0)


def test_the_run_environment_puts_the_repo_and_the_probe_on_the_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/existing")
    env = build_run_env("/repo", "/probe", {"EXTRA": "1"})

    assert env["PYTHONPATH"].split(":") == ["/probe", "/repo", "/existing"]
    assert env["EXTRA"] == "1"
    assert env["SWANLAB_MODE"] == "disabled", "a measurement is not an experiment"
    assert build_run_env("/repo", None, {})["PYTHONPATH"] == "/repo:/existing"


def test_the_command_matches_how_gate_ab_launches_training(tmp_path):
    command = build_run_command(["configs/base.toml", str(tmp_path / "screen.toml")],
                                "screen-run", ["--no-compile"])

    assert command[1:3] == ["-m", "src.train.train"]
    assert command[3:5] == ["--config", "configs/base.toml"]
    assert command[5:7] == ["--config", str(tmp_path / "screen.toml")]
    assert command[7:9] == ["--run_name", "screen-run"]
    assert command[9:] == ["--no-compile"]


def test_the_temporary_config_leaves_the_recipe_alone(tmp_path):
    text = screen_config_text(mix="a:1.0", plan=plan_with([64, 1280]), tokenizer="/enc",
                              vae="/vae", output_dir="/run", steps=25, warmup=5,
                              accumulation=4)

    assert 'mix = "a:1.0"' in text
    assert "max_steps = 25" in text
    assert "steady_state_skip_steps = 5" in text
    assert "gradient_accumulation_steps = 4" in text
    assert "loss_interval = 0" in text
    # No checkpoint or eval can land inside the measured steps.
    assert "checkpoint_interval = 26" in text
    assert "eval_interval = 26" in text
    # Everything else comes from the recipe, which is the point of the screen.
    assert "hidden_size" not in text and "learning_rate" not in text


# ---------------------------------------------------------------------------
# What a bucket's draw share says about how much it can be screened.
# ---------------------------------------------------------------------------


def test_draw_share_is_per_bucket_and_per_resolution():
    records = [
        lengths_dataset("short", [[10, 90]] * 10, weight=0.5),
        lengths_dataset("long", [[500, 1200]] * 10, weight=0.5),
    ]
    plan = plan_with([64, 256, 1280])

    found = {combo.bucket_index: combo for combo in combinations(plan, [1], records)}

    # Every caption of both datasets is weighted by its dataset's mix weight over
    # that dataset's row count: 10 captions of length 10 land in bucket 0, ten of
    # length 90 in bucket 1, and all twenty of 500 and 1200 in bucket 2.
    assert found[0].captions == 10 and found[1].captions == 10 and found[2].captions == 20
    assert found[0].draw_share == pytest.approx(0.25)
    assert found[1].draw_share == pytest.approx(0.25)
    assert found[2].draw_share == pytest.approx(0.50)
    assert sum(combo.draw_share for combo in found.values()) == pytest.approx(1.0)


def test_a_caption_above_every_bound_is_refused():
    """Such a row has no bucket, so training would fail on it too."""
    records = [lengths_dataset("long", [[1400]])]

    with pytest.raises(ValueError, match="no bucket"):
        combinations(plan_with([64, 1280]), [1], records)


def test_bucket_mass_is_the_share_of_all_draws(tmp_path):
    """The merge aligns the buckets with this distribution, so it has to follow
    the sampler's draw rule: dataset by weight, then a row, then a caption."""
    records = [
        # Two captions per row, so each row contributes two draws, and both
        # land in bucket 0 (bound 64) or bucket 1 (bound 256).
        lengths_dataset("two-per-row", [[10, 90]] * 5, weight=0.5),
        # One caption per row: bucket 0 once, bucket 2 (bound 1280) four times.
        lengths_dataset("one-per-row", [[10], [300], [1200], [300], [1200]],
                        weight=0.5),
    ]
    plan = plan_with([64, 256, 1280])

    mass = bucket_draw_mass_by_resolution(plan, [1], records)

    assert list(mass) == [1]
    assert sum(mass[1]) == pytest.approx(1.0)
    # Row-weighted, not caption-weighted: the two-per-row dataset contributes
    # 0.5/5 per caption, not 0.5/10 or 0.5 per caption.
    assert mass[1][0] == pytest.approx(0.6 / 1.5)
    assert mass[1][1] == pytest.approx(0.5 / 1.5)
    assert mass[1][2] == pytest.approx(0.4 / 1.5)


def test_bucket_mass_out_writes_the_distribution_without_screening(tmp_path, monkeypatch):
    import scripts.bench.batch_size_screen as module

    rows = [[10, 20, 30, 50]] * 60 + [[80, 120, 200]] * 30 + [[700]]
    monkeypatch.setattr(
        module, "load_dataset_lengths",
        lambda entries, tokenizer: [lengths_dataset("only", rows, weight=1.0)])
    calls = fake_trainer(module, monkeypatch, times={4: 2.0})
    plan_path = tmp_path / "plan.json"
    write_plan(str(plan_path), plan_with([64, 256, 1280]))
    out = tmp_path / "mass.json"

    assert module.main(["--plan", str(plan_path), "--mix", "only:1.0",
                        "--text-encoder", "/enc", "--bucket-mass-out", str(out)]) == 0

    assert calls == [], "writing the mass must not reach the training runs"
    mass = json.loads(out.read_text(encoding="utf-8"))
    assert list(mass) == ["1"]
    assert len(mass["1"]) == 3
    assert sum(mass["1"]) == pytest.approx(1.0)
    # The long bucket holds a single caption of the whole mix.
    assert mass["1"][2] < mass["1"][1] < mass["1"][0]


def test_screening_without_a_vae_is_refused_with_a_reason(tmp_path, monkeypatch, capsys):
    import scripts.bench.batch_size_screen as module

    rows = [[10, 20, 30, 50]] * 60
    monkeypatch.setattr(
        module, "load_dataset_lengths",
        lambda entries, tokenizer: [lengths_dataset("only", rows, weight=1.0)])
    plan_path = tmp_path / "plan.json"
    write_plan(str(plan_path), plan_with([64, 256, 1280]))

    assert module.main(["--plan", str(plan_path), "--mix", "only:1.0",
                        "--text-encoder", "/enc", "--out", str(tmp_path / "out.json"),
                        "--gpu", "GPU-A", "--device-memory-gb", "48",
                        "--device-count", "1"]) == 2
    assert "--vae" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------


def report_context(**overrides):
    combination = Combination(resolution_id=1, bucket_index=0, max_length=64,
                              lower_bound=0, draw_share=0.5, captions=200)
    measured = Combination(resolution_id=1, bucket_index=1, max_length=256,
                           lower_bound=64, draw_share=0.4, captions=150)
    rare = Combination(resolution_id=1, bucket_index=2, max_length=1280,
                       lower_bound=256, draw_share=0.002, captions=6)
    stats = candidate_stats(records_for(8, [1.5, 1.6]), [4, 8, 16])
    from scripts.bench.batch_size_screen import Selection

    selection = Selection(combination=measured, batch_size=8, source="measured",
                          reason="lowest per-sample time (1.550 ms) of the screened candidates",
                          candidate=stats[1], runner_up=stats[2], margin_gb=30.0,
                          margin_fraction=0.625)
    context = dict(
        argv=["--plan", "plan.json"], plan_path="plan.json",
        out_path="plan.screened.json", measurements_path="plan.screened.json.measurements.json",
        cache_path="plan.screened.json.cache.json", run_dir="plan.screened.json.runs",
        mix="a:1.0", entries=[], base_config="configs/base.toml", extra_configs=[],
        trainer_args=[], vae="/vae", text_encoder="/enc", gpu="GPU-A",
        device_memory_gb=48.0, candidates=[4, 8, 16], steps=30, warmup=10, repeats=2,
        accumulation=16, sink=1_000_000, min_caption_share=0.05, min_margin_fraction=0.1,
        cache_key="abc123",
        settings={"execution": {"env": {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}}},
        selections=[selection],
        unscreened=[(rare, "holds 0.200% of its resolution's draw mass (6 captions), "
                           "below --min-caption-share 5.0%")],
        stats={(1, 1): stats}, probes={(1, 1): SamplingProbe(
            emitted_batches=16, emitted_samples=128, draws=640, seconds=0.01,
            unexpected_batches=0, truncated=False)},
        records=[], resolutions=[1], kept_resolutions={2: [16, 16]}, fallback_size=4,
        warnings=[], draw_probe=True,
    )
    context.update(overrides)
    context.pop("combinations", None)
    return ReportContext(**context)


def test_the_report_names_every_bucket_that_was_not_screened_with_its_fallback():
    report = render_report(report_context())

    assert "## Not screened" in report
    assert "res1/bucket2" in report or "| 1 | 2 |" in report
    assert "(256, 1280]" in report
    assert "0.200%" in report, "how rare the bucket is has to be visible"
    assert "--min-caption-share" in report, "and why it was skipped"
    assert "fallback" in report and "4" in report
    # The bucket that was screened states its size, its spread and its margin.
    assert "## Selections" in report and "## Memory" in report
    assert "1.550" in report and "62.5%" in report
    # Resolutions the screen was not asked for are named rather than silently
    # passed on as if they had been measured.
    assert "batch sizes kept from the plan" in report and "| 2 | 16, 16 |" in report
    # The environment is part of how the memory numbers have to be read.
    assert "`PYTORCH_ALLOC_CONF=expandable_segments:True`" in report
    # And the cache key is recorded, with what it covers.
    assert "abc123" in report and "## Cache" in report


def test_the_report_states_that_the_sink_costs_discarded_draws():
    report = render_report(report_context())

    assert "draws per sample" in report
    assert "5.0" in report, "640 draws for 128 emitted samples"
    # The cost is visible, but it is not silently folded into the numbers.
    assert "reported as an estimate rather than" in report
    assert "does not decide which candidate wins" in report


def test_candidate_stats_take_the_worst_peak_but_the_median_time():
    records = records_for(8, [1.5, 1.9])
    records[1].peak_reserved_gb = 20.0
    stats = candidate_stats(records, [8])[0]

    assert stats.ms_per_sample == pytest.approx(1.7)
    assert stats.spread_ms == pytest.approx(0.4)
    assert stats.peak_reserved_gb == 20.0


# ---------------------------------------------------------------------------
# The whole screen, with the training process replaced by a stub.
# ---------------------------------------------------------------------------


def fake_trainer(module, monkeypatch, times, oom_at=(), calls=None):
    """Stand in for the training runs, so the screen's own logic is what runs.

    The stub answers with a summary the real trainer's shape: the per-sample
    time comes from ``times``, ``samples_per_step`` is derived from the plan it
    was handed exactly as an isolated run would report it, and the candidates in
    ``oom_at`` fail the way a CUDA out-of-memory would.
    """
    calls = calls if calls is not None else []

    def runner(**kwargs):
        calls.append(kwargs)
        sizes = [bucket.batch_size for buckets in kwargs["plan"].values()
                 for bucket in buckets]
        candidate = min(sizes)
        if candidate in oom_at:
            return RunOutcome(status="oom", returncode=1, log_path=str(kwargs["run_dir"]),
                              command=[], note="CUDA out of memory")
        ms = times[candidate]
        return RunOutcome(
            status="ok", returncode=0, log_path=str(kwargs["run_dir"]), command=[],
            summary={"steps": float(kwargs["steps"]),
                     "steady_steps": float(kwargs["steps"] - kwargs["warmup"]),
                     "samples": 100.0,
                     "samples_per_step": float(kwargs["accumulation"] * candidate),
                     "samples_per_sec": 1000.0 / (ms * 1.2),
                     "samples_per_sec_steady": 1000.0 / ms,
                     "peak_mem_gb": 10.0 + 0.5 * candidate},
            mem_probe={"peak_reserved_gb": 12.0 + 0.5 * candidate},
        )

    monkeypatch.setattr(module, "run_training_run", runner)
    return calls


def test_the_screen_writes_a_measured_plan_a_report_and_a_cache(tmp_path, monkeypatch):
    import scripts.bench.batch_size_screen as module

    rows = [[10, 20, 30, 50]] * 60 + [[80, 120, 200]] * 30 + [[700]]
    monkeypatch.setattr(
        module, "load_dataset_lengths",
        lambda entries, tokenizer: [lengths_dataset("only", rows, weight=1.0)])
    calls = fake_trainer(module, monkeypatch, times={4: 2.0, 8: 1.0, 16: 1.5},
                         oom_at=(16,))
    plan_path = tmp_path / "plan.json"
    write_plan(str(plan_path), plan_with([64, 256, 1280]))
    out = tmp_path / "screened.json"
    argv = [
        "--plan", str(plan_path), "--mix", "only:1.0", "--vae", "/vae",
        "--text-encoder", "/enc", "--batch-sizes", "4", "8", "16", "--steps", "6",
        "--warmup", "2", "--repeats", "1", "--accumulation", "1", "--max-runs", "20",
        "--gpu", "GPU-A", "--device-memory-gb", "48", "--device-count", "1",
        "--out", str(out),
    ]

    assert module.main(argv) == 0

    plan = load_bucket_plan(str(out), resolution_ids=[1])
    sizes = [bucket.batch_size for bucket in plan.buckets_for(1)]
    # Bucket 0 and 1: measured, and batch 8 is the fastest (batch 16 OOMs).
    # Bucket 2 holds a single caption, far below --min-caption-share, so it is
    # not screened and carries the declared fallback instead of the placeholder.
    assert sizes == [8, 8, 4]
    sink = sink_batch_size(6, 1, [4, 8, 16])
    for call in calls:
        bounds = [bucket.max_length for bucket in call["plan"][1]]
        bucket_sizes = [bucket.batch_size for bucket in call["plan"][1]]
        assert bounds == [64, 256, 1280], "the bounds are the plan's, never re-derived"
        assert sum(1 for size in bucket_sizes if size != sink) == 1, \
            "exactly one bucket emits: the one under test"
        assert min(bucket_sizes) in (4, 8, 16), "and it carries the candidate size"

    report = Path(str(out) + ".report.md").read_text(encoding="utf-8")
    measurements = json.loads(Path(str(out) + ".measurements.json").read_text(encoding="utf-8"))
    assert "## Not screened" in report and "(256, 1280]" in report
    entry = measurements["not_screened"][0]
    assert entry["bucket_index"] == 2 and entry["fallback_batch_size"] == 4
    assert entry["captions"] == 1 and "below --min-caption-share" in entry["why"]
    assert measurements["fallback_batch_size"] == 4
    assert {record["status"] for record in measurements["runs"]} == {"ok", "oom"}
    assert sorted((selection["resolution_id"], selection["bucket_index"],
                   selection["batch_size"]) for selection in measurements["selections"]) == [
        (1, 0, 8), (1, 1, 8)]
    # The OOM at batch 16 is recorded against the candidate that hit it, in both
    # buckets that were screened.
    assert [record["batch_size"] for record in measurements["runs"]
            if record["status"] == "oom"] == [16, 16]
    assert (tmp_path / "screened.json.cache.json").is_file()

    # A second screen with the same settings runs nothing: the numbers describe
    # the same model, GPU, settings and length distribution.
    before = len(calls)
    assert module.main(argv) == 0
    assert len(calls) == before, "cached measurements must not be re-run"
    assert load_bucket_plan(str(out), [1]).buckets_for(1)[0].batch_size == 8


def test_a_spent_run_budget_is_reported_with_the_fallback(tmp_path, monkeypatch):
    """Buckets the budget never reached still get a size, and a stated one."""
    import scripts.bench.batch_size_screen as module

    rows = [[10, 20, 30, 50]] * 60 + [[80, 120, 200]] * 30
    monkeypatch.setattr(
        module, "load_dataset_lengths",
        lambda entries, tokenizer: [lengths_dataset("only", rows, weight=1.0)])
    calls = fake_trainer(module, monkeypatch, times={4: 2.0, 8: 1.0, 16: 1.5})
    plan_path = tmp_path / "plan.json"
    write_plan(str(plan_path), plan_with([64, 256, 1280]))
    out = tmp_path / "screened.json"

    assert module.main([
        "--plan", str(plan_path), "--mix", "only:1.0", "--vae", "/vae",
        "--text-encoder", "/enc", "--batch-sizes", "4", "8", "16", "--steps", "6",
        "--warmup", "2", "--repeats", "1", "--accumulation", "1", "--max-runs", "1",
        "--fallback-batch-size", "8", "--gpu", "GPU-A", "--device-memory-gb", "48",
        "--device-count", "1", "--out", str(out),
    ]) == 0

    measurements = json.loads(Path(str(out) + ".measurements.json").read_text(encoding="utf-8"))
    assert len(calls) == 1, "the budget stops the scan after one run"
    why = {entry["bucket_index"]: entry["why"] for entry in measurements["not_screened"]}
    assert any("run budget" in text for text in why.values())
    assert all(entry["fallback_batch_size"] == 8 for entry in measurements["not_screened"])
    sizes = [bucket.batch_size
             for bucket in load_bucket_plan(str(out), [1]).buckets_for(1)]
    # The one measured bucket kept its measurement; the rest carry the declared
    # fallback rather than the placeholder they arrived with.
    assert 4 in sizes and sizes.count(8) == 2


def test_a_different_gpu_does_not_reuse_the_other_measurements(tmp_path, monkeypatch):
    import scripts.bench.batch_size_screen as module

    rows = [[10, 20, 30, 50]] * 60 + [[80, 120, 200]] * 30
    monkeypatch.setattr(
        module, "load_dataset_lengths",
        lambda entries, tokenizer: [lengths_dataset("only", rows, weight=1.0)])
    calls = fake_trainer(module, monkeypatch, times={4: 2.0, 8: 1.0, 16: 1.5})
    plan_path = tmp_path / "plan.json"
    write_plan(str(plan_path), plan_with([64, 256, 1280]))
    base = ["--plan", str(plan_path), "--mix", "only:1.0", "--vae", "/vae",
            "--text-encoder", "/enc", "--batch-sizes", "4", "8", "16", "--steps", "6",
            "--warmup", "2", "--repeats", "1", "--accumulation", "1",
            "--device-memory-gb", "48", "--device-count", "1",
            "--out", str(tmp_path / "screened.json")]

    assert module.main([*base, "--gpu", "GPU-A"]) == 0
    first = len(calls)
    assert module.main([*base, "--gpu", "GPU-B"]) == 0

    assert len(calls) > first, "another GPU is another measurement"


def test_bucket_batch_sizes_names_candidates_per_bucket():
    assert parse_bucket_batch_sizes(["0:80,96", "4:2:40"], "x") == {
        (None, 0): [80, 96], (4, 2): [40]}


def test_bucket_batch_sizes_rejects_malformed_entries():
    with pytest.raises(ValueError):
        parse_bucket_batch_sizes(["0-80"], "x")
    with pytest.raises(ValueError):
        parse_bucket_batch_sizes(["0:80", "0:96"], "x")


def test_a_dense_pass_measures_only_the_buckets_it_names(tmp_path, monkeypatch):
    import scripts.bench.batch_size_screen as module

    rows = [[10]] * 50 + [[100]] * 40 + [[900]] * 10
    monkeypatch.setattr(
        module, "load_dataset_lengths",
        lambda entries, tokenizer: [lengths_dataset("only", rows, weight=1.0)])
    calls = fake_trainer(module, monkeypatch, times={40: 4.0, 80: 3.0, 96: 2.0})
    plan_path = tmp_path / "plan.json"
    write_plan(str(plan_path), plan_with([64, 256, 1280]))
    out_path = tmp_path / "screened.json"

    assert module.main([
        "--plan", str(plan_path), "--mix", "only:1.0", "--vae", "/vae",
        "--text-encoder", "/enc", "--bucket-batch-sizes", "0:80,96", "2:40",
        "--steps", "6", "--warmup", "2", "--repeats", "1", "--accumulation", "1",
        "--min-caption-share", "0.001", "--fallback-batch-size", "16",
        "--gpu", "GPU-A", "--device-memory-gb", "48", "--device-count", "1",
        "--out", str(out_path)]) == 0

    seen = sorted({min(bucket.batch_size for buckets in call["plan"].values()
                       for bucket in buckets) for call in calls})
    assert seen == [40, 80, 96], "only the listed buckets, at the listed sizes"
    screened = read_plan(str(out_path))
    sizes = [bucket.batch_size for bucket in screened[1]]
    assert sizes[0] == 96 and sizes[2] == 40, "the listed buckets get measured sizes"
    assert sizes[1] == 16, "an unlisted bucket keeps the declared fallback"

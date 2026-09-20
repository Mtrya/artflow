import json
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.bench import infra_memory_stress as stress
from src.dataset.length_metadata import RowLengthMetadata, prompt_metadata_contract
from src.dataset.sampler import BucketPlan, RowDescriptorDataset, row_length_collate_fn


@pytest.mark.parametrize("accumulation", [1, 5, 7])
def test_prepare_requires_explicit_stage_accumulation(monkeypatch, tmp_path, accumulation):
    seen = []
    monkeypatch.setattr(stress, "prepare", lambda args: seen.append(args.accumulation))
    argv = ["stress", "prepare", "--config", "configs/base.toml", "--out", str(tmp_path),
            "--ranks", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        stress.main()
    monkeypatch.setattr(sys, "argv", [*argv, "--accumulation", "0"])
    with pytest.raises(SystemExit):
        stress.main()
    assert not seen
    monkeypatch.setattr(sys, "argv", [*argv, "--accumulation", str(accumulation)])
    stress.main()
    assert seen == [accumulation]


def fixture():
    cases = [dict(resolution_id=1, bucket_index=0, length=4, batch_size=3,
                  latent_shape=[16, 4, 4], row_start=0),
             dict(resolution_id=2, bucket_index=0, length=2048, batch_size=1,
                  latent_shape=[16, 2, 8], row_start=2)]
    manifest = dict(kind="synthetic-memory-stress-v1", ranks=2, accumulation=2,
                    cases=cases, cycles=2, updates=4, caveat="synthetic")
    metadata = RowLengthMetadata([1, 1, 2, 2], [0, 1, 2, 3, 4], [4, 4, 2048, 2048],
                                metadata_info=prompt_metadata_contract(4))
    plan = BucketPlan({1: [(4, 3)], 2: [(2048, 1)]})
    rows = [dict(latents=torch.zeros(case["latent_shape"]), captions=["ink"],
                 resolution_bucket_id=case["resolution_id"]) for case in cases for _ in range(2)]
    return manifest, metadata, plan, rows


@pytest.mark.parametrize("rank", [0, 1])
def test_stress_sampler_exercises_upper_bounds_with_real_collation(rank):
    manifest, metadata, plan, rows = fixture()
    sampler = stress.stress_sampler_type(manifest)(metadata=[metadata], bucket_plan=plan,
                                                  num_replicas=2, rank=rank)
    loader = DataLoader(RowDescriptorDataset([rows], [metadata]), batch_sampler=sampler,
                        collate_fn=row_length_collate_fn, num_workers=0)
    iterator = iter(loader)
    lengths = []
    for batch_id in range(8):
        batch = next(iterator)
        case = stress.ordered_case(batch_id, 2, manifest["cases"])
        assert list(batch["latents"].shape) == [case["batch_size"], *case["latent_shape"]]
        assert batch["retained_lengths"].tolist() == [case["length"]] * case["batch_size"]
        assert batch["row_positions"] == [(0, case["row_start"] + rank)] * case["batch_size"]
        assert batch["batch_id"] == batch_id
        sampler.ack_batch(batch_id)
        lengths.append(batch["bucket_hi"])
    assert lengths == [4, 4, 2048, 2048, 2048, 2048, 4, 4]
    assert not sampler.state_dict()["inflight"]


def test_caption_synthesis_requires_exact_contract_length(monkeypatch):
    monkeypatch.setattr(stress, "_tokenize_prompt_lengths",
                        lambda tokenizer, prompts: np.array([min(2048, p.count(" ink") + 3)
                                                            for p in prompts]))
    assert stress.caption_at_length(None, 4) == " ink"
    assert stress.caption_at_length(None, 2048).count(" ink") == 2045
    with pytest.raises(ValueError, match="exact retained length"):
        stress.caption_at_length(None, 2)


def test_health_observer_does_not_retain_snapshot():
    from src.train.health import snapshot_weights
    param = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.SGD([param], lr=0.1)
    observer = stress.HealthSnapshotObserver(snapshot_weights)
    snapshot = observer.snapshot([optimizer])
    assert observer.finish_update() == dict(calls=1, devices=["cpu"], released=False)
    del snapshot
    snapshot = observer.snapshot([optimizer])
    del snapshot
    assert observer.finish_update() == dict(calls=1, devices=["cpu"], released=True)
    assert observer.finish_update() == dict(calls=0, devices=[], released=True)


@pytest.mark.parametrize("health_interval", [None, 1, 2])
def test_verify_requires_both_cycles_at_declared_micro_shapes(tmp_path, monkeypatch, health_interval):
    manifest, _, _, _ = fixture()
    if health_interval is not None:
        manifest["health_interval"] = health_interval
    monkeypatch.setattr(stress, "read_manifest", lambda _: manifest)
    directory = tmp_path / "synthetic-memory/infra"
    directory.mkdir(parents=True)
    for rank in range(2):
        records = []
        for index in range(4):
            case = stress.ordered_case(index * 2, 2, manifest["cases"])
            records.append(dict(step=index + 1, rank=rank, loss=1.,
                                shapes=[dict(shape=[case["batch_size"], *case["latent_shape"],
                                                    case["length"]], count=2)],
                                peak_allocated_bytes=100, peak_reserved_bytes=200))
        (directory / f"rank-{rank}.jsonl").write_text("\n".join(map(json.dumps, records)))
        contexts = [dict(step=i + 1, rank=rank, total_bytes=1000, estimated_external_bytes=100)
                    for i in range(4)]
        if health_interval is not None:
            for context in contexts:
                calls = int(context["step"] % health_interval == 0)
                context["gpu_health_snapshot"] = True
                context["health_snapshot"] = dict(calls=calls, devices=["cuda"] if calls else [],
                                                   released=True)
        (directory / f"rank-{rank}.memory-context.jsonl").write_text(
            "\n".join(map(json.dumps, contexts)))
    args = type("Args", (), {"manifest": tmp_path / "manifest.json"})()
    stress.verify(args)
    assert json.loads((tmp_path / "verification.json").read_text())["cases"] == 2
    records[-1]["shapes"][0]["count"] = 1
    (directory / "rank-1.jsonl").write_text("\n".join(map(json.dumps, records)))
    with pytest.raises(ValueError, match="wrong shape/order"):
        stress.verify(args)
    if health_interval is not None:
        records[-1]["shapes"][0]["count"] = 2
        (directory / "rank-1.jsonl").write_text("\n".join(map(json.dumps, records)))
        contexts[-1]["health_snapshot"]["released"] = False
        (directory / "rank-1.memory-context.jsonl").write_text("\n".join(map(json.dumps, contexts)))
        with pytest.raises(ValueError, match="health boundary not exercised"):
            stress.verify(args)

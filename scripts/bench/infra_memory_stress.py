"""Deterministic bucket-boundary stress through the real trainer.

Synthetic latents/captions replace the data and a diagnostic sampler visits every
plan entry. Dropout is disabled to retain every text token. This is memory and
finite-update evidence ONLY, not sampling, quality, throughput or resume evidence.
Run after choosing execution flags; use eight ranks for final acceptance. The
second pass revisits shapes after optimizer state and compiled caches exist.
Preparation and final checkpoint writes must not overlap throughput measurement.
Pass the intended stage accumulation explicitly: the external stage configs do
not necessarily override the base config's accumulation.
Use --health-interval 1 to exercise GPU health snapshots at every boundary;
this deliberately changes diagnostic cadence, not production cadence.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import weakref

import numpy as np

from src.dataset.length_metadata import _prompt_text, _tokenize_prompt_lengths
from src.dataset.sampler import RowLengthQueueBatchSampler, RowRef


def caption_at_length(tokenizer, length):
    """Require an exact online-contract length, never fake the sidecar count."""
    def count(n):
        caption = " ink" * n
        retained = int(_tokenize_prompt_lengths(tokenizer, [_prompt_text(caption)])[0])
        return caption, retained

    lo, hi = 0, length * 2 + 64
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid)[1] < length:
            lo = mid + 1
        else:
            hi = mid
    caption, retained = count(lo)
    if retained != length:
        raise ValueError(f"cannot synthesize exact retained length {length}; got {retained}")
    return caption


def ordered_case(batch_id, accumulation, cases):
    update = batch_id // accumulation
    cycle, position = divmod(update, len(cases))
    # Reverse alternate passes to exercise both directions of shape changes.
    return cases[position if cycle % 2 == 0 else len(cases) - 1 - position]


def stress_sampler_type(manifest):
    class StressSampler(RowLengthQueueBatchSampler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.num_replicas != manifest["ranks"] or len(self.metadata) != 1:
                raise ValueError("stress dataset/rank contract differs from its manifest")

        def _next_ready_batch(self):
            batch_id = self._next_batch_id
            case = ordered_case(batch_id, manifest["accumulation"], manifest["cases"])
            row_idx = case["row_start"] + self.rank
            metadata = self.metadata[0]
            retained = int(metadata.prompt_lengths[metadata.row_slice(row_idx)][0])
            bucket = self.bucket_plan.buckets_for(case["resolution_id"])[case["bucket_index"]]
            if (retained != case["length"] or bucket.max_length != case["length"]
                    or bucket.batch_size != case["batch_size"]
                    or int(metadata.resolution_ids[row_idx]) != case["resolution_id"]):
                raise ValueError("stress case no longer matches dataset metadata or plan")
            self._next_batch_id += 1
            ref = RowRef(0, row_idx, 0, case["resolution_id"], retained,
                         case["bucket_index"], case["length"], batch_id)
            return [ref] * bucket.batch_size

    return StressSampler


def prepare(args):
    from datasets import Dataset, load_from_disk
    from transformers import AutoTokenizer
    from src.dataset.length_metadata import (
        METADATA_VERSION, RowLengthMetadata, _source_signature, ensure_sidecar,
    )
    from src.dataset.mix import parse_dataset_mix
    from src.train.config import load_config
    from src.train.train import load_bucket_plan

    configs = [str(Path(p).resolve()) for p in args.config]
    cfg = load_config(configs)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Derive latent shapes from real precomputed rows, not a guessed aspect map.
    shapes = {}
    for entry in parse_dataset_mix(cfg.data.mix):
        # Existing hero inputs are read-only, including when a sidecar is stale.
        metadata = RowLengthMetadata.load(Path(entry.path) / "length_metadata.npz")
        if (metadata.metadata_version != METADATA_VERSION
                or (metadata.metadata_info or {}).get("source_signature")
                != _source_signature(entry.path, cfg.text_encoder.path)):
            raise ValueError(f"source sidecar is stale; refusing to rewrite {entry.path}")
        dataset = None
        for resolution_id in np.unique(metadata.resolution_ids):
            resolution_id = int(resolution_id)
            if resolution_id in shapes:
                continue
            if dataset is None:
                dataset = load_from_disk(str(entry.path))
            index = int(np.flatnonzero(metadata.resolution_ids == resolution_id)[0])
            shape = list(np.asarray(dataset[index]["latents"]).shape)
            if len(shape) != 3 or shape[0] != 16 or any(x % 2 for x in shape[1:]):
                raise ValueError(f"unexpected precomputed latent shape {shape}")
            shapes[resolution_id] = shape
    plan = load_bucket_plan(cfg.data.bucket_plan, set(shapes))
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder.path)
    captions, cases = {}, []
    for resolution_id, buckets in sorted(plan.by_resolution.items()):
        if resolution_id not in shapes:
            raise ValueError(f"no real latent shape for plan resolution {resolution_id}")
        for index, bucket in enumerate(buckets):
            if bucket.max_length not in captions:
                captions[bucket.max_length] = caption_at_length(tokenizer, bucket.max_length)
            cases.append(dict(resolution_id=resolution_id, bucket_index=index,
                              length=bucket.max_length, batch_size=bucket.batch_size,
                              latent_shape=shapes[resolution_id], row_start=len(cases) * args.ranks))

    def rows():
        for case in cases:
            for rank in range(args.ranks):
                rng = np.random.default_rng(42 + case["row_start"] + rank)
                yield dict(latents=rng.standard_normal(case["latent_shape"], dtype=np.float32),
                           captions=[captions[case["length"]]],
                           resolution_bucket_id=case["resolution_id"])

    data_path = out / "synthetic-data"
    Dataset.from_generator(rows, cache_dir=str(out / "dataset-cache")).save_to_disk(str(data_path))
    generated = ensure_sidecar(data_path, cfg.text_encoder.path)
    for case in cases:
        for rank in range(args.ranks):
            lengths = generated.prompt_lengths[generated.row_slice(case["row_start"] + rank)]
            if lengths.tolist() != [case["length"]]:
                raise ValueError("generated sidecar disagrees with requested stress length")
    plan_path = out / "stress-plan.json"
    plan_path.write_text(json.dumps({str(r): [[b.max_length, b.batch_size] for b in buckets]
                                    for r, buckets in plan.by_resolution.items()}) + "\n")
    updates = args.cycles * len(cases)
    health_interval = (cfg.telemetry.health_interval if args.health_interval is None
                       else args.health_interval)
    cache_interval = (cfg.telemetry.cache_clear_interval if args.cache_clear_interval is None
                      else args.cache_clear_interval)
    override = out / "stress.toml"
    override.write_text(f'''# Synthetic memory diagnostic; never a hero recipe.
[data]
mix = {json.dumps(str(data_path) + ':1')}
bucket_plan = {json.dumps(str(plan_path))}
caption_dropout_prob = 0.0
[train]
gradient_accumulation_steps = {args.accumulation}
stop_at_step = {updates}
checkpoint_interval = 1000000000
eval_interval = 1000000000
[eval]
dataset_path = ""
grid_steps = []
loss_interval = 0
kid_at_end = false
[telemetry]
health_interval = {health_interval}
cache_clear_interval = {cache_interval}
[paths]
output_dir = {json.dumps(str(out))}
''')
    load_config([*configs, str(override)])  # Reject an incompatible global T before allocating GPUs.
    manifest = dict(kind="synthetic-memory-stress-v1", ranks=args.ranks, cycles=args.cycles,
                    accumulation=args.accumulation, health_interval=health_interval, cases=cases,
                    cache_clear_interval=cache_interval,
                    updates=updates, configs=[*configs, str(override)],
                    plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    caveat="Not sampling/quality/throughput/resume evidence; checkpoint is synthetic.")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(manifest=str(out / "manifest.json"), cases=len(cases), updates=updates)))


def read_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest["kind"] != "synthetic-memory-stress-v1" or manifest["cycles"] < 2:
        raise ValueError("unsupported stress manifest or missing post-state replay cycle")
    plan = path.parent / "stress-plan.json"
    if hashlib.sha256(plan.read_bytes()).hexdigest() != manifest["plan_sha256"]:
        raise ValueError("stress plan changed after preparation")
    return manifest


class HealthSnapshotObserver:
    """Observe real trainer snapshots without extending their tensor lifetime."""
    def __init__(self, snapshot_fn):
        self.snapshot_fn = snapshot_fn
        self.refs = []
        self.devices = set()
        self.calls = 0

    def snapshot(self, *args, **kwargs):
        result = self.snapshot_fn(*args, **kwargs)
        self.calls += 1
        for group in result:
            for _, tensor in group:
                self.refs.append(weakref.ref(tensor))
                self.devices.add(tensor.device.type)
        return result

    def finish_update(self):
        result = dict(calls=self.calls, devices=sorted(self.devices),
                      released=all(ref() is None for ref in self.refs))
        self.refs.clear()
        self.devices.clear()
        self.calls = 0
        return result


def train(args):
    import torch
    from src.train import train as trainer
    manifest = read_manifest(args.manifest)
    from src.train.config import load_config
    cfg = load_config(manifest["configs"])
    if ("health_interval" in manifest
            and cfg.telemetry.health_interval != manifest["health_interval"]):
        raise ValueError("health cadence changed after stress preparation")
    if ("cache_clear_interval" in manifest
            and cfg.telemetry.cache_clear_interval != manifest["cache_clear_interval"]):
        raise ValueError("cache-clearing policy changed after stress preparation")
    if int(os.environ.get("WORLD_SIZE", "1")) != manifest["ranks"]:
        raise ValueError("launch rank count differs from stress manifest")
    # Only execution switches are forwarded; the manifest owns configs and stops.
    allowed = {"--compile_dynamic", "--compile_autotune", "--disable_ddp_compile_split",
               "--native_flash_varlen", "--real_rope", "--hoist_double_rope",
               "--muon_compile_square_ns", "--foreach_updates", "--local_cache_clear",
               "--gpu_health_snapshot"}
    if set(args.train_arg) - allowed:
        raise ValueError("unsupported stress execution switch")
    gpu_health = "--gpu_health_snapshot" in args.train_arg
    if gpu_health and manifest.get("health_interval") != 1:
        raise ValueError("GPU health memory stress requires prepare --health-interval 1")
    os.environ["ARTFLOW_INFRA_METRICS"] = "1"
    os.environ["ARTFLOW_TRACE_START"] = "-1"
    trainer.RowLengthQueueBatchSampler = stress_sampler_type(manifest)
    health = HealthSnapshotObserver(trainer.snapshot_weights)
    trainer.snapshot_weights = health.snapshot

    class MemoryRecorder(trainer.InfraRecorder):
        def end_update(self, **kwargs):
            free, total = torch.cuda.mem_get_info()
            reserved = torch.cuda.memory_reserved()
            context = dict(step=kwargs["step"], rank=self.rank, total_bytes=total,
                           free_at_update_end_bytes=free, reserved_at_update_end_bytes=reserved,
                           estimated_external_bytes=max(0, total - free - reserved))
            context["health_snapshot"] = health.finish_update()
            context["gpu_health_snapshot"] = gpu_health
            if not context["health_snapshot"]["released"]:
                raise RuntimeError("health snapshot survived to the next forward boundary")
            with (self.directory / f"rank-{self.rank}.memory-context.jsonl").open("a") as handle:
                handle.write(json.dumps(context) + "\n")
            super().end_update(**kwargs)

    trainer.InfraRecorder = MemoryRecorder
    sys.argv = ["memory-stress-trainer", *[a for p in manifest["configs"] for a in ("--config", p)],
                "--run_name", "synthetic-memory", *args.train_arg]
    trainer.main()


def verify(args):
    manifest = read_manifest(args.manifest)
    root = args.manifest.parent / "synthetic-memory" / "infra"
    peaks = {}
    for rank in range(manifest["ranks"]):
        records = [json.loads(line) for line in (root / f"rank-{rank}.jsonl").read_text().splitlines()]
        contexts = [json.loads(line) for line in
                    (root / f"rank-{rank}.memory-context.jsonl").read_text().splitlines()]
        if len(records) != manifest["updates"] or len(contexts) != len(records):
            raise ValueError(f"rank {rank}: incomplete stress run")
        for index, (row, context) in enumerate(zip(records, contexts)):
            case = ordered_case(index * manifest["accumulation"], manifest["accumulation"], manifest["cases"])
            expected = [case["batch_size"], *case["latent_shape"], case["length"]]
            if (row["step"] != index + 1 or row["rank"] != rank or not math.isfinite(row["loss"])
                    or context["step"] != row["step"] or context["rank"] != rank
                    or row["shapes"] != [dict(shape=expected, count=manifest["accumulation"])]):
                raise ValueError(f"rank {rank}, update {index + 1}: wrong shape/order or nonfinite loss")
            if "health_interval" in manifest:
                interval = manifest["health_interval"]
                calls = int(bool(interval) and row["step"] % interval == 0)
                device = "cuda" if context["gpu_health_snapshot"] else "cpu"
                expected_health = dict(calls=calls, devices=[device] if calls else [], released=True)
                if context["health_snapshot"] != expected_health:
                    raise ValueError(f"rank {rank}, update {index + 1}: health boundary not exercised")
        peaks[str(rank)] = {key: max(row[key] for row in records)
                            for key in ("peak_allocated_bytes", "peak_reserved_bytes")}
        peaks[str(rank)]["estimated_min_headroom_bytes"] = min(
            context["total_bytes"] - row["peak_reserved_bytes"] - context["estimated_external_bytes"]
            for row, context in zip(records, contexts))
    result = dict(verified=True, ranks=manifest["ranks"], cases=len(manifest["cases"]),
                  cycles=manifest["cycles"], peaks=peaks, caveat=manifest["caveat"],
                  headroom_caveat="External allocations are sampled at update ends, not their peak; "
                                 "estimated headroom needs inspection, not an automatic safety verdict.")
    with (args.manifest.parent / "verification.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--config", action="append", required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--ranks", type=int, choices=range(1, 9), required=True)
    prep.add_argument("--cycles", type=int, default=2)
    prep.add_argument("--accumulation", type=int, required=True,
                      help="Intended stage accumulation; never inherit a base-config default")
    prep.add_argument("--health-interval", type=int,
                      help="Diagnostic override; use 1 to test every optimizer boundary")
    prep.add_argument("--cache-clear-interval", type=int,
                      help="Execution override; use 0 to stress without periodic allocator clearing")
    for name in ("train", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        if name == "train":
            command.add_argument("--train-arg", action="append", default=[])
    args = parser.parse_args()
    if args.command == "prepare" and args.cycles < 2:
        parser.error("at least two cycles are required to revisit every shape after state allocation")
    if args.command == "prepare" and args.accumulation < 1:
        parser.error("accumulation must be positive")
    if args.command == "prepare" and args.health_interval is not None and args.health_interval < 0:
        parser.error("health interval must be nonnegative")
    if args.command == "prepare" and args.cache_clear_interval is not None and args.cache_clear_interval < 0:
        parser.error("cache-clear interval must be nonnegative")
    globals()[args.command](args)


if __name__ == "__main__":
    main()

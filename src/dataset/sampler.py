"""Custom samplers and collate functions for precomputed training data."""

import random
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Any, Deque, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..utils.prompt_contract import MAX_SEQUENCE_LENGTH

from .length_metadata import RowLengthMetadata
from .captions import (
    CaptionPolicy,
    caption_probabilities_from_lengths,
)


def _weighted_index(rng, probabilities) -> int:
    """Draw an index from a probability sequence using ``rng``."""
    draw = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index
    return len(probabilities) - 1


@dataclass(frozen=True)
class LenBucket:
    """A closed retained-length bucket with its rank-local batch size."""

    max_length: int
    batch_size: int

    def __post_init__(self):
        if int(self.max_length) < 1:
            raise ValueError("LenBucket.max_length must be positive")
        if int(self.batch_size) < 1:
            raise ValueError("LenBucket.batch_size must be positive")


class BucketPlan:
    """Per-resolution closed length buckets.

    Each resolution maps to ordered ``LenBucket`` values.  A length is assigned
    to the first bucket whose ``max_length`` is greater than or equal to it, so
    a retained length equal to the final bound is still valid.
    """

    def __init__(self, by_resolution: Mapping[int, Sequence[LenBucket | Tuple[int, int]]]):
        if not by_resolution:
            raise ValueError("BucketPlan requires at least one resolution")
        normalized: Dict[int, Tuple[LenBucket, ...]] = {}
        for resolution_id, raw_buckets in by_resolution.items():
            resolution_id = int(resolution_id)
            if len(raw_buckets) == 0:
                raise ValueError(f"resolution {resolution_id} has no length buckets")
            buckets = tuple(
                bucket if isinstance(bucket, LenBucket) else LenBucket(*bucket)
                for bucket in raw_buckets
            )
            max_lengths = [bucket.max_length for bucket in buckets]
            if any(right <= left for left, right in zip(max_lengths, max_lengths[1:])):
                raise ValueError(
                    f"resolution {resolution_id} bucket bounds must be strictly increasing"
                )
            normalized[resolution_id] = buckets
        self._by_resolution = normalized

    @property
    def by_resolution(self) -> Mapping[int, Tuple[LenBucket, ...]]:
        return self._by_resolution

    def __getitem__(self, resolution_id: int) -> Tuple[LenBucket, ...]:
        return self.buckets_for(resolution_id)

    def buckets_for(self, resolution_id: int) -> Tuple[LenBucket, ...]:
        try:
            return self._by_resolution[int(resolution_id)]
        except KeyError as exc:
            raise ValueError(f"no bucket plan for resolution {resolution_id}") from exc

    def bucket_for(self, resolution_id: int, retained_length: int) -> Tuple[int, LenBucket]:
        retained_length = int(retained_length)
        if retained_length < 0 or retained_length > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"retained length must be in [0, {MAX_SEQUENCE_LENGTH}], got {retained_length}"
            )
        buckets = self.buckets_for(resolution_id)
        for index, bucket in enumerate(buckets):
            if retained_length <= bucket.max_length:
                return index, bucket
        raise ValueError(
            f"retained length {retained_length} does not fit resolution {resolution_id}"
        )

    @classmethod
    def uniform(
        cls, resolution_ids: Sequence[int], buckets: Sequence[LenBucket | Tuple[int, int]]
    ) -> "BucketPlan":
        buckets = tuple(buckets)
        return cls({int(resolution_id): buckets for resolution_id in resolution_ids})


def load_bucket_plan(path, resolution_ids=()) -> BucketPlan:
    """Read the canonical JSON file: resolution IDs map to named bucket objects."""
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError("bucket plan must be a nonempty object keyed by resolution ID")
    by_resolution = {}
    for resolution, buckets in raw.items():
        if not resolution.isdecimal() or str(int(resolution)) != resolution:
            raise ValueError(f"invalid resolution ID: {resolution!r}")
        if not isinstance(buckets, list) or not buckets:
            raise ValueError(f"resolution {resolution} needs a nonempty bucket list")
        parsed = []
        for bucket in buckets:
            if not isinstance(bucket, dict) or set(bucket) != {"max_length", "batch_size"}:
                raise ValueError("each bucket requires max_length and batch_size")
            if any(type(value) is not int for value in bucket.values()):
                raise ValueError("bucket bounds and batch sizes must be integers")
            parsed.append(LenBucket(**bucket))
        by_resolution[int(resolution)] = parsed
    if missing := set(map(int, resolution_ids)).difference(by_resolution):
        raise ValueError(f"bucket plan is missing metadata resolution IDs: {sorted(missing)}")
    return BucketPlan(by_resolution)


@dataclass(frozen=True)
class RowRef:
    """One row-level training draw with a caption selected inside that row."""

    dataset_id: int
    row_idx: int
    caption_idx: int
    resolution_id: int
    retained_length: int
    len_bucket_idx: int
    bucket_hi: int
    batch_id: int = -1

    def to_state(self) -> Tuple[int, int, int, int, int, int, int, int]:
        return (
            self.dataset_id,
            self.row_idx,
            self.caption_idx,
            self.resolution_id,
            self.retained_length,
            self.len_bucket_idx,
            self.bucket_hi,
            self.batch_id,
        )

    @classmethod
    def from_state(cls, value: Sequence[int]) -> "RowRef":
        if len(value) != 8:
            raise ValueError("stored RowRef must contain all eight fields")
        return cls(*(int(field) for field in value))


class RowDescriptorDataset(Dataset):
    """Resolve ``RowRef`` descriptors to one selected captioned row sample."""

    def __init__(
        self,
        entry_datasets: Sequence[Any],
        metadata: Optional[Sequence[RowLengthMetadata]] = None,
    ):
        if not entry_datasets:
            raise ValueError("RowDescriptorDataset requires at least one entry dataset")
        self.entry_datasets = list(entry_datasets)
        if metadata is not None:
            if isinstance(metadata, RowLengthMetadata):
                metadata = [metadata]
            if len(metadata) != len(self.entry_datasets):
                raise ValueError("metadata must have one entry per dataset")
            if any(len(dataset) != entry.num_rows for dataset, entry in zip(self.entry_datasets, metadata)):
                raise ValueError("dataset and row metadata lengths disagree")
        self.metadata = None if metadata is None else list(metadata)

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.entry_datasets)

    def __getitem__(self, row_ref: RowRef) -> Dict[str, Any]:
        if not isinstance(row_ref, RowRef):
            raise TypeError("RowDescriptorDataset expects a RowRef from its batch sampler")
        try:
            row = self.entry_datasets[row_ref.dataset_id][row_ref.row_idx]
        except IndexError as exc:
            raise IndexError(f"invalid row reference {row_ref}") from exc
        captions = row["captions"]
        if captions is None or not 0 <= row_ref.caption_idx < len(captions):
            raise ValueError(f"caption index does not exist for row reference {row_ref}")
        resolution_id = int(row["resolution_bucket_id"])
        if resolution_id != row_ref.resolution_id:
            raise ValueError(
                f"row resolution {resolution_id} disagrees with row reference {row_ref}"
            )
        return {
            "latents": torch.as_tensor(row["latents"]),
            "captions": str(captions[row_ref.caption_idx]),
            "dataset_id": int(row_ref.dataset_id),
            "row_idx": int(row_ref.row_idx),
            "caption_idx": int(row_ref.caption_idx),
            "resolution_bucket_id": resolution_id,
            "retained_length": int(row_ref.retained_length),
            "len_bucket_idx": int(row_ref.len_bucket_idx),
            "bucket_hi": int(row_ref.bucket_hi),
            "batch_id": int(row_ref.batch_id),
        }


class RowLengthQueueBatchSampler(Sampler[List[RowRef]]):
    """Infinite row-level mixed sampler with persistent length-bucket queues.

    Dataset choice occurs before row choice and follows ``dataset_weights``.
    Captions are never expanded into global items: after selecting a row, the
    existing caption curriculum selects exactly one caption inside that row.
    Queues are keyed only by ``(resolution_id, len_bucket_idx)`` so batches may
    mix datasets while retaining a static image/text shape.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        metadata: Sequence[RowLengthMetadata],
        bucket_plan: BucketPlan,
        dataset_weights: Optional[Sequence[float]] = None,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        initial_stage: float = 0.5,
        *,
        caption_policy: CaptionPolicy,
    ):
        if isinstance(metadata, RowLengthMetadata):
            metadata = [metadata]
        if not metadata:
            raise ValueError("at least one metadata entry is required")
        if bucket_plan is None:
            raise ValueError("bucket_plan is required")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        self.metadata = list(metadata)
        self.bucket_plan = bucket_plan
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self._rng = random.Random(seed)
        self._stage = min(max(float(initial_stage), 0.0), 1.0)
        self.caption_policy = caption_policy

        if dataset_weights is None:
            dataset_weights = [1.0] * len(self.metadata)
        if len(dataset_weights) != len(self.metadata):
            raise ValueError("dataset_weights must have one value per metadata entry")
        if any(float(weight) < 0 for weight in dataset_weights):
            raise ValueError("dataset_weights cannot be negative")
        self._dataset_weights = [float(weight) for weight in dataset_weights]

        self._cycles: List[List[int]] = []
        self._cursors: List[int] = []
        for dataset_id, entry in enumerate(self.metadata):
            local_rows = [
                int(row_idx)
                for row_idx in range(self.rank, entry.num_rows, self.num_replicas)
                if entry.caption_offsets[row_idx] < entry.caption_offsets[row_idx + 1]
            ]
            for row_idx in local_rows:
                self.bucket_plan.buckets_for(int(entry.resolution_ids[row_idx]))
            if self.shuffle:
                self._rng.shuffle(local_rows)
            self._cycles.append(local_rows)
            self._cursors.append(0)

        self._active_dataset_ids = [
            dataset_id
            for dataset_id, rows in enumerate(self._cycles)
            if rows and self._dataset_weights[dataset_id] > 0
        ]
        if not self._active_dataset_ids:
            raise ValueError("no rank-local rows with a positive dataset weight")

        self._queues: Dict[Tuple[int, int], Deque[RowRef]] = defaultdict(deque)
        self._ready_batches: Deque[List[RowRef]] = deque()
        self._inflight: Dict[int, List[RowRef]] = {}
        self._replay: Deque[List[RowRef]] = deque()
        self._next_batch_id = 0

    @property
    def stage(self) -> float:
        return self._stage

    def set_stage(self, stage: float) -> None:
        self._stage = min(max(float(stage), 0.0), 1.0)

    def _draw_dataset_id(self) -> int:
        weights = [self._dataset_weights[index] for index in self._active_dataset_ids]
        return self._rng.choices(self._active_dataset_ids, weights=weights, k=1)[0]

    def _next_row(self, dataset_id: int) -> int:
        cycle = self._cycles[dataset_id]
        cursor = self._cursors[dataset_id]
        if cursor >= len(cycle):
            cycle = list(cycle)
            if self.shuffle:
                self._rng.shuffle(cycle)
            self._cycles[dataset_id] = cycle
            cursor = 0
        row_idx = int(cycle[cursor])
        self._cursors[dataset_id] = cursor + 1
        return row_idx

    def _draw_row_ref(self) -> RowRef:
        dataset_id = self._draw_dataset_id()
        row_idx = self._next_row(dataset_id)
        entry = self.metadata[dataset_id]
        caption_slice = entry.row_slice(row_idx)
        probabilities = caption_probabilities_from_lengths(
            entry.prompt_lengths[caption_slice].tolist(),
            self.caption_policy.beta(self._stage),
            self.caption_policy.short_reserve,
            self.caption_policy.short_threshold,
        )
        caption_idx = _weighted_index(self._rng, probabilities)
        flat_caption_idx = caption_slice.start + caption_idx
        resolution_id = int(entry.resolution_ids[row_idx])
        retained_length = int(entry.prompt_lengths[flat_caption_idx])
        len_bucket_idx, bucket = self.bucket_plan.bucket_for(resolution_id, retained_length)
        return RowRef(
            dataset_id=dataset_id,
            row_idx=row_idx,
            caption_idx=caption_idx,
            resolution_id=resolution_id,
            retained_length=retained_length,
            len_bucket_idx=len_bucket_idx,
            bucket_hi=bucket.max_length,
        )

    def _enqueue_draw(self) -> None:
        row_ref = self._draw_row_ref()
        # Deliberately key by shape only, not by dataset id: rows from
        # different mix entries may share one micro-batch (the row-level mix
        # is set by dataset_weights at draw time) while every emitted batch
        # still has one static (resolution, padded-length) shape.
        key = (row_ref.resolution_id, row_ref.len_bucket_idx)
        queue = self._queues[key]
        queue.append(row_ref)
        # Micro-batches carry different sample counts because the batch size
        # is a per-bucket property, so a downstream mean over per-sample
        # losses must weight each micro-batch by its actual sample count — a
        # constant batch size would mis-weight every bucket whose local size
        # differs.
        batch_size = self.bucket_plan.buckets_for(row_ref.resolution_id)[
            row_ref.len_bucket_idx
        ].batch_size
        # Emit only full micro-batches; the incomplete tail stays queued for
        # the next draw of this bucket. Dropping a tail would change the fill
        # cadence of the bucket and discard rows already drawn into it,
        # shifting the sample mix the sampler emits over time.
        if len(queue) >= batch_size:
            batch_id = self._next_batch_id
            self._next_batch_id += 1
            batch = [replace(queue.popleft(), batch_id=batch_id) for _ in range(batch_size)]
            self._ready_batches.append(batch)

    def _next_ready_batch(self) -> List[RowRef]:
        while not self._ready_batches:
            self._enqueue_draw()
        return self._ready_batches.popleft()

    def __iter__(self) -> Iterator[List[RowRef]]:
        while self._replay:
            batch = self._replay.popleft()
            self._inflight[batch[0].batch_id] = batch
            yield batch

        while True:
            batch = self._next_ready_batch()
            self._inflight[batch[0].batch_id] = batch
            yield batch

    def ack_batch(self, batch_id: int) -> None:
        """Acknowledge a batch after its backward pass has completed.

        A generator cannot observe when its yielded batch has been consumed.
        Training acknowledges it explicitly so a checkpoint taken at an
        optimizer boundary does not replay a batch that was already trained.
        If a process dies before the acknowledgement, replaying that batch is
        the safe choice.
        """
        self._inflight.pop(int(batch_id), None)

    def __len__(self) -> int:
        raise TypeError("RowLengthQueueBatchSampler is an infinite batch stream")

    @staticmethod
    def _serialize_batches(batches: Sequence[Sequence[RowRef]]) -> List[List[Tuple[int, ...]]]:
        return [[row_ref.to_state() for row_ref in batch] for batch in batches]

    @staticmethod
    def _deserialize_batches(values: Sequence[Sequence[Sequence[int]]]) -> Deque[List[RowRef]]:
        return deque(
            [[RowRef.from_state(row_ref) for row_ref in batch] for batch in values]
        )

    def state_dict(self) -> Dict[str, Any]:
        """Return enough sampler state to resume without dropping queue tails."""
        # The caller saves this snapshot as one file per rank next to the
        # Accelerate checkpoint: Accelerate never persists this custom batch
        # sampler (its DataLoader stays outside prepare()), and no rank's
        # state could restore another's — each instance walks a different
        # stride-sharded row order under its own RNG stream.
        return {
            "version": self.STATE_VERSION,
            "stage": self._stage,
            "rng_state": self._rng.getstate(),
            "cycles": [list(cycle) for cycle in self._cycles],
            "cursors": list(self._cursors),
            "queues": {
                key: [row_ref.to_state() for row_ref in queue]
                for key, queue in self._queues.items()
                if queue
            },
            "ready_batches": self._serialize_batches(self._ready_batches),
            "inflight": self._serialize_batches(self._inflight.values()),
            "replay": self._serialize_batches(self._replay),
            "next_batch_id": self._next_batch_id,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore queues and replay unacknowledged emitted batches first.

        Same-stage recovery requires the same dataset entries. Resolution
        transitions construct a fresh sampler instead of adapting saved state.
        """
        if int(state.get("version", -1)) != self.STATE_VERSION:
            raise ValueError("unsupported RowLengthQueueBatchSampler state version")
        required = {"version", "stage", "rng_state", "cycles", "cursors", "queues",
                    "ready_batches", "inflight", "replay", "next_batch_id"}
        if missing := required.difference(state):
            raise ValueError(f"sampler state is missing fields: {sorted(missing)}")
        cycles = state["cycles"]
        cursors = state["cursors"]
        if len(cursors) != len(cycles) or len(cycles) != len(self._cycles):
            raise ValueError("state metadata entries do not match this sampler")
        if any(cursor < 0 or cursor > len(cycle) for cycle, cursor in zip(cycles, cursors)):
            raise ValueError("invalid row-cycle cursor in sampler state")

        progress = float(state["stage"])
        if not 0.0 <= progress <= 1.0:
            raise ValueError("invalid saved sampler progress")
        self._stage = progress
        self._rng.setstate(state["rng_state"])
        for dataset_id, (cycle, cursor) in enumerate(zip(cycles, cursors)):
            self._cycles[dataset_id] = [int(row_idx) for row_idx in cycle]
            self._cursors[dataset_id] = int(cursor)
        self._queues = defaultdict(deque)
        for key, refs in state["queues"].items():
            normalized_key = (int(key[0]), int(key[1]))
            self._queues[normalized_key].extend(RowRef.from_state(ref) for ref in refs)

        inflight = self._deserialize_batches(state["inflight"])
        replay = self._deserialize_batches(state["replay"])
        ready = self._deserialize_batches(state["ready_batches"])
        self._replay = deque((*inflight, *replay, *ready))
        self._ready_batches = deque()
        self._inflight = {}
        self._next_batch_id = int(state["next_batch_id"])


def row_length_collate_fn(
    batch: List[Dict[str, Any]], *, return_numpy: bool = False
) -> Dict[str, Any]:
    """Strict collate for row-level length-bucketed batches.

    With ``return_numpy=True`` every array payload leaves the worker as a
    numpy array instead of a torch tensor. numpy is pickled by value through
    the worker->main queue, while tensors are transferred through shared
    memory — which dies with bus errors on pods whose /dev/shm is tiny
    (Ascend nodes ship 64MB). The training loop converts back with
    ``torch.from_numpy`` after the queue crossing.
    """
    if not batch:
        raise ValueError("row_length_collate_fn requires a non-empty batch")

    def field(sample: Dict[str, Any], name: str) -> Any:
        if name not in sample:
            raise ValueError(f"row-level sample is missing field {name!r}")
        return sample[name]

    resolution_ids = [
        int(field(sample, "resolution_bucket_id")) for sample in batch
    ]
    len_bucket_ids = [
        int(field(sample, "len_bucket_idx")) for sample in batch
    ]
    bucket_his = [int(field(sample, "bucket_hi")) for sample in batch]
    batch_ids = [int(field(sample, "batch_id")) for sample in batch]
    # Fail here instead of repairing: every consumer downstream assumes a
    # micro-batch is exactly one (resolution, length bucket) padded to
    # bucket_hi, so a silently mixed batch would break that static-shape
    # contract (compile graphs, padding bounds, batch bookkeeping) somewhere
    # far from the actual error.
    for name, values in (
        ("resolution_bucket_id", resolution_ids),
        ("len_bucket_idx", len_bucket_ids),
        ("bucket_hi", bucket_his),
        ("batch_id", batch_ids),
    ):
        if len(set(values)) != 1:
            raise ValueError(f"mixed {name} values in one row-length batch")

    latents = [field(sample, "latents") for sample in batch]
    latent_shape = tuple(latents[0].shape)
    if any(tuple(latent.shape) != latent_shape for latent in latents[1:]):
        raise ValueError("mixed latent shapes in one row-length batch")
    captions = [field(sample, "captions") for sample in batch]
    if not all(isinstance(caption, str) for caption in captions):
        raise ValueError("row-level captions must be strings")
    retained_lengths = [
        int(field(sample, "retained_length")) for sample in batch
    ]
    if any(length < 1 or length > bucket_his[0] for length in retained_lengths):
        raise ValueError(
            "retained lengths must be positive and no greater than the bucket upper bound"
        )
    if any(bucket_id < 0 for bucket_id in len_bucket_ids):
        raise ValueError("length bucket indices must be non-negative")
    if any(batch_id < 0 for batch_id in batch_ids):
        raise ValueError("batch IDs must be non-negative")

    dataset_ids = [int(field(sample, "dataset_id")) for sample in batch]
    row_indices = [int(field(sample, "row_idx")) for sample in batch]

    if return_numpy:
        return {
            "latents": np.stack([np.asarray(latent) for latent in latents], axis=0),
            "captions": captions,
            "dataset_ids": np.asarray(dataset_ids, dtype=np.int64),
            "row_indices": np.asarray(row_indices, dtype=np.int64),
            "row_positions": list(zip(dataset_ids, row_indices)),
            "caption_indices": np.asarray(
                [int(field(sample, "caption_idx")) for sample in batch],
                dtype=np.int64,
            ),
            "resolution_bucket_ids": np.asarray(resolution_ids, dtype=np.int64),
            "retained_lengths": np.asarray(retained_lengths, dtype=np.int64),
            "len_bucket_idx": len_bucket_ids[0],
            "bucket_hi": bucket_his[0],
            "batch_id": batch_ids[0],
        }

    return {
        "latents": torch.stack([torch.as_tensor(latent) for latent in latents], dim=0),
        "captions": captions,
        "dataset_ids": torch.tensor(dataset_ids, dtype=torch.long),
        "row_indices": torch.tensor(row_indices, dtype=torch.long),
        # Which row of which dataset each sample came from, kept as host values:
        # per-draw bookkeeping (how many distinct rows a run has consumed, how
        # often a row repeats) then needs no device round-trip per micro-batch,
        # unlike the tensors above, which move to the compute device with the
        # rest of the batch.
        "row_positions": list(zip(dataset_ids, row_indices)),
        "caption_indices": torch.tensor(
            [int(field(sample, "caption_idx")) for sample in batch],
            dtype=torch.long,
        ),
        "resolution_bucket_ids": torch.tensor(resolution_ids, dtype=torch.long),
        "retained_lengths": torch.tensor(retained_lengths, dtype=torch.long),
        "len_bucket_idx": len_bucket_ids[0],
        "bucket_hi": bucket_his[0],
        "batch_id": batch_ids[0],
    }


def pad_text_to_hi(
    txt: torch.Tensor, txt_mask: torch.Tensor, hi: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad text tensors to an exact bucket upper bound without truncation."""
    import torch.nn.functional as F

    if txt.ndim != 3 or txt_mask.ndim != 2:
        raise ValueError("expected txt [B, L, D] and txt_mask [B, L]")
    if txt.shape[:2] != txt_mask.shape:
        raise ValueError("txt and txt_mask batch/sequence shapes must agree")
    hi = int(hi)
    if hi < 1:
        raise ValueError("bucket upper bound must be positive")
    sequence_length = int(txt.size(1))
    if sequence_length > hi:
        raise ValueError(
            f"encoded text length {sequence_length} exceeds bucket upper bound {hi}"
        )
    if sequence_length == hi:
        return txt, txt_mask
    pad = hi - sequence_length
    return F.pad(txt, (0, 0, 0, pad)), F.pad(txt_mask, (0, pad))

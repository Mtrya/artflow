"""Offline row-level caption length metadata."""

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional

import numpy as np

from .captions import _estimate_token_counts
from ..utils.prompt_contract import (
    DROP_IDX,
    MAX_SEQUENCE_LENGTH,
    PROMPT_TEMPLATE,
    RETAINED_MIN_LENGTH,
    SYSTEM_PROMPT,
)


METADATA_VERSION = "row-length-v2"
PROMPT_METADATA_VERSION = "qwen-prompt-v1"


def prompt_metadata_contract(num_rows: int) -> dict:
    return {
        "prompt_metadata_version": PROMPT_METADATA_VERSION,
        "num_rows": int(num_rows),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "drop_idx": DROP_IDX,
        "retained_min_length": RETAINED_MIN_LENGTH,
        "prompt_template": PROMPT_TEMPLATE,
        "system_prompt": SYSTEM_PROMPT,
    }


def _column(dataset: Any, name: str) -> List[Any]:
    try:
        return list(dataset[name])
    except (KeyError, TypeError, IndexError):
        return [dataset[index][name] for index in range(len(dataset))]


def _prompt_text(caption: str) -> str:
    return PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, user_prompt=caption)


def _tokenize_prompt_lengths(tokenizer: Any, prompts: List[str]) -> np.ndarray:
    """Tokenize complete prompts with the same cap used by ``encode_text``.

    This function intentionally runs in the caller's offline process.  In
    particular it does not create a process pool around a tokenizer or a GPU
    text encoder.
    """
    if not prompts:
        return np.empty(0, dtype=np.int64)

    encoded = tokenizer(
        prompts,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH + DROP_IDX,
        padding=False,
    )

    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    attention = None
    if isinstance(encoded, dict):
        attention = encoded.get("attention_mask")
    elif hasattr(encoded, "attention_mask"):
        attention = encoded.attention_mask

    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if attention is not None and hasattr(attention, "tolist"):
        attention = attention.tolist()

    lengths = []
    for index, ids in enumerate(input_ids):
        full_length = int(sum(attention[index])) if attention is not None else len(ids)
        full_length = min(full_length, MAX_SEQUENCE_LENGTH + DROP_IDX)
        lengths.append(max(full_length - DROP_IDX, RETAINED_MIN_LENGTH))
    return np.asarray(lengths, dtype=np.int64)


@dataclass
class RowLengthMetadata:
    """Flattened per-caption lengths indexed by row caption offsets.

    ``caption_offsets`` has one more element than the number of rows.  For row
    ``r``, caption metadata is stored in ``[caption_offsets[r]:
    caption_offsets[r + 1]]``.  ``prompt_lengths`` are the effective retained
    lengths after the complete prompt, ``DROP_IDX`` removal, and the 2048-token
    cap.  They are at least one so an empty retained sequence has the same
    shape contract online and offline. ``curriculum_lengths`` are the
    inexpensive heuristic counts used by the existing within-row caption
    curriculum.
    """

    resolution_ids: np.ndarray
    caption_offsets: np.ndarray
    curriculum_lengths: np.ndarray
    prompt_lengths: np.ndarray
    metadata_version: Any = METADATA_VERSION
    metadata_info: Optional[Mapping[str, Any]] = None

    def __post_init__(self):
        self.resolution_ids = np.asarray(self.resolution_ids, dtype=np.int64)
        self.caption_offsets = np.asarray(self.caption_offsets, dtype=np.int64)
        self.curriculum_lengths = np.asarray(self.curriculum_lengths, dtype=np.int64)
        self.prompt_lengths = np.asarray(self.prompt_lengths, dtype=np.int64)
        if self.caption_offsets.ndim != 1 or self.caption_offsets.size != self.resolution_ids.size + 1:
            raise ValueError("caption_offsets must have one more element than resolution_ids")
        if self.caption_offsets.size and self.caption_offsets[0] != 0:
            raise ValueError("caption_offsets must start at zero")
        if np.any(np.diff(self.caption_offsets) < 0):
            raise ValueError("caption_offsets must be non-decreasing")
        count = int(self.caption_offsets[-1]) if self.caption_offsets.size else 0
        if self.curriculum_lengths.size != count or self.prompt_lengths.size != count:
            raise ValueError("caption length arrays must match caption_offsets[-1]")
        if np.any(self.prompt_lengths < RETAINED_MIN_LENGTH) or np.any(
            self.prompt_lengths > MAX_SEQUENCE_LENGTH
        ):
            raise ValueError(
                f"prompt_lengths must be in [{RETAINED_MIN_LENGTH}, {MAX_SEQUENCE_LENGTH}]"
            )
        if self.metadata_info is not None:
            if not isinstance(self.metadata_info, Mapping):
                raise ValueError("metadata_info must be a mapping")
            self.metadata_info = dict(self.metadata_info)
            expected = prompt_metadata_contract(self.num_rows)
            for key, value in expected.items():
                if self.metadata_info.get(key) != value:
                    raise ValueError(f"metadata_info has invalid {key}")

    @property
    def num_rows(self) -> int:
        return int(self.resolution_ids.size)

    @property
    def num_captions(self) -> int:
        return int(self.prompt_lengths.size)

    @property
    def lengths(self) -> np.ndarray:
        """Compatibility alias for the retained prompt lengths."""
        return self.prompt_lengths

    @property
    def version(self) -> Any:
        return self.metadata_version

    def validate_against_dataset(self, dataset: Any) -> None:
        """Reject row or prompt-contract mismatches before training starts."""
        if len(dataset) != self.num_rows:
            raise ValueError(
                f"metadata rows ({self.num_rows}) do not match dataset rows ({len(dataset)})"
            )
        if self.metadata_info is None:
            raise ValueError(
                "metadata has no prompt contract; rebuild it with "
                "build_from_dataset() or ensure_sidecar()"
            )
        expected = prompt_metadata_contract(self.num_rows)
        for key, value in expected.items():
            if self.metadata_info.get(key) != value:
                raise ValueError(f"metadata prompt contract mismatch for {key}")

    def row_slice(self, row_idx: int) -> slice:
        row_idx = int(row_idx)
        if row_idx < 0 or row_idx >= self.num_rows:
            raise IndexError(row_idx)
        return slice(int(self.caption_offsets[row_idx]), int(self.caption_offsets[row_idx + 1]))

    def save(self, path: str | Path) -> None:
        """Save metadata as a compact, portable NumPy archive."""
        path = Path(path)
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                resolution_ids=self.resolution_ids,
                caption_offsets=self.caption_offsets,
                curriculum_lengths=self.curriculum_lengths,
                prompt_lengths=self.prompt_lengths,
                metadata_version=np.asarray(self.metadata_version),
                metadata_info=np.asarray(
                    json.dumps(self.metadata_info or prompt_metadata_contract(self.num_rows),
                               sort_keys=True)
                ),
            )

    @classmethod
    def load(cls, path: str | Path) -> "RowLengthMetadata":
        """Load and validate a metadata archive."""
        with np.load(Path(path), allow_pickle=False) as data:
            required = {
                "resolution_ids",
                "caption_offsets",
                "curriculum_lengths",
                "prompt_lengths",
                "metadata_version",
            }
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"metadata is missing fields: {sorted(missing)}")
            version = data["metadata_version"]
            if np.ndim(version) == 0:
                version = version.item()
            else:
                version = version.tolist()
            metadata_info = None
            if "metadata_info" in data.files:
                raw_info = data["metadata_info"]
                raw_info = raw_info.item() if np.ndim(raw_info) == 0 else raw_info.tolist()
                metadata_info = json.loads(raw_info)
            return cls(
                resolution_ids=data["resolution_ids"],
                caption_offsets=data["caption_offsets"],
                curriculum_lengths=data["curriculum_lengths"],
                # v1 sidecars used zero for an empty retained sequence; normalize
                # that legacy representation to the current one-token contract.
                prompt_lengths=np.maximum(
                    data["prompt_lengths"], RETAINED_MIN_LENGTH
                ),
                metadata_version=version,
                metadata_info=metadata_info,
            )

    @classmethod
    def from_hf_dataset(
        cls,
        dataset: Any,
        tokenizer: Any,
        caption_column: str = "captions",
        resolution_column: str = "resolution_bucket_id",
        metadata_version: Any = METADATA_VERSION,
        tokenizer_batch_size: int = 256,
    ) -> "RowLengthMetadata":
        """Build metadata from every row without retaining all prompt strings.

        The row metadata arrays are small, but the fully formatted prompts can
        be several gigabytes for a million-row corpus.  Tokenize bounded
        batches while iterating through the dataset so the builder remains a
        CPU-only, low-memory preprocessing step.
        """
        tokenizer_batch_size = int(tokenizer_batch_size)
        if tokenizer_batch_size < 1:
            raise ValueError("tokenizer_batch_size must be positive")

        resolution_ids: List[int] = []
        offsets = [0]
        curriculum_lengths: List[int] = []
        retained_lengths: List[int] = []
        prompt_buffer: List[str] = []

        def flush_prompts() -> None:
            if prompt_buffer:
                retained_lengths.extend(
                    _tokenize_prompt_lengths(tokenizer, prompt_buffer)
                )
                prompt_buffer.clear()

        for row in dataset:
            captions = row[caption_column]
            if captions is None:
                captions = []
            elif isinstance(captions, str):
                captions = [captions]
            else:
                captions = list(captions)
            captions = [str(caption) for caption in captions]
            resolution_ids.append(int(row[resolution_column]))
            curriculum_lengths.extend(_estimate_token_counts(captions))
            prompt_buffer.extend(_prompt_text(caption) for caption in captions)
            offsets.append(offsets[-1] + len(captions))
            if len(prompt_buffer) >= tokenizer_batch_size:
                flush_prompts()
        flush_prompts()

        resolution_ids_array = np.asarray(resolution_ids, dtype=np.int64)
        metadata_info = prompt_metadata_contract(resolution_ids_array.size)
        metadata_info.update(
            {
                "caption_column": caption_column,
                "resolution_column": resolution_column,
                "tokenizer_batch_size": tokenizer_batch_size,
            }
        )
        return cls(
            resolution_ids=resolution_ids_array,
            caption_offsets=np.asarray(offsets, dtype=np.int64),
            curriculum_lengths=np.asarray(curriculum_lengths, dtype=np.int64),
            prompt_lengths=np.asarray(retained_lengths, dtype=np.int64),
            metadata_version=metadata_version,
            metadata_info=metadata_info,
        )

    @classmethod
    def from_dataset(cls, dataset: Any, tokenizer: Any, **kwargs) -> "RowLengthMetadata":
        return cls.from_hf_dataset(dataset, tokenizer, **kwargs)

    @classmethod
    def from_hf(cls, dataset: Any, tokenizer: Any, **kwargs) -> "RowLengthMetadata":
        return cls.from_hf_dataset(dataset, tokenizer, **kwargs)

    @classmethod
    def build(cls, dataset: Any, tokenizer: Any, **kwargs) -> "RowLengthMetadata":
        return cls.from_hf_dataset(dataset, tokenizer, **kwargs)


def build_row_length_metadata(dataset: Any, tokenizer: Any, **kwargs) -> RowLengthMetadata:
    """Functional wrapper for offline metadata generation."""
    return RowLengthMetadata.from_hf_dataset(dataset, tokenizer, **kwargs)


build_length_metadata = build_row_length_metadata


# ---------------------------------------------------------------------------
# Dataset-companion sidecar API.
#
# Why this is a companion file and not a runtime computation: the length-bucket
# sampler must decide which bucket a row belongs to without loading any sample.
# The queues are prefilled, so every drawn row must immediately reveal the
# retained-token count of its selected caption.  A row can hold several
# captions (the curriculum picks one caption inside the row), so computing that
# on the fly would mean reading the whole row's captions and tokenizing them on
# the sampling hot path.  The offline lengths are therefore stored once, next
# to the dataset it describes, and the sampler only indexes into the arrays.
#
# Tokenization here MUST match training exactly (same tokenizer checkpoint,
# same PROMPT_TEMPLATE/SYSTEM_PROMPT prompt, same max_length=DROP_IDX +
# MAX_SEQUENCE_LENGTH truncation cap, same DROP_IDX removal, same
# RETAINED_MIN_LENGTH floor as encode_text in src/utils/encode_text.py).  A
# mismatch between the two paths never raises: it silently misbuckets rows.
# Both paths share the constants in src/utils/prompt_contract.py and load the
# same tokenizer, which is why build_from_dataset takes the tokenizer_path the
# trainer is configured with rather than guessing.
# ---------------------------------------------------------------------------

SIDECAR_FILENAME = "length_metadata.npz"


def sidecar_path(dataset_dir: str) -> Path:
    """Companion-file location: ``<dataset_dir>/length_metadata.npz``."""
    return Path(dataset_dir) / SIDECAR_FILENAME


def _load_text_columns(dataset_dir: str) -> Any:
    """Open a saved dataset's Arrow shards and keep only text-side columns.

    Precomputed datasets are commonly saved with ``format=torch`` and large
    latent columns.  The metadata pass only needs captions and resolution IDs,
    so the shards are opened directly and projected to those two columns:
    this stays CPU-only, never touches the latent column, and never re-applies
    a saved torch formatter.

    Shards are read in the order recorded in ``state.json``, which is the row
    order training's ``load_from_disk`` will see; a lexicographic glob over
    ``data-*.arrow`` is the fallback for directories without a state file.
    """
    from datasets import Dataset, concatenate_datasets

    root = Path(dataset_dir)
    state_path = root / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        filenames = [item["filename"] for item in state.get("_data_files", [])]
    else:
        filenames = []
    if not filenames:
        filenames = sorted(path.name for path in root.glob("data-*.arrow"))
    if not filenames:
        raise ValueError(f"no HF Arrow data files found under {root}")

    shards = [
        Dataset.from_file(str(root / filename)).select_columns(
            ["captions", "resolution_bucket_id"]
        )
        for filename in filenames
    ]
    return shards[0] if len(shards) == 1 else concatenate_datasets(shards)


def build_from_dataset(dataset_dir: str, tokenizer_path: str) -> RowLengthMetadata:
    """Scan a saved dataset's Arrow shards and build its row-length metadata.

    Tokenizes every caption with the tokenizer at ``tokenizer_path`` (loaded
    with ``local_files_only=True``; no network access is attempted).  Only the
    ``captions`` and ``resolution_bucket_id`` columns are read: latents are
    never loaded and no GPU is required.  The returned metadata rows are, by
    construction, in the exact row order of the saved dataset.
    """
    from transformers import AutoTokenizer

    dataset = _load_text_columns(dataset_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return RowLengthMetadata.from_hf_dataset(dataset, tokenizer)


def ensure_sidecar(dataset_dir: str, tokenizer_path: str) -> RowLengthMetadata:
    """Load the dataset's companion file, building and writing it on first use.

    If ``<dataset_dir>/length_metadata.npz`` exists it is loaded (a torn or
    unreadable file from an interrupted write is rebuilt from the shards);
    otherwise ``build_from_dataset`` runs and the result is persisted at
    ``sidecar_path(dataset_dir)`` before being returned.
    """
    path = sidecar_path(dataset_dir)
    if path.is_file():
        try:
            return RowLengthMetadata.load(path)
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            # Partial write or an unreadable archive: the companion is derived
            # data, so rebuilding it from the shards is always safe.
            metadata = build_from_dataset(dataset_dir, tokenizer_path)
            metadata.save(path)
            return metadata
    metadata = build_from_dataset(dataset_dir, tokenizer_path)
    metadata.save(path)
    return metadata

"""
Fixed held-out eval-loss probe for ablation comparability.

The probe holds one fixed slice of a held-out dataset: the same images, the
same caption per image and caption-length band, the same noise and the same
timestep grid at every probe of every run, so ``eval/loss`` curves are directly
comparable across runs.  Forward-only; every rank builds the same probe
(deterministic, acts as a natural sync point) and the main process logs it.

Captions are chosen by retained length, not by position in the row's caption
list.  The bands are this module's own (``LENGTH_BINS``): doublings of the
token count, on the same log2 scale the training-time length weighting's curve
lives on.  A caption's retained length is read from the dataset's row-length
sidecar (``length_metadata.npz`` beside the dataset), which holds the token
counts the training prompt contract itself produces.  Within a band a row
always contributes the earliest caption of that row falling in the band, i.e.
the lowest caption index: appending another caption to a row cannot displace
the one already chosen, so a row that gains a longer caption later keeps
contributing the caption it contributed before.

A row that holds captions in several bands contributes one sample per band.
Each band is filled independently up to the sample budget, so an image is in
every band it serves unless that band was already full; the images that made it
into two or more bands are counted and reported, which is what makes a long
band versus short band readout a paired comparison over the same images.  Noise
and timesteps belong to the image, not to the band, so an image's samples
differ only in the caption the model is conditioned on.

Every reported loss is weighted by sample count.  Batches are cut by latent
shape and therefore have different sizes, so averaging batch means would let a
small batch count as much as a large one; per-sample losses are accumulated
instead.  Each band also reports the number of samples it was measured on,
because a band that could not be filled to the requested size gives a noisier
number: the shortfall is reported rather than silently measured on less data.

A dataset without a usable sidecar falls back to one implicit caption per row
and reports the aggregate metrics only; the omission is stated on stdout,
because that run's curve cannot be read by caption length.
"""

import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..flow.paths import shift_timesteps
# Eval-loss caption-length bins, uniform on the log2 scale - the same scale
# the training-time length weighting's curve lives on.  Grouping the probe
# loss by caption length answers "is the model worse on long captions"; the
# edges are doublings because any finer or uneven set would invent boundaries
# the training itself has no use for.  The last bin ends at the prompt
# contract's 2048-token cap.
LENGTH_BINS: Tuple[Tuple[str, int, int], ...] = (
    ("le128", 1, 128),
    ("129_256", 129, 256),
    ("257_512", 257, 512),
    ("513_1024", 513, 1024),
    ("1025_2048", 1025, 2048),
)
from ..utils.encode_text import encode_text


def _autocast_ctx(device: torch.device):
    """Match training-time numerics (accelerate bf16 mixed precision)."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    import contextlib

    return contextlib.nullcontext()


def _announce(message: str) -> None:
    """Print a probe-level notice once per run, not once per rank."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    print(message)


# Captions per encode_text call when the probe pre-encodes its caption set.
# One forward keeps every layer's hidden state of the whole call live at once
# (output_hidden_states), so the peak scales with batch x padded length; 64
# captions at the 2048-token cap peak around 8 GB, which fits beside the
# training state on a 48 GB card.
_ENCODE_CHUNK = 64


def _pad_to_width(t: torch.Tensor, width: int) -> torch.Tensor:
    pad = width - t.size(1)
    if pad == 0:
        return t
    if t.dim() == 3:
        return torch.nn.functional.pad(t, (0, 0, 0, pad))
    return torch.nn.functional.pad(t, (0, pad))


def _encode_caption_chunks(captions, text_encoder, tokenizer, pooling, exit_layer):
    """Encode the probe captions in chunks and re-pad to the widest chunk."""
    if not captions:
        raise ValueError("captions must contain at least one caption.")
    parts_txt, parts_mask, parts_pooled = [], [], []
    for start in range(0, len(captions), _ENCODE_CHUNK):
        txt, mask, pooled = encode_text(
            captions[start : start + _ENCODE_CHUNK],
            text_encoder,
            tokenizer,
            pooling,
            exit_layer=exit_layer,
        )
        parts_txt.append(txt)
        parts_mask.append(mask)
        if pooled is not None:
            parts_pooled.append(pooled)
    width = max(t.size(1) for t in parts_txt)
    txt = torch.cat([_pad_to_width(t, width) for t in parts_txt], dim=0)
    mask = torch.cat([_pad_to_width(m, width) for m in parts_mask], dim=0)
    pooled = torch.cat(parts_pooled, dim=0) if parts_pooled else None
    return txt, mask, pooled


def _caption_rows(caption_offsets: np.ndarray) -> np.ndarray:
    """Row index of every flat caption index (the sidecar's CSR layout)."""
    count = int(caption_offsets[-1]) if caption_offsets.size else 0
    return np.searchsorted(
        caption_offsets, np.arange(count, dtype=np.int64), side="right"
    ) - 1


def _band_candidates(
    caption_offsets: np.ndarray,
    prompt_lengths: np.ndarray,
    canonical_rank: np.ndarray,
    num_samples: int,
    bins: Sequence[Tuple[str, int, int]] = LENGTH_BINS,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Rows drawn for each band, as ``(band name, row indices, caption indices)``.

    A row contributes at most one caption to a band: the first one in its
    caption list that falls inside the band; the returned caption index is
    local to that row.  Rows are then kept in the probe's canonical row order
    and cut at ``num_samples`` per band, so the draw is a pure function of the
    sidecar, the canonical order and that budget — never of how many samples
    another band happened to fill.
    """
    caption_row = _caption_rows(caption_offsets)
    selection: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for name, low, high in bins:
        inside = np.flatnonzero((prompt_lengths >= low) & (prompt_lengths <= high))
        if inside.size == 0:
            selection.append((name, inside, inside))
            continue
        rows = caption_row[inside]
        # Rows are non-decreasing over that (caption-index ordered) slice, so
        # the first occurrence of each row is its earliest caption in the band.
        _, first = np.unique(rows, return_index=True)
        order = np.argsort(canonical_rank[rows[first]], kind="stable")[:num_samples]
        rows = rows[first][order]
        selection.append((name, rows, inside[first][order] - caption_offsets[rows]))
    return selection


def _load_sidecar(dataset_path: str, dataset: Any) -> Tuple[Any, str]:
    """Row-length metadata for the probe's dataset, plus why it is unusable.

    A sidecar that does not describe this dataset is worse than no sidecar: the
    band of a row would be read off another row's lengths.  Missing, unreadable
    and mismatched files all fall back instead of failing the run.
    """
    from ..dataset.length_metadata import RowLengthMetadata, sidecar_path

    path = sidecar_path(dataset_path)
    try:
        metadata = RowLengthMetadata.load(path)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, AttributeError) as exc:
        return None, f"cannot read {path} ({type(exc).__name__}: {exc})"
    try:
        metadata.validate_against_dataset(dataset)
    except ValueError as exc:
        return None, f"{path} does not describe this dataset ({exc})"
    return metadata, ""


def _load_rows(
    dataset: Any, rows: Sequence[int]
) -> Dict[int, Tuple[List[str], torch.Tensor]]:
    """Read the captions and raw latents of ``rows`` (one arrow read each)."""
    loaded: Dict[int, Tuple[List[str], torch.Tensor]] = {}
    for row in sorted(int(row) for row in rows):
        item = dataset[row]
        loaded[row] = (
            [str(caption) for caption in item["captions"]],
            torch.as_tensor(item["latents"]).float(),
        )
    return loaded


def _row_with_other_caption_count(
    metadata: Any, loaded: Dict[int, Tuple[List[str], torch.Tensor]]
) -> Optional[int]:
    """A loaded row the sidecar disagrees with, if the sidecar is stale.

    Per-row caption counts, not just the row count, pin the sidecar to the
    caption lists the probe is reading: appending captions to rows leaves the
    row count unchanged but shifts every later row's lengths.  Only the rows the
    probe uses are checked, so the check stays bounded by the probe's size.
    """
    offsets = metadata.caption_offsets
    for row in sorted(loaded):
        described = int(offsets[row + 1]) - int(offsets[row])
        if described != len(loaded[row][0]):
            return row
    return None


class EvalLossProbe:
    """
    Fixed-set flow-matching loss probe.

    Holds a fixed slice of the eval dataset: normalized latents, pre-encoded
    text (fixed caption choice, no dropout), per-sample fixed noise, and a fixed
    timestep grid.  Samples are drawn per caption-length band, and each band
    reports its own loss and sample count.
    """

    def __init__(
        self,
        dataset_path: str,
        text_encoder: Any,
        tokenizer: Any,
        pooling: bool,
        exit_layer: Optional[int],
        vae_mean: torch.Tensor,
        vae_std: torch.Tensor,
        num_samples: int = 512,
        batch_size: int = 64,
        t_grid: Tuple[float, ...] = (0.15, 0.4, 0.65, 0.9),
        seed: int = 123,
        device: Optional[torch.device] = None,
    ):
        """Build the probe. ``num_samples`` is per caption-length band.

        An image that serves several bands is stored once and contributes one
        sample per band, so the probe holds at most ``num_samples`` samples per
        band.  Bands that cannot be filled to that size are measured on what
        the dataset has, and the gap is reported.
        """
        from datasets import load_from_disk

        if device is None:
            device = next(text_encoder.parameters()).device
        self.device = device
        self.batch_size = batch_size
        self.t_grid = tuple(t_grid)
        self.num_samples = int(num_samples)

        dataset = load_from_disk(dataset_path)
        total_rows = len(dataset)

        # Canonical row order: the same permutation at every probe, derived
        # from the seed alone so it does not depend on how the dataset library
        # happens to shuffle.
        permutation = np.random.default_rng(seed).permutation(total_rows)
        canonical_rank = np.empty(total_rows, dtype=np.int64)
        canonical_rank[permutation] = np.arange(total_rows, dtype=np.int64)

        metadata, unusable = _load_sidecar(dataset_path, dataset)
        # (dataset row, caption index or None for the positional rule, band)
        entries: List[Tuple[int, Optional[int], Optional[str]]] = []
        loaded: Dict[int, Tuple[List[str], torch.Tensor]] = {}
        while metadata is not None:
            entries = []
            for name, rows, captions in _band_candidates(
                metadata.caption_offsets,
                metadata.prompt_lengths,
                canonical_rank,
                self.num_samples,
            ):
                for row, caption in zip(rows.tolist(), captions.tolist()):
                    entries.append((row, caption, name))
            if not entries:
                # A caption longer than the last band is kept by precompute
                # rather than cut to it, so an eval set can hold none at all.
                metadata = None
                unusable = (
                    "no caption of this dataset has a retained length inside a "
                    "reported band "
                    f"({', '.join(name for name, _, _ in LENGTH_BINS)})"
                )
                break
            loaded = _load_rows(dataset, {row for row, _, _ in entries})
            missing_row = _row_with_other_caption_count(metadata, loaded)
            if missing_row is None:
                break
            # Captions can be appended to a dataset without its sidecar being
            # rebuilt; the row count still matches, but every later row's band
            # would be read off the wrong lengths.
            metadata = None
            unusable = (
                f"row {missing_row} holds {len(loaded[missing_row][0])} captions "
                "where the length sidecar describes a different number"
            )
        if metadata is None:
            _announce(
                f"[eval-loss] caption bands are off ({unusable}). One caption is "
                "taken per row by list position and no per-band metric is "
                "reported; this probe's loss cannot be read by caption length."
            )
            entries = [
                (int(permutation[position]), None, None)
                for position in range(min(self.num_samples, total_rows))
            ]
            loaded = _load_rows(dataset, {row for row, _, _ in entries})
        self.banded = metadata is not None

        # Distinct rows in canonical order: a row that serves several bands is
        # loaded once, and its noise is drawn once, so both of its samples see
        # the identical noise and the identical shifted timestep.
        unique_rows = sorted(
            loaded, key=lambda row: int(canonical_rank[row])
        )

        # Normalize latents exactly like training: z = (z - mean) / std
        mean = vae_mean.float().cpu().squeeze()
        std = vae_std.float().cpu().squeeze()
        gen = torch.Generator().manual_seed(seed)
        latent_by_row: Dict[int, torch.Tensor] = {}
        noise_by_row: Dict[int, torch.Tensor] = {}
        for row in unique_rows:
            z = loaded[row][1]
            z = (z - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)
            latent_by_row[row] = z
            noise_by_row[row] = torch.randn(z.shape, generator=gen)

        self.captions: List[str] = []
        self.bands: List[Optional[str]] = []
        self.sample_rows: List[int] = []
        self.latents: List[torch.Tensor] = []
        self.noise: List[torch.Tensor] = []
        for row, caption_index, band in entries:
            captions = loaded[row][0]
            if caption_index is None:
                caption_index = 1 if len(captions) > 1 else 0
            self.captions.append(captions[caption_index])
            self.bands.append(band)
            self.sample_rows.append(row)
            self.latents.append(latent_by_row[row])
            self.noise.append(noise_by_row[row])

        # Group by latent shape: buckets have different HxW and cannot be stacked
        # into one batch.
        shape_groups: Dict[Tuple[int, int], List[int]] = {}
        for idx, z in enumerate(self.latents):
            shape_groups.setdefault((z.shape[-2], z.shape[-1]), []).append(idx)
        self.groups = list(shape_groups.values())

        # Slot 0 aggregates every sample; each band owns one slot after it.
        self.band_names = [name for name, _, _ in LENGTH_BINS] if self.banded else []
        slots = {name: index + 1 for index, name in enumerate(self.band_names)}
        self._slots = [slots.get(band, 0) for band in self.bands]

        self.band_sample_counts = {name: 0 for name in self.band_names}
        images_per_band: Dict[int, int] = {}
        for band, row in zip(self.bands, self.sample_rows):
            if band is not None:
                self.band_sample_counts[band] += 1
                images_per_band[row] = images_per_band.get(row, 0) + 1
        self.paired_images = sum(1 for count in images_per_band.values() if count > 1)
        self.band_shortfall = {
            name: max(self.num_samples - count, 0)
            for name, count in self.band_sample_counts.items()
        }
        if self.banded:
            short = {name: gap for name, gap in self.band_shortfall.items() if gap}
            if short:
                missing = ", ".join(
                    f"{name} needs {gap} more" for name, gap in short.items()
                )
                _announce(
                    "[eval-loss] caption-length bands below the requested "
                    f"{self.num_samples} samples: {missing}"
                )

        # Pre-encode text once (fixed captions; identical across probes and
        # runs).  The encode runs in chunks: one forward keeps every layer's
        # hidden state for the whole call live at once (output_hidden_states),
        # so a single call over a full probe - five length bands at the
        # 512-sample budget - does not fit on the card.  Chunks re-pad to the
        # widest chunk, which changes no embedding: right padding is masked.
        txt, txt_mask, txt_pooled = _encode_caption_chunks(
            self.captions, text_encoder, tokenizer, pooling, exit_layer
        )
        self.txt = txt.cpu()
        self.txt_mask = txt_mask.cpu()
        self.txt_pooled = txt_pooled.cpu() if txt_pooled is not None else None

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module) -> Dict[str, float]:
        was_training = model.training
        model.eval()

        t_count = len(self.t_grid)
        slot_count = 1 + len(self.band_names)
        sums = np.zeros((t_count, slot_count), dtype=np.float64)
        counts = np.zeros((t_count, slot_count), dtype=np.int64)

        for group in self.groups:
            for gstart in range(0, len(group), self.batch_size):
                idxs = group[gstart : gstart + self.batch_size]
                z1 = torch.stack([self.latents[i] for i in idxs]).to(
                    self.device, torch.bfloat16
                )
                z0 = torch.stack([self.noise[i] for i in idxs]).to(
                    self.device, torch.bfloat16
                )
                txt = self.txt[idxs].to(self.device)
                txt_mask = self.txt_mask[idxs].to(self.device)
                txt_pooled = (
                    self.txt_pooled[idxs].to(self.device)
                    if self.txt_pooled is not None
                    else None
                )
                bs = z1.shape[0]
                slots = [self._slots[i] for i in idxs]

                for ti, t_val in enumerate(self.t_grid):
                    t = torch.full((bs,), t_val, device=self.device)
                    t = shift_timesteps(t, z1)
                    z_t = (1.0 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * z1
                    with _autocast_ctx(self.device):
                        out = model(
                            z_t, t, txt=txt, txt_pooled=txt_pooled, txt_mask=txt_mask
                        )
                    # Per-sample losses rather than one batch mean: a batch may
                    # mix bands, and a sample's weight is its sample count.
                    per_sample = torch.nn.functional.mse_loss(
                        out.float(), (z1 - z0).float(), reduction="none"
                    ).flatten(1).mean(dim=1)
                    for value, slot in zip(per_sample.tolist(), slots):
                        sums[ti, 0] += value
                        counts[ti, 0] += 1
                        sums[ti, slot] += value
                        counts[ti, slot] += 1

        if was_training:
            model.train()

        metrics: Dict[str, float] = {}
        metrics["eval/loss"] = sums[:, 0].sum() / max(counts[:, 0].sum(), 1)
        for ti, t_val in enumerate(self.t_grid):
            tag = f"{t_val:.2f}".replace(".", "").ljust(3, "0")
            metrics[f"eval/loss_t{tag}"] = sums[ti, 0] / max(counts[ti, 0], 1)
        for slot, name in enumerate(self.band_names, start=1):
            metrics[f"eval/samples/band_{name}"] = float(self.band_sample_counts[name])
            metrics[f"eval/shortfall/band_{name}"] = float(self.band_shortfall[name])
            if counts[:, slot].sum():
                metrics[f"eval/loss/band_{name}"] = (
                    sums[:, slot].sum() / counts[:, slot].sum()
                )
        if self.banded:
            metrics["eval/paired_images"] = float(self.paired_images)
        return metrics

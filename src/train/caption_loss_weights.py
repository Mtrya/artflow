"""Per-sample loss weights read off a caption's retained length.

The captions a step draws are mostly short, so a plain mean over its samples
spends most of its gradient on short captions: a long caption receives a share
of that gradient proportional to how rare it is.  This module turns that share
into a deliberate choice by giving each sample a weight that is a function of
its caption's retained length.

The weight is a curve, not a lookup table: ``max(1, log2(L / reference))``
evaluated at the sample's own length.  A table of per-band multipliers would
invent boundaries the training has no use for and would force a code change
every time the caption length cap moved; the curve extends itself to whatever
length shows up.

Two rules complete the weighting:

* A caption dropped for classifier-free guidance keeps weight 1.0.  Such a
  sample carries no statement about caption length, and the curve speaks about
  how much a caption's length should matter, not about how the unconditional
  examples should be weighted.
* Lengths are clamped to at least 1 before the logarithm, so an empty retained
  caption still lands on the floor of 1.0.

What the weights are normalized by is the subject of the second half of this
module.  A step's gradient is ``sum(w_i * loss_i) / sum(w_i)`` over the samples
of the step, never ``sum(w_i * loss_i)`` and never an average of the
micro-batches' own means; :class:`StepLossAccumulator` is where that is kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence

if TYPE_CHECKING:  # torch is imported where the tensors are built, so that
    import torch      # reading a config needs no torch


CURVES = ("none", "log2")


@dataclass(frozen=True)
class MicroBatchWeights:
    """What one micro-batch's caption-length weights mean for its loss.

    ``values`` holds one weight per sample, and is None when weighting is off.
    ``tensor`` carries those weights on the compute device and is None whenever
    the loss does not need them: either because weighting is off, or because
    every sample of the micro-batch has the same weight, in which case the
    weighted mean of the micro-batch is a plain rescale of the mean the
    algorithm already reduces and the two agree exactly.  ``total`` is the sum
    of the weights: the share of the optimizer step this micro-batch
    contributes to its normalization, and the micro-batch's sample count when
    nothing is reweighted.
    """

    values: Optional[List[float]]
    tensor: Optional[torch.Tensor]
    total: float


class CaptionLossWeights:
    """A weight per sample, as a function of its retained caption length.

    ``curve`` selects the function: ``none`` (every weight is 1.0, weighting
    off) or ``log2`` (``max(1, log2(L / reference))``).  Anything else is
    rejected rather than repaired: a silently corrected curve would change
    what a run measures without changing what its config says.
    """

    def __init__(self, curve: str = "none", reference: int = 128):
        if curve not in CURVES:
            raise ValueError(
                f"caption loss weight curve must be one of {CURVES}, got "
                f"{curve!r}"
            )
        if int(reference) < 2:
            raise ValueError(
                f"caption loss weight reference must be >= 2, got {reference!r}"
            )
        self.curve = curve
        self.reference = int(reference)

    @property
    def enabled(self) -> bool:
        """Whether any sample is weighted away from 1.0.

        The ``none`` curve makes the weighted mean the arithmetic mean, so it
        is the curve that switches the weighting off, and the off state is the
        default.
        """
        return self.curve != "none"

    def weight_for(self, retained_length: int) -> float:
        """The multiplier a sample with this retained caption length receives."""
        if not self.enabled:
            return 1.0
        length = max(int(retained_length), 1)
        return max(1.0, math.log2(length / self.reference))

    def for_micro_batch(
        self,
        retained_lengths: Sequence[int],
        dropped: Sequence[bool],
        device: Optional[torch.device] = None,
    ) -> MicroBatchWeights:
        """The weights one micro-batch's samples apply to its loss.

        ``dropped`` is the caption-dropout mask the step actually used, so a
        sample the model was not conditioned on is not weighted by the length
        of a caption it did not see.  The weights are plain host values: a
        curve evaluation costs no device round-trip, and the total the
        optimizer step normalizes by is a sum of them.
        """
        if len(retained_lengths) != len(dropped):
            raise ValueError(
                "caption loss weights need one dropout flag per retained length"
            )
        import torch

        sample_count = len(retained_lengths)
        if not self.enabled:
            return MicroBatchWeights(None, None, float(sample_count))
        values = [
            1.0 if was_dropped else self.weight_for(length)
            for length, was_dropped in zip(retained_lengths, dropped)
        ]
        total = sum(values)
        if total <= 0.0:
            raise ValueError(
                "the caption loss weights of a micro-batch sum to zero: a curve "
                "that zeroes every sample the micro-batch holds leaves the step "
                "no loss to normalize"
            )
        uniform = all(value == values[0] for value in values)
        weights = None if uniform else torch.tensor(
            values, device=device, dtype=torch.float32
        )
        return MicroBatchWeights(values, weights, float(total))


def weighted_mean(loss_weight_sum, weight_sum):
    """``sum(w_i * loss_i) / sum(w_i)``, the objective one optimizer step takes.

    The two arguments are the step's two sums after they were summed across
    ranks, and they are divided as they arrive: the loop hands over the reduced
    tensors, so a run without weights reports the number it reported before the
    weights existed, and a test can hand over plain floats.

    Dividing by the summed weights rather than by the sample count is what
    keeps a reweighting a reweighting.  With the weight sum in the denominator
    a step's objective is a weighted average of its samples' losses: samples
    move share between each other and the size of the step stands still.  A
    denominator of "number of samples" would instead scale the step with the
    weights themselves — a change of learning rate in disguise, which is
    exactly what would make a reweighted run incomparable with its unweighted
    baseline at equal step counts.
    """
    if weight_sum <= 0.0:
        raise ValueError(
            "the total loss weight of an optimizer step must be positive"
        )
    return loss_weight_sum / weight_sum


class StepLossAccumulator:
    """The weighted mean an optimizer step is built from, gathered across the
    micro-batches that step was split into.

    Both sums are kept — the weighted losses and the weights — and divided
    exactly once, at the optimizer boundary.  Averaging the micro-batches' own
    (already normalized) means instead would weight each micro-batch by
    ``1 / micro_batches`` rather than by the weight of the samples it holds,
    and micro-batches differ in both their sample count and their
    caption-length mix, so the step's objective would depend on how it happened
    to be split.

    With weighting off every weight is 1.0, the two sums are the loss sum
    and the sample count the loop kept before this existed, and the weighted
    mean is the arithmetic mean over the step's samples.
    """

    def __init__(self, device: Optional[torch.device] = None):
        import torch

        # Held on the compute device: adding a device scalar to it stays on the
        # device, so accumulating a micro-batch never synchronizes.
        self._loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.sample_count = 0
        self.weight_sum = 0.0

    def add(
        self, loss: torch.Tensor, weight_sum: float, sample_count: int
    ) -> torch.Tensor:
        """Record one micro-batch; return the scalar to back-propagate.

        ``loss`` is the micro-batch's own loss (its weighted mean when the
        micro-batch mixes lengths), ``weight_sum`` the sum of its samples'
        weights, and ``sample_count`` how many samples it holds.
        """
        weight = float(weight_sum)
        self.sample_count += int(sample_count)
        self.weight_sum += weight
        self._loss_sum += loss.detach().float() * weight
        return loss * weight

    def loss_sum(self) -> torch.Tensor:
        """The numerator, in the form that is summed across ranks."""
        return self._loss_sum

    def reset(self) -> None:
        """Start a new optimizer step."""
        self._loss_sum.zero_()
        self.sample_count = 0
        self.weight_sum = 0.0

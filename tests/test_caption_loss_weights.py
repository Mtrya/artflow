"""Caption-weighted gradients checked against analytic values and batch partition invariance."""

import pytest
import torch

from src.flow.paths import FlowMatchingOT
from src.pretrain.caption_loss_weights import (
    CaptionLossWeights,
    StepLossAccumulator,
    weighted_mean,
)

REFERENCE = 128


def _flow_matching_batch(targets, theta):
    """A batch whose sample ``i`` has loss ``(theta - targets[i]) ** 2``.

    The optimal-transport target is ``z1 - z0``, so one element per sample with
    ``z0`` at zero gives every sample a loss the test can state in closed form
    while the model output stays a scalar the gradient can be read off.
    """
    z1 = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1, 1, 1)
    model_output = theta.reshape(1, 1, 1, 1).expand_as(z1)
    return model_output, torch.zeros_like(z1), z1, torch.zeros(z1.shape[0])


def _step(micro_batches, theta, weights):
    """Run one optimizer step of the training loop over ``micro_batches``.

    Each micro-batch is a ``(targets, lengths)`` pair.  The loop multiplies a
    micro-batch's loss by the sum of its samples' weights for the backward
    pass, and divides the accumulated gradient by the step's total weight once,
    at the boundary; one rank, so there is no cross-rank average.  Returns the
    weighted mean the loop would report and leaves the gradient on ``theta``.
    """
    accumulator = StepLossAccumulator()
    algorithm = FlowMatchingOT()
    for targets, lengths in micro_batches:
        micro = weights.for_micro_batch(lengths, [False] * len(lengths))
        model_output, z0, z1, t = _flow_matching_batch(targets, theta)
        loss = algorithm.compute_loss(
            model_output, z0, z1, t, sample_weights=micro.tensor)
        accumulator.add(loss, micro.total, len(lengths)).backward()
    theta.grad.div_(accumulator.weight_sum)
    return weighted_mean(accumulator.loss_sum().item(), accumulator.weight_sum)


class TestWeightedReduction:

    def test_the_gradient_is_that_of_the_weighted_mean(self):
        # Losses (theta - 1)^2 and (theta - 3)^2 under weights 1.0 and 3.0:
        # the weighted mean at theta = 0 is 7.0 with gradient -5.0, not the
        # weighted sum's 28.0 with gradient -20.0.
        theta = torch.tensor(0.0, requires_grad=True)
        model_output, z0, z1, t = _flow_matching_batch([1.0, 3.0], theta)
        loss = FlowMatchingOT().compute_loss(
            model_output, z0, z1, t, sample_weights=torch.tensor([1.0, 3.0]))
        loss.backward()
        assert loss.item() == pytest.approx(7.0)
        assert float(theta.grad) == pytest.approx(-5.0)


class TestStepEquivalence:
    def test_a_step_split_into_micro_batches_lands_where_an_unsplit_one_does(self):
        weights = CaptionLossWeights(curve="log2", reference=REFERENCE)
        targets = [1.0, 3.0, 9.0, 2.0]
        lengths = [128, 512, 512, 512]
        # Weights 1.0 / 2.0 / 2.0 / 2.0; the per-sample losses at theta = 0 are
        # the targets squared, so the step's weighted mean is
        # (1*1 + 2*9 + 2*81 + 2*4) / 7 = 189/7 and its gradient at theta = 0
        # is -2 * (1*1 + 2*3 + 2*9 + 2*2) / 7 = -58/7.
        expected_loss = (1 * 1 + 2 * 9 + 2 * 81 + 2 * 4) / 7
        expected_grad = -2 * (1 * 1 + 2 * 3 + 2 * 9 + 2 * 2) / 7

        whole = torch.zeros((), requires_grad=True)
        whole_loss = _step([(targets, lengths)], whole, weights)

        split = torch.zeros((), requires_grad=True)
        split_loss = _step(
            [(targets[:2], lengths[:2]), (targets[2:], lengths[2:])],
            split, weights)

        assert whole_loss == pytest.approx(expected_loss, rel=1e-6)
        assert split_loss == pytest.approx(expected_loss, rel=1e-6)
        assert float(whole.grad) == pytest.approx(expected_grad, rel=1e-6)
        assert float(split.grad) == pytest.approx(expected_grad, rel=1e-6)


    def test_a_uniform_multiplier_leaves_the_step_alone(self):
        # A step whose samples all carry the same weight keeps the plain mean
        # and the plain gradient: the weights move share between samples, they
        # do not scale the step.
        weights = CaptionLossWeights(curve="log2", reference=REFERENCE)
        targets = [-1.0, 3.0]

        plain = torch.zeros((), requires_grad=True)
        plain_loss = _step([(targets, [128, 128])], plain, CaptionLossWeights())

        doubled = torch.zeros((), requires_grad=True)
        doubled_loss = _step([(targets, [512, 512])], doubled, weights)

        assert doubled_loss == pytest.approx(plain_loss)
        assert float(doubled.grad) == pytest.approx(float(plain.grad))

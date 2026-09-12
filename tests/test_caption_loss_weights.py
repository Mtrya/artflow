"""Caption-length loss weights and the step objective they normalize.

The weighting moves the share of a step's gradient between caption lengths
without resizing the step, so these tests pin both halves: the curve's rules
(what weight a retained length receives, what a dropped sample receives, what
an unknown curve or reference does) and the arithmetic that turns per-sample
weights into one optimizer step.  That step accumulates the weighted losses and
the weights and divides once at its boundary, so a step split across
micro-batches has to land where an unsplit one does; that equivalence is
asserted against the gradient and not only against the reported loss.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.flow.paths import FlowMatchingOT
from src.train.caption_loss_weights import (
    CaptionLossWeights,
    StepLossAccumulator,
    weighted_mean,
)
from src.train.caption_telemetry import CaptionTelemetry

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


class TestCurve:
    def test_the_default_curve_weights_nothing(self):
        weights = CaptionLossWeights()
        assert weights.curve == "none"
        assert weights.reference == REFERENCE
        assert not weights.enabled
        assert [weights.weight_for(length)
                for length in (0, 1, 128, 900, 2048)] == [1.0] * 5

    def test_the_none_curve_micro_batch_carries_no_weights(self):
        micro = CaptionLossWeights(curve="none").for_micro_batch(
            [100, 900], [False, False])
        assert micro.values is None
        assert micro.tensor is None
        assert micro.total == 2.0

    def test_the_log2_curve_lifts_each_doubling_by_one(self):
        weights = CaptionLossWeights(curve="log2", reference=REFERENCE)
        assert weights.enabled
        # log2(L / 128): 256 sits at exactly 1.0, the floor keeps everything
        # shorter there, and each further doubling adds one.
        assert weights.weight_for(0) == 1.0
        assert weights.weight_for(127) == 1.0
        assert weights.weight_for(128) == 1.0
        assert weights.weight_for(256) == 1.0
        assert weights.weight_for(384) == pytest.approx(math.log2(3))
        assert weights.weight_for(512) == pytest.approx(2.0)
        assert weights.weight_for(1024) == pytest.approx(3.0)
        assert weights.weight_for(2048) == pytest.approx(4.0)

    def test_the_curve_never_weights_a_sample_below_one(self):
        weights = CaptionLossWeights(curve="log2", reference=REFERENCE)
        assert min(weights.weight_for(length)
                   for length in range(0, 4097)) == 1.0

    def test_the_reference_anchors_the_curve(self):
        weights = CaptionLossWeights(curve="log2", reference=64)
        assert weights.weight_for(128) == pytest.approx(1.0)
        assert weights.weight_for(256) == pytest.approx(2.0)
        assert weights.weight_for(512) == pytest.approx(3.0)
        assert weights.weight_for(1024) == pytest.approx(4.0)
        assert weights.weight_for(2048) == pytest.approx(5.0)

    def test_an_unknown_curve_is_rejected(self):
        with pytest.raises(ValueError, match="must be one of"):
            CaptionLossWeights(curve="linear")

    def test_a_reference_below_two_is_rejected(self):
        for reference in (1, 0, -1):
            with pytest.raises(ValueError, match="reference must be >= 2"):
                CaptionLossWeights(curve="log2", reference=reference)


class TestMicroBatchWeights:
    def test_each_sample_is_weighted_by_its_own_length(self):
        weights = CaptionLossWeights(curve="log2", reference=REFERENCE)
        micro = weights.for_micro_batch([128, 512, 2048], [False] * 3)
        assert micro.values == pytest.approx([1.0, 2.0, 4.0])
        assert micro.total == pytest.approx(7.0)
        assert micro.tensor.tolist() == pytest.approx([1.0, 2.0, 4.0])

    def test_a_dropped_caption_keeps_weight_one(self):
        # The curve is about how much a caption's length should matter, and a
        # sample the model was not conditioned on says nothing about length.
        weights = CaptionLossWeights(curve="log2", reference=REFERENCE)
        micro = weights.for_micro_batch([2048, 2048, 128], [True, False, True])
        assert micro.values == pytest.approx([1.0, 4.0, 1.0])
        assert micro.total == pytest.approx(6.0)

    def test_a_uniform_micro_batch_needs_no_per_sample_reduction(self):
        # Every sample carries the same multiplier, so the weighted mean of the
        # micro-batch is a plain rescale of its mean.
        weights = CaptionLossWeights(curve="log2", reference=REFERENCE)
        micro = weights.for_micro_batch([512, 512], [False, False])
        assert micro.values == pytest.approx([2.0, 2.0])
        assert micro.tensor is None
        assert micro.total == pytest.approx(4.0)

    def test_an_all_dropped_micro_batch_is_uniformly_unweighted(self):
        weights = CaptionLossWeights(curve="log2", reference=REFERENCE)
        micro = weights.for_micro_batch([2048, 128], [True, True])
        assert micro.values == pytest.approx([1.0, 1.0])
        assert micro.tensor is None
        assert micro.total == pytest.approx(2.0)

    def test_the_tensor_is_float32_on_the_named_device(self):
        weights = CaptionLossWeights(curve="log2", reference=REFERENCE)
        micro = weights.for_micro_batch(
            [128, 2048], [False, False], device=torch.device("cpu"))
        assert micro.tensor is not None
        assert micro.tensor.dtype == torch.float32
        assert micro.tensor.device == torch.device("cpu")

    def test_mismatched_lengths_and_dropout_are_rejected(self):
        with pytest.raises(ValueError, match="dropout flag"):
            CaptionLossWeights(curve="log2").for_micro_batch([100, 200], [False])


class TestWeightedMean:
    def test_the_loss_sum_is_divided_by_the_weight_sum(self):
        # Losses 1.0 and 9.0 with weights 1.0 and 3.0: (1*1 + 3*9) / 4 = 7.0,
        # not the weighted sum 28.0 and not the plain mean 5.0.
        assert weighted_mean(28.0, 4.0) == pytest.approx(7.0)

    def test_a_fractional_weight_sum_is_just_as_valid(self):
        assert weighted_mean(11.5, 4.5) == pytest.approx(11.5 / 4.5)

    def test_a_non_positive_weight_sum_is_rejected(self):
        for weight_sum in (0.0, -1.0):
            with pytest.raises(ValueError, match="must be positive"):
                weighted_mean(1.0, weight_sum)

    def test_the_sums_may_be_the_reduced_tensors(self):
        # The loop hands over the tensors its cross-rank reduction produced, so
        # the reported loss is divided in the dtype that reduction ran in.
        value = weighted_mean(torch.tensor(6.0), torch.tensor(3.0))
        assert isinstance(value, torch.Tensor)
        assert float(value) == 2.0
        with pytest.raises(ValueError, match="must be positive"):
            weighted_mean(torch.tensor(1.0), torch.tensor(0.0))


class TestWeightedReduction:
    def test_uniform_weights_reduce_where_the_plain_mean_does(self):
        theta = torch.tensor(0.0)
        model_output, z0, z1, t = _flow_matching_batch([1.0, 3.0], theta)
        algorithm = FlowMatchingOT()
        assert torch.equal(
            algorithm.compute_loss(
                model_output, z0, z1, t, sample_weights=torch.ones(2)),
            algorithm.compute_loss(model_output, z0, z1, t),
        )

    def test_the_gradient_is_that_of_the_weighted_mean(self):
        # Losses (theta - 1)^2 and (theta - 3)^2 under weights 1.0 and 3.0:
        # the weighted mean at theta = 0 is 7.0 with gradient -5.0, not the
        # weighted sum's 28.0 with gradient -20.0.
        theta = torch.tensor(0.0, requires_grad=True)
        model_output, z0, z1, t = _flow_matching_batch([1.0, 3.0], theta)
        loss = FlowMatchingOT().compute_loss(
            model_output, z0, z1, t, sample_weights=torch.tensor([1.0, 3.0]))
        loss.backward()
        assert float(loss) == pytest.approx(7.0)
        assert float(theta.grad) == pytest.approx(-5.0)

    def test_weights_of_the_wrong_length_are_rejected(self):
        theta = torch.tensor(0.0)
        model_output, z0, z1, t = _flow_matching_batch([1.0, 3.0], theta)
        with pytest.raises(ValueError, match="one multiplier per sample"):
            FlowMatchingOT().compute_loss(
                model_output, z0, z1, t, sample_weights=torch.ones(3))


class TestStepAccumulator:
    def test_micro_batch_means_are_not_averaged(self):
        # Three samples at loss 1.0 and one at loss 9.0, all weights 1.0: the
        # step is (3*1 + 1*9) / 4 = 3.0.  Averaging the two micro-batches' own
        # means would give 5.0.
        accumulator = StepLossAccumulator()
        accumulator.add(torch.tensor(1.0), 3.0, 3)
        accumulator.add(torch.tensor(9.0), 1.0, 1)
        assert accumulator.sample_count == 4
        assert accumulator.weight_sum == pytest.approx(4.0)
        assert weighted_mean(float(accumulator.loss_sum()),
                             accumulator.weight_sum) == pytest.approx(3.0)

    def test_add_returns_the_scalar_to_back_propagate(self):
        accumulator = StepLossAccumulator()
        loss = torch.tensor(4.0, requires_grad=True)
        backward_loss = accumulator.add(loss, 2.5, 2)
        assert torch.equal(backward_loss, loss * 2.5)
        backward_loss.backward()
        assert float(loss.grad) == pytest.approx(2.5)
        # The numerator the optimizer boundary divides is kept, detached.
        assert float(accumulator.loss_sum()) == pytest.approx(10.0)

    def test_reset_starts_a_new_step(self):
        accumulator = StepLossAccumulator()
        accumulator.add(torch.tensor(2.0), 2.0, 2)
        accumulator.reset()
        assert accumulator.sample_count == 0
        assert accumulator.weight_sum == 0.0
        assert float(accumulator.loss_sum()) == 0.0


class TestDefaultIsOff:
    def test_the_default_curve_never_reaches_the_loss(self):
        # Not "equivalent to" the unweighted loss: the weights do not exist, so
        # the algorithm reduces exactly as it did before this feature.
        theta = torch.tensor(0.0)
        model_output, z0, z1, t = _flow_matching_batch([1.0, 3.0], theta)
        micro = CaptionLossWeights().for_micro_batch([100, 900], [False, False])
        algorithm = FlowMatchingOT()
        loss = algorithm.compute_loss(
            model_output, z0, z1, t, sample_weights=micro.tensor)
        assert loss == F.mse_loss(model_output, z1 - z0)

    def test_the_default_curve_keeps_the_accumulation_unweighted(self):
        accumulator = StepLossAccumulator()
        micro = CaptionLossWeights().for_micro_batch([100, 900], [False, False])
        loss = torch.tensor(4.0)
        backward_loss = accumulator.add(loss, micro.total, 2)
        assert torch.equal(backward_loss, loss * 2)
        assert accumulator.sample_count == 2
        assert accumulator.weight_sum == 2.0
        assert torch.equal(accumulator.loss_sum(), loss.detach().float() * 2)

    def test_a_step_with_the_default_curve_is_the_plain_mean(self):
        theta = torch.zeros((), requires_grad=True)
        loss = _step([([-1.0, 3.0], [100, 900])], theta, CaptionLossWeights())
        assert loss == pytest.approx(5.0)
        assert float(theta.grad) == pytest.approx(-2.0)


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

    def test_the_two_reduction_paths_are_both_exercised(self):
        # The split above holds a mixed-weight micro-batch and a uniform one;
        # the equivalence is only interesting if both paths were taken.
        weights = CaptionLossWeights(curve="log2", reference=REFERENCE)
        mixed = weights.for_micro_batch([128, 512], [False, False])
        uniform = weights.for_micro_batch([512, 512], [False, False])
        assert mixed.tensor is not None
        assert uniform.tensor is None

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


class TestGradientsFollowTheWeights:
    def test_reweighting_moves_the_gradient_toward_longer_captions(self):
        # Short sample at -1 pulls the parameter down, long sample at +3 pulls
        # it up.  At theta = 0 the plain gradient is -2 * their mean = -2.0;
        # weighting the long caption four times moves it toward the long one.
        weights = CaptionLossWeights(curve="log2", reference=REFERENCE)
        targets = [-1.0, 3.0]
        lengths = [64, 2048]  # weights 1.0 and 4.0

        plain = torch.zeros((), requires_grad=True)
        plain_loss = _step([(targets, lengths)], plain, CaptionLossWeights())
        weighted = torch.zeros((), requires_grad=True)
        weighted_loss = _step([(targets, lengths)], weighted, weights)

        assert plain_loss == pytest.approx(5.0)
        assert float(plain.grad) == pytest.approx(-2.0)
        # (1*(-1) + 4*3) / 5 = 11/5, so the gradient is -2 * 11/5.
        assert weighted_loss == pytest.approx((1 * 1 + 4 * 9) / 5)
        assert float(weighted.grad) == pytest.approx(-2 * 11 / 5)
        assert float(weighted.grad) < float(plain.grad)


class TestLoggedWeightMoments:
    def test_the_moments_cover_the_conditioned_samples(self):
        telemetry = CaptionTelemetry()
        telemetry.record([128, 2048, 2048], [False, False, True], 2048,
                         loss_weights=[1.0, 4.0, 1.0])
        metrics = telemetry.snapshot()
        # The dropped sample's weight is in neither denominator: the moments
        # describe where the conditioned samples' gradient went.
        assert metrics["caption/weight_mean"] == pytest.approx(2.5)
        assert metrics["caption/grad_weighted_mean_tokens"] == pytest.approx(
            (1.0 * 128 + 4.0 * 2048) / 5.0)

    def test_a_window_that_recorded_no_weights_reports_no_moment(self):
        telemetry = CaptionTelemetry()
        telemetry.record([128, 2048], [False, False], 2048)
        metrics = telemetry.snapshot()
        assert "caption/weight_mean" not in metrics
        assert "caption/grad_weighted_mean_tokens" not in metrics

    def test_a_fully_dropped_window_reports_no_moment(self):
        telemetry = CaptionTelemetry()
        telemetry.record([2048], [True], 2048, loss_weights=[1.0])
        metrics = telemetry.snapshot()
        assert "caption/weight_mean" not in metrics

    def test_weight_sums_merge_across_ranks(self):
        rank_a = CaptionTelemetry()
        rank_b = CaptionTelemetry()
        rank_a.record([128] * 2, [False] * 2, 256, loss_weights=[1.0, 1.0])
        rank_b.record([2048] * 3, [False] * 3, 2048, loss_weights=[4.0] * 3)

        rank_a.merge_window_counts(rank_b.window_counts())
        metrics = rank_a.snapshot()

        assert metrics["caption/weight_mean"] == pytest.approx(14.0 / 5)
        assert metrics["caption/grad_weighted_mean_tokens"] == pytest.approx(
            (2 * 1.0 * 128 + 3 * 4.0 * 2048) / 14.0)

    def test_weights_of_the_wrong_length_are_rejected(self):
        telemetry = CaptionTelemetry()
        with pytest.raises(ValueError, match="one entry per retained length"):
            telemetry.record([100, 900], [False, False], 1024,
                             loss_weights=[1.0])

import torch

from scripts.bench.comm_update_probe import sequential_mean, summarize, tensor_error


def test_shared_captures_not_mutated_and_cancellation_is_visible():
    values = [torch.tensor([1.0001, 0.12345]), torch.tensor([-1.0, 0.4321])]
    originals = [v.clone() for v in values]
    reference = sequential_mean(values, torch.float32, "cpu")
    compressed = sequential_mean(values, torch.bfloat16, "cpu")
    assert torch.equal(reference, (values[0] + values[1]) / 2)
    assert all(torch.equal(a, b) for a, b in zip(values, originals))
    assert compressed[0] == 0 and reference[0] != 0
    metrics = summarize([tensor_error(compressed, reference)])
    assert metrics["finite"] and not metrics["exact"]
    assert metrics["relative_l2"] > 0


def test_group_error_uses_norm_of_updates_not_mean_of_parameter_ratios():
    rows = [tensor_error(torch.tensor([2.0]), torch.tensor([1.0])),
            tensor_error(torch.tensor([100.0]), torch.tensor([100.0]))]
    assert abs(summarize(rows)["relative_l2"] - 1 / 10001**.5) < 1e-12
    assert summarize([tensor_error(torch.zeros(2), torch.zeros(2))])["relative_l2"] == 0

import torch

from scripts.bench.ddp_comm_probe import apply_hooks, error_metrics


def test_same_fp32_boundaries_and_all_futures_waited():
    tensor = torch.zeros(23, dtype=torch.float32)
    sizes, waited = [], []

    class Pending:
        def __init__(self, chunk):
            self.chunk = chunk

        def wait(self):
            waited.append(self.chunk.numel())
            self.chunk.fill_(1)

    def hook(group, bucket):
        assert group is None
        chunk = bucket.buffer()
        assert chunk.dtype == torch.float32
        sizes.append(chunk.numel())
        return Pending(chunk)

    apply_hooks(tensor, hook, 8)
    assert sizes == waited == [8, 8, 7]
    assert torch.equal(tensor, torch.ones_like(tensor))


def test_roundtrip_compression_is_not_reported_as_exact_gradient_preservation():
    reference = torch.tensor([1.001, -1.001, 0.123456], dtype=torch.float32)
    metrics = error_metrics(reference.bfloat16().float(), reference)
    assert metrics["finite"]
    assert not metrics["exact"]
    assert metrics["max_abs"] > 0
    assert metrics["relative_l2"] > 0
    assert error_metrics(reference, reference)["relative_l2"] == 0

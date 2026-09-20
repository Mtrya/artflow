import pytest

from scripts.bench.plan_buckets import mean_emitted_batch
from scripts.bench.resize_bucket_plan import resize


def test_preserves_target_with_integer_sizes_and_nonuniform_draws():
    sizes, shares = [4, 8, 12], [.1, .7, .2]
    target = mean_emitted_batch(shares, sizes) * 8 * 5
    result = resize(sizes, shares, old_accumulation=5, new_accumulation=2,
                    ranks=8, target=target)
    assert mean_emitted_batch(shares, result) * 16 >= target * (1-1e-12)
    assert all(1 <= n <= (b*5+1)//2 for b,n in zip(sizes,result))
    assert sizes == [4,8,12]
    for i in range(len(result)):
        trial = result.copy()
        trial[i] -= 1
        assert mean_emitted_batch(shares, trial) * 16 < target


def test_exact_integer_scale():
    assert resize([4,8], [.5,.5], old_accumulation=4,new_accumulation=2,
                  ranks=8,target=mean_emitted_batch([.5,.5],[4,8])*32) == [8,16]


@pytest.mark.parametrize("shares", [[float("nan")],[-1],[0],[]])
def test_rejects_invalid_shares(shares):
    with pytest.raises(ValueError):
        resize([4],shares,old_accumulation=5,new_accumulation=2,ranks=8,target=1)


def test_refuses_unjustified_growth():
    with pytest.raises(ValueError,match="cannot meet target"):
        resize([4],[1],old_accumulation=5,new_accumulation=2,ranks=8,target=1000)

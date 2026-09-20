import copy
import json

import pytest

from scripts.bench.compare_infra import compare, read_records


def test_compare_requires_matching_work_and_excludes_profiled_updates(tmp_path):
    path = tmp_path / "rank-0.jsonl"
    rows = [dict(step=i, rank=0, global_samples=8, progress=.375, loss=1., seconds=2.,
                 shapes=[dict(shape=[8, 16, 32, 32, 128], count=1)], profiled=i == 3)
            for i in range(1, 4)]
    path.write_text("\n".join(map(json.dumps, rows)))
    baseline = read_records(path)
    candidate = copy.deepcopy(baseline)
    for row in candidate.values():
        row["seconds"] = 1.
    result = compare(baseline, candidate)
    assert result["matched_nonprofiled"]["speedup"] == 2
    assert result["matched_nonprofiled"]["updates"] == 2
    assert result["matched_repeat_shapes"]["updates"] == 1
    candidate[1]["global_samples"] = 9
    with pytest.raises(ValueError, match="unmatched global_samples"):
        compare(baseline, candidate)

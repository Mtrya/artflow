from scripts.ascend.hero_watch import fetch, window


def test_cloud_scalar_formats_preserve_step_timestamp_and_zero():
    class Run:
        def metrics(self, **kwargs):
            assert kwargs == {"keys": ["train/loss"], "range_query": {"tail": 2}}
            return {"list": [{"key": "train/loss", "metrics": [
                {"step": 10, "value": 0.0, "timestamp": 1000},
                {"index": 11, "data": 0.5, "timestamp": 2000},
            ]}]}
    points = fetch(Run(), ["train/loss"], range_query={"tail": 2})["train/loss"]
    assert points == [(10, 0.0, 1000), (11, 0.5, 2000)]
    stats = window(points)
    assert stats["max_step"] == 11 and stats["median"] == 0.25

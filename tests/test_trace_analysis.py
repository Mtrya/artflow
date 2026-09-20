from scripts.bench.analyze_trace import merged_intervals, overlap_duration, summary


def test_busy_union_merges_concurrent_stream_work():
    assert merged_intervals([(0, 10), (5, 15), (20, 25), (25, 30)]) == [[0, 15], [20, 30]]


def test_communication_overlap_uses_unions_and_the_same_gpu():
    assert overlap_duration([(5, 15), (10, 20)], [(0, 10), (12, 14)]) == 7
    events = [dict(ph="X", cat=cat, name=name, ts=start * 1000, dur=duration * 1000, pid=device)
              for cat, name, start, duration, device in [
                  ("kernel", "ncclAllReduce", 5, 10, 0),
                  ("kernel", "ncclAllReduce", 10, 10, 0),
                  ("kernel", "gemm", 0, 10, 0),
                  ("gpu_memcpy", "Memcpy", 12, 2, 0),
                  ("kernel", "gemm", 0, 100, 1),
              ]]
    windows = summary(dict(traceEvents=events))["gpu_windows"]
    assert windows["0"]["communication_union_ms"] == 15
    assert windows["0"]["communication_overlapped_other_gpu_ms"] == 7
    assert windows["0"]["communication_unoverlapped_ms"] == 8
    assert windows["1"]["communication_union_ms"] == 0


def test_compiled_annotations_and_cpu_waits_are_not_gpu_work():
    def event(cat, name, start, duration, args=None):
        return dict(ph="X", cat=cat, name=name, ts=start, dur=duration, pid=0, args=args or {})
    result = summary(dict(traceEvents=[
        event("kernel", "gemm", 0, 10, {"External id": 4}),
        event("kernel", "triton_poi", 5, 10),
        event("gpu_memcpy", "Memcpy HtoD", 20, 10),
        event("gpu_user_annotation", "CompiledFxGraph", 0, 30),
        event("cpu_op", "aten::mm", 0, 4, {"External id": 4, "Input Dims": [[2, 3], [3, 4]]}),
        event("cuda_runtime", "cudaDeviceSynchronize", 0, 30),
    ]))
    window = result["gpu_windows"]["0"]
    assert window["gpu_duration_sum_ms"] == .03
    assert window["busy_union_ms"] == .025
    assert window["internal_idle_ms"] == .005
    assert result["top_launch_operators_and_input_dims"][0]["summed_ms"] == .01
    assert result["top_launch_operators_and_input_dims"][0]["gpu_events"] == 1
    assert result["cpu_synchronization_calls"][0]["calls"] == 1
    assert all("Compiled" not in row["name"] for row in result["top_gpu_kernels"])

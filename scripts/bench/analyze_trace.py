"""Summarize real Kineto GPU events without double-counting compiled annotations.

GPU duration sums are work, not critical-path latency: concurrent streams can
overlap. Busy unions/internal gaps are reported separately per GPU process.
CPU synchronization durations can overlap GPU work and are not additive savings.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def merged_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def overlap_duration(first, second):
    """Intersection of interval unions, not the sum of pairwise intersections."""
    a, b = merged_intervals(first), merged_intervals(second)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def kernel_class(name):
    name = name.lower()
    if "nccl" in name:
        return "communication"
    if any(word in name for word in ("fmha", "flash", "attention")):
        return "attention"
    if any(word in name for word in ("gemm", "cutlass::kernel", "matmul", "bmm")):
        return "gemm"
    if any(word in name for word in ("triton_poi", "elementwise", "vectorized")):
        return "pointwise"
    if any(word in name for word in ("reduce", "norm", "softmax")):
        return "reduction"
    return "other"


def summary(payload, top=20):
    events = [event for event in payload["traceEvents"]
              if event.get("ph") == "X" and event.get("dur", 0) > 0]
    cpu = {event.get("args", {}).get("External id"): event for event in events
           if event.get("cat") == "cpu_op" and "External id" in event.get("args", {})}
    devices = defaultdict(list)
    communication, other_gpu = defaultdict(list), defaultdict(list)
    kernels, classes, operators, syncs = (defaultdict(lambda: [0, 0.0]) for _ in range(4))
    for event in events:
        category, name, duration = event.get("cat", ""), event.get("name", ""), event["dur"]
        if category in ("kernel", "gpu_memcpy", "gpu_memset"):
            device = str(event.get("pid"))
            interval = (event["ts"], event["ts"] + duration)
            devices[device].append(interval)
            target = communication if category == "kernel" and "nccl" in name.lower() else other_gpu
            target[device].append(interval)
            for table, key in ((kernels, name),
                               (classes, kernel_class(name) if category == "kernel" else category)):
                table[key][0] += 1
                table[key][1] += duration
            owner = cpu.get(event.get("args", {}).get("External id"))
            if owner:
                key = json.dumps([owner["name"], owner.get("args", {}).get("Input Dims"),
                                  owner.get("args", {}).get("Input Strides")])
                operators[key][0] += 1
                operators[key][1] += duration
        elif category in ("cuda_runtime", "cuda_driver") and "Synchronize" in name:
            syncs[name][0] += 1
            syncs[name][1] += duration

    def ranked(table, count_name="gpu_events"):
        return [dict(name=name, **{count_name: values[0]}, summed_ms=round(values[1] / 1000, 3))
                for name, values in sorted(table.items(), key=lambda item: item[1][1], reverse=True)[:top]]

    windows = {}
    for device, intervals in devices.items():
        merged = merged_intervals(intervals)
        span = merged[-1][1] - merged[0][0]
        busy = sum(end - start for start, end in merged)
        comm = sum(end - start for start, end in merged_intervals(communication[device]))
        overlap = overlap_duration(communication[device], other_gpu[device])
        gaps = [(left[1], right[0] - left[1]) for left, right in zip(merged, merged[1:])]
        windows[device] = dict(
            first_activity_us=merged[0][0], last_activity_us=merged[-1][1],
            activity_span_ms=round(span / 1000, 3), busy_union_ms=round(busy / 1000, 3),
            internal_idle_ms=round((span - busy) / 1000, 3), busy_fraction=busy / span,
            gpu_duration_sum_ms=round(sum(end - start for start, end in intervals) / 1000, 3),
            communication_union_ms=round(comm / 1000, 3),
            communication_overlapped_other_gpu_ms=round(overlap / 1000, 3),
            communication_unoverlapped_ms=round((comm - overlap) / 1000, 3),
            largest_internal_gaps=[dict(start_us=start, duration_ms=round(duration / 1000, 3))
                                   for start, duration in sorted(gaps, key=lambda item: item[1],
                                                                 reverse=True)[:top]])
    return dict(gpu_windows=windows, kernel_classes=ranked(classes),
                top_gpu_kernels=ranked(kernels), top_launch_operators_and_input_dims=ranked(operators),
                cpu_synchronization_calls=ranked(syncs, "calls"),
                caveats=["Only actual GPU kernel/memcpy/memset events are summed; compiled annotations excluded.",
                         "Activity span excludes leading/trailing CPU-only time and is not full update latency.",
                         "CPU waits overlap GPU work; they cannot be added to GPU time or called removable overhead.",
                         "NCCL overlap is temporal overlap with other GPU work (including copies), not proof communication is cost-free. Unoverlapped time is not automatically removable.",
                         "Operator gpu_events count kernel/copy launches, not CPU operator invocations; keys include dims and strides.",
                         "Kernel classes are name-based hints. Inspect the trace before attributing a bottleneck.",
                         "Profiler overhead and sampled shapes prevent using this as a production throughput gate."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    result = summary(json.loads(args.trace.read_text()), args.top)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

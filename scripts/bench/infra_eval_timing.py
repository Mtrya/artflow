"""Run the unchanged trainer with bounded evaluation timing instrumentation.

Used by infra_stage_chain --with-evaluation, not by the production launcher.
Evaluate after real updates so optimizer state and DDP buffers remain resident.
Per-phase barriers include rank waiting in the allocated wall time. These runs
are correctness/overhead probes, never steady-state throughput measurements.
"""

import functools
import json
import os
from pathlib import Path
import time


def main():
    import torch
    from src.train import train

    output = Path(os.environ["ARTFLOW_EVAL_TIMING_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    path = output / f"eval-rank-{rank}.jsonl"
    counts = {}

    def boundary():
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.synchronize()

    def timed(name, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            boundary()
            counts[name] = counts.get(name, 0) + 1
            torch.cuda.reset_peak_memory_stats()
            event = dict(phase=name, occurrence=counts[name], rank=rank,
                         start_allocated_bytes=torch.cuda.memory_allocated(),
                         start_reserved_bytes=torch.cuda.memory_reserved(),
                         complete=False)
            start = time.monotonic()
            try:
                result = function(*args, **kwargs)
                boundary()
                event["complete"] = True
                if name == "loss":
                    event["metrics"] = {k: float(v) for k, v in result.items()}
                if "current_step" in kwargs:
                    event["step"] = kwargs["current_step"]
                return result
            finally:
                event.update(seconds=time.monotonic() - start,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             peak_reserved_bytes=torch.cuda.max_memory_reserved())
                with path.open("a") as handle:
                    handle.write(json.dumps(event) + "\n")
        return wrapped

    train.EvalLossProbe.__init__ = timed("loss_setup", train.EvalLossProbe.__init__)
    train.EvalLossProbe.evaluate = timed("loss", train.EvalLossProbe.evaluate)
    train.run_prompt_grid_eval = timed("grid", train.run_prompt_grid_eval)
    train.run_kid_eval = timed("kid", train.run_kid_eval)
    train.main()


if __name__ == "__main__":
    main()

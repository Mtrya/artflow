#!/usr/bin/env python
"""Print a compact summary of one swanlab run from its latest logged values.

Run on Andromeda (holds the swanlab netrc login). Usage:
  python swanlab_summary.py --project artflow-stage2 --run s2-mod-none [--keys a b c]
Default keys: eval/loss, eval/loss_t015/t040/t065/t090, kid/mean, kid/std,
train/samples_per_sec, train/step_time_s, train/mem_peak_gb, train/grad_norm.
Prints "key: value@step". Missing keys print "key: (absent)".
"""

import argparse

from swanlab import Api


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--run", required=True)
    p.add_argument(
        "--keys",
        nargs="*",
        default=[
            "eval/loss",
            "eval/loss_t015",
            "eval/loss_t040",
            "eval/loss_t065",
            "eval/loss_t090",
            "kid/mean",
            "kid/std",
            "train/samples_per_sec",
            "train/step_time_s",
            "train/mem_peak_gb",
            "train/mem_alloc_gb",
            "train/txt_seq_len",
            "train/grad_norm",
        ],
    )
    args = p.parse_args()

    summary = Api().run(f"{args.project}/{args.run}").summary()
    for k in args.keys:
        if k in summary:
            v = summary[k]
            if isinstance(v, dict):
                print(f"{k}: {v.get('value', v)} @step {v.get('step', '?')}")
            else:
                print(f"{k}: {v}")
        else:
            print(f"{k}: (absent)")


if __name__ == "__main__":
    main()

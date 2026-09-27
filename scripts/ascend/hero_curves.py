"""Render SwanLab loss, optimizer and internal-health histories as PNG panels.

Usage: python -m scripts.ascend.hero_curves --run USER/PROJECT/RUN --out output/curves
Use --highlight-steps START END to mark an interval for review.
"""

import os
import argparse
from pathlib import Path

from scripts.ascend.hero_watch import fetch

KEYS = [
    "train/loss",
    "train/grad_norm",
    "train/lr",
    "train/samples_per_sec",
    "train/consecutive_skips",
    "eval/loss",
    "eval_live/loss",
    "health/update_weight_ratio_muon",
    "health/update_weight_ratio_aux",
    "health/attn_qk_gain_max",
    "health/attn_qk_gain_mean",
    "health/ema_rel_distance",
    "stability/conditioning/update_rms",
    "stability/conditioning/condition_rms",
    "stability/conditioning/condition_caption_ratio",
    "stability/conditioning/condition_time_ratio",
    "stability/conditioning/hidden_rms",
    "stability/conditioning/hidden_caption_ratio",
    "stability/conditioning/hidden_time_ratio",
    "stability/conditioning/negative_tail_fraction",
    "stability/summary/residual_rms_max",
    "stability/summary/mlp_ratio_max",
    "stability/summary/gate_rms_max",
    "stability/summary/branch_norm_gain_max",
] + [
    f"stability/sampled_weights/{role}/{stat}"
    for role in ("conditioning", "attention", "ffn", "modulation")
    for stat in ("weight_rms_mean", "update_ratio_mean", "radial_mean",
                 "decay_mean")
]


def series(data, key):
    pts = data.get(key) or []
    return ([p[0] for p in pts], [p[1] for p in pts])


def panel(ax, data, key, label=None, roll=False, logy=False, highlight=None):
    s, v = series(data, key)
    if not s:
        ax.text(0.5, 0.5, f"{key}\nno data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return
    if roll and len(v) > 2000:
        idx = list(range(0, len(v), max(1, len(v) // 4000)))
        s_r = [s[i] for i in range(0, len(s), max(1, len(s) // 4000))]
        ax.plot(s_r, [v[i] for i in idx], lw=0.3, alpha=0.25, color="C0")
        # rolling median subsampled
        w = 51
        sm, vm = [], []
        import statistics
        for i in range(0, len(v), 25):
            vm.append(statistics.median(v[max(0, i - w):i + 1]))
            sm.append(s[i])
        ax.plot(sm, vm, lw=1.0, color="C0")
    else:
        ax.plot(s, v, lw=0.9)
    if logy:
        ax.set_yscale("log")
    if highlight:
        ax.axvspan(*highlight, color="red", alpha=0.08)
    ax.set_title(label or key, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="SwanLab user/project/run-id")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--highlight-steps", type=int, nargs=2, metavar=("START", "END"),
                        help="optional interval to shade in each panel")
    args = parser.parse_args()
    from swanlab import Api
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    run = Api().run(path=args.run)
    data = fetch(run, KEYS)
    print("fetched %d keys; steps %s..%s" % (
        len(data),
        min((p[0] for pts in data.values() for p in pts), default=None),
        max((p[0] for pts in data.values() for p in pts), default=None)))

    groups = {
        "loss_eval": [
            ("train/loss", dict(roll=True)),
            ("eval/loss", {}),
            ("eval_live/loss", {}),
            ("health/ema_rel_distance", {}),
        ],
        "opt_health": [
            ("train/grad_norm", dict(roll=True)),
            ("train/lr", {}),
            ("train/samples_per_sec", {}),
            ("train/consecutive_skips", {}),
            ("health/update_weight_ratio_muon", {}),
            ("health/update_weight_ratio_aux", {}),
        ],
        "conditioning": [
            ("stability/conditioning/condition_rms", {}),
            ("stability/conditioning/condition_time_ratio", {}),
            ("stability/conditioning/condition_caption_ratio", {}),
            ("stability/conditioning/update_rms", {}),
        ],
        "hidden": [
            ("stability/conditioning/hidden_rms", {}),
            ("stability/conditioning/hidden_time_ratio", {}),
            ("stability/conditioning/hidden_caption_ratio", {}),
            ("stability/conditioning/negative_tail_fraction", {}),
        ],
        "weights": [
            ("stability/sampled_weights/conditioning/weight_rms_mean", {}),
            ("stability/sampled_weights/conditioning/radial_mean", {}),
            ("stability/sampled_weights/conditioning/decay_mean", {}),
            ("stability/sampled_weights/attention/weight_rms_mean", {}),
            ("stability/sampled_weights/ffn/weight_rms_mean", {}),
            ("stability/sampled_weights/modulation/weight_rms_mean", {}),
        ],
        "branch": [
            ("stability/summary/residual_rms_max", {}),
            ("stability/summary/gate_rms_max", {}),
            ("stability/summary/mlp_ratio_max", {}),
            ("stability/summary/branch_norm_gain_max", {}),
            ("health/attn_qk_gain_max", {}),
            ("health/attn_qk_gain_mean", {}),
        ],
    }

    # Derived: absolute time-varying spread = condition_rms x time_ratio.
    s1, cr = series(data, "stability/conditioning/condition_rms")
    s2, tr = series(data, "stability/conditioning/condition_time_ratio")
    if s1 and s2:
        tr_map = dict(zip(s2, tr))
        cr_map = dict(zip(s1, cr))
        xs = [x for x in s1 if x in tr_map]
        data["_derived/spread"] = [(x, tr_map[x] * cr_map[x], None)
                                   for x in xs]
        groups["conditioning"].insert(1, ("_derived/spread", {}))

    for name, specs in groups.items():
        n = len(specs)
        cols = 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(13, 2.6 * rows))
        axes = [a for row in (axes if rows > 1 else [axes]) for a in
                (row if isinstance(row, (list, tuple)) else
                 list(row) if hasattr(row, "__iter__") else [row])]
        for ax, (key, kw) in zip(axes, specs):
            panel(ax, data, key, highlight=args.highlight_steps, **kw)
        for ax in axes[len(specs):]:
            ax.axis("off")
        fig.suptitle(f"hero patrol — {name}", fontsize=10)
        fig.tight_layout()
        path = os.path.join(out, f"{name}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()

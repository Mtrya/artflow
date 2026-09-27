#!/usr/bin/env python3
"""Read-only watch report for the Ascend pretraining hero run.

Pulls metric windows from SwanLab cloud and prints a report over the
structural ("surgery": optimizer, architecture, execution) and the internal
("internal": conditioning path, feature statistics, functional response)
metric surfaces, plus the evaluation and data context needed to read them.

Signals printed at the end are triage leads, not calibrated thresholds; the
authoritative interpretation rules are in notes/ascend_pretraining.md.

Deep-dive keys that never reach SwanLab -- `stability/blocks/NN/*` and
`stability/weights/<param>/grad_rms` -- live only in the run's
`stability.jsonl` on sj-ssd3; read it from the job shell when a signal fires.

Usage:
  .venv/bin/python scripts/ascend/hero_watch.py --run <user>/<project>/<run-id>
      [--window 400] [--probes 6]
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics as st


# Fine-grained per-update window (full-resolution fetch; keep this list short).
FINE_KEYS = [
    "train/loss",
    "train/grad_norm",
    "train/update_applied",
    "train/consecutive_skips",
]
SAMPLED_KEYS = [
    "train/lr",
    "train/samples_per_sec",
    "train/mem_peak_gb",
    "eval/loss",
    "eval_live/loss",
    "perf/eval_seconds",
]
# Cadence 250 or 100; fetched as exact tails rather than downsampled.
TAIL_KEYS = [
    "health/update_weight_ratio_muon",
    "health/update_weight_ratio_aux",
    "health/attn_qk_gain_max",
    "health/attn_qk_gain_mean",
    "health/ema_rel_distance",
    "stability/panel/rows_per_timestep",
    "stability/panel/text_tokens",
    "stability/probe_seconds",
    "stability/update_applied",
    "stability/conditioning/update_rms",
    "stability/conditioning/shared_update_rms",
    "stability/conditioning/centered_update_rms",
    "stability/conditioning/c2_shared_shift_rms",
    "stability/conditioning/c2_bias_shift_rms",
    "stability/conditioning/condition_rms",
    "stability/conditioning/condition_caption_ratio",
    "stability/conditioning/condition_time_ratio",
    "stability/conditioning/hidden_rms",
    "stability/conditioning/hidden_caption_ratio",
    "stability/conditioning/hidden_time_ratio",
    "stability/conditioning/negative_tail_fraction",
    "stability/response/prediction_change_rms",
    "stability/response/conditioning_prediction_change_rms",
    "stability/response/conditioning_gain",
    "stability/summary/residual_rms_max",
    "stability/summary/attention_ratio_max",
    "stability/summary/mlp_ratio_max",
    "stability/summary/gate_rms_max",
    "stability/summary/branch_norm_gain_max",
] + [
    f"stability/sampled_weights/{role}/{stat}"
    for role in ("conditioning", "attention", "ffn", "modulation")
    for stat in ("weight_rms_mean", "update_ratio_mean", "radial_mean",
                 "energy_mean", "decay_mean", "norm_sq_change_mean")
] + [
    f"stability/response/{t}/{m}"
    for t in ("t010", "t050", "t090")
    for m in ("loss_before", "loss_after", "loss_held_condition",
              "loss_twice_condition_update", "conditioning_loss_delta",
              "second_difference")
]
# Eval bands: reported together with their sample counts so evidence
# strength is visible next to every number.
BANDS = ("le128", "129_256", "257_512", "513_1024", "1025_2048")
# bf16 keeps ~2^-8 relative resolution at any magnitude; the conditioning
# path's usable structure is measured against this floor.
EPS = 2.0 ** -8
CTX_KEYS = [
    "policy/length_preference_beta",
    "caption/dropout_rate",
    "caption/selected_mean_tokens",
    "caption/selected_p50",
    "caption/selected_p90",
    "caption/selected_p99",
    "caption/weight_mean",
    "caption/grad_weighted_mean_tokens",
    "exec/padding_fraction",
    "exec/samples_per_micro_batch",
    "repetition/repeat_rate",
    "repetition/unique_rows_cumulative",
    "eval/paired_images",
    "eval_live/paired_images",
] + [f"eval/loss/band_{b}" for b in BANDS] \
  + [f"eval/samples/band_{b}" for b in BANDS] \
  + [f"eval/shortfall/band_{b}" for b in BANDS] \
  + ["eval/loss_t015", "eval/loss_t040", "eval/loss_t065", "eval/loss_t090",
     "eval_live/loss_t015", "eval_live/loss_t040", "eval_live/loss_t065",
     "eval_live/loss_t090"]


def fetch(run, keys, **kw):
    """{key: [(step, value, ts)]} for the scalar series that exist."""
    out = {}
    for item in run.metrics(keys=keys, **kw)["list"]:
        pts = [(p.get("step", p.get("index")), p.get("value", p.get("data")),
                p.get("timestamp")) for p in item["metrics"]]
        out[item["key"]] = pts
    return out


def f(x, nd=4):
    if x is None:
        return "n/a"
    if x == 0:
        return "0"
    a = abs(x)
    if a < 1e-3 or a >= 1e5:
        return f"{x:.{nd}g}"
    return f"{x:.{nd}f}".rstrip("0").rstrip(".") if a < 1 else f"{x:.{nd}g}"


def trend(pts, nd=4, n=6):
    """Compact 'v -> v -> v' rendering of the tail of a series."""
    if not pts:
        return "no data"
    tail = pts[-n:]
    return " -> ".join(f(v, nd) for _, v, _ in tail)


def mono(pts, n=5):
    """Return 'up'/'down'/'flat' over the last n values."""
    if len(pts) < n:
        return "n/a"
    vals = [v for _, v, _ in pts[-n:]]
    if all(b > a for a, b in zip(vals, vals[1:])):
        return "up"
    if all(b < a for a, b in zip(vals, vals[1:])):
        return "down"
    return "mixed"


def window(pts):
    """Window statistics for the fine per-update series."""
    vals = [v for _, v, _ in pts]
    steps = [s for s, _, _ in pts]
    mx = max(vals)
    return {"last": vals[-1], "median": st.median(vals), "max": mx,
            "max_step": steps[vals.index(mx)], "first": vals[0],
            "lo": steps[0], "hi": steps[-1], "n": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="SwanLab user/project/run-id")
    ap.add_argument("--window", type=int, default=400,
                    help="last updates summarised at full resolution")
    ap.add_argument("--probes", type=int, default=6,
                    help="stability observations to print")
    args = ap.parse_args()

    from swanlab import Api

    run = Api().run(path=args.run)
    signals: list[str] = []

    # ---- liveness / progress -------------------------------------------
    fine = fetch(run, FINE_KEYS, range_query={"tail": args.window})
    loss_pts = fine.get("train/loss", [])
    if not loss_pts:
        print(f"run {args.run}: no train/loss data")
        return
    step, _, ts = loss_pts[-1]
    age = (dt.datetime.now(dt.UTC).timestamp() - ts / 1000) / 60
    print(f"run {args.run}  state={run.state}")
    print(f"progress: step {step}, last point {age:.1f} min ago")
    if age > 20:
        signals.append(f"STALL: no new metrics for {age:.0f} min")

    sampled = fetch(run, SAMPLED_KEYS, sample=1500)
    tail = fetch(run, TAIL_KEYS + CTX_KEYS, range_query={"tail": max(args.probes, 8)})
    ctx = fetch(run, CTX_KEYS, sample=1500)

    def one(store, key):
        pts = store.get(key, [])
        return pts[-1][1] if pts else None

    lr = one(sampled, "train/lr")
    print(f"lr={f(lr)}  samples/s={f(one(sampled,'train/samples_per_sec'),5)}  "
          f"mem_peak={f(one(sampled,'train/mem_peak_gb'))} GiB  "
          f"eval_seconds={f(one(sampled,'perf/eval_seconds'))}  "
          f"probe_seconds={f(one(tail,'stability/probe_seconds'))}")
    print(f"panel: rows_per_timestep={f(one(tail,'stability/panel/rows_per_timestep'),3)}"
          f"  text_tokens={f(one(tail,'stability/panel/text_tokens'),4)}"
          f"  stability update_applied={f(one(tail,'stability/update_applied'),3)}")

    # ---- fine per-update window ----------------------------------------
    print(f"\n[window] last {args.window} updates, full resolution")
    ls = window(loss_pts)
    gs = window(fine.get("train/grad_norm", []))
    print(f"  train/loss      last={f(ls['last'])} median={f(ls['median'])} "
          f"p95={f(sorted(v for _, v, _ in loss_pts)[int(0.95 * (len(loss_pts) - 1))])} "
          f"max={f(ls['max'])}@step{ls['max_step']} [steps {ls['lo']}-{ls['hi']}]")
    print(f"  train/grad_norm last={f(gs['last'])} median={f(gs['median'])} "
          f"max={f(gs['max'])}@step{gs['max_step']} "
          f"clip_frac(>1.0)={f(sum(1 for _, v, _ in fine['train/grad_norm'] if v > 1.0) / gs['n'], 3)}")
    half = len(loss_pts) // 2
    if half >= 25:
        old = st.median([v for _, v, _ in loss_pts[:half]])
        new = st.median([v for _, v, _ in loss_pts[half:]])
        print(f"  loss median     first half {f(old)} -> second half {f(new)} "
              f"({'+' if new >= old else ''}{100 * (new / old - 1):.1f}%)")
        if new > old * 1.15:
            signals.append(f"LOSS-RISE: last-half median {f(new)} vs earlier "
                           f"{f(old)} (+{100 * (new / old - 1):.0f}%)")
    if ls["max"] > 2.5 * ls["median"]:
        signals.append(f"LOSS-TAIL: window max {f(ls['max'])} vs median "
                       f"{f(ls['median'])} at step {ls['max_step']}")
    if gs["max"] > 10 * gs["median"]:
        signals.append(f"GRAD-TAIL: window max {f(gs['max'])} vs median "
                       f"{f(gs['median'])} at step {gs['max_step']}")
    sk = fine.get("train/consecutive_skips", [])
    ap_ = fine.get("train/update_applied", [])
    # Both are constants in this revision: the finite guard raises instead of
    # skipping, so these series only move if the run's behavior changed.
    if sk and max(v for _, v, _ in sk) > 0:
        signals.append("NO-UPDATE: consecutive_skips > 0 (series should be "
                       "constant 0 in this revision)")
    if ap_ and min(v for _, v, _ in ap_) < 1:
        signals.append("NO-UPDATE: update_applied < 1")

    # ---- surgery: optimizer, architecture, execution --------------------
    print("\n[surgery] optimizer / architecture / execution")
    print("  whole-optimizer update ratios (health, every 250):")
    for key in ("health/update_weight_ratio_muon", "health/update_weight_ratio_aux"):
        pts = tail.get(key, []) or sampled.get(key, [])
        if pts:
            print(f"    {key.split('/')[-1]}: {trend(pts)}  ({mono(pts)})")
    print("  sampled tensors (unweighted mean over ~15 matrices; "
          "roles: conditioning / attention / ffn / modulation):")
    for role in ("conditioning", "attention", "ffn", "modulation"):
        cells = []
        for stat in ("weight_rms_mean", "update_ratio_mean", "radial_mean",
                     "energy_mean", "decay_mean"):
            points = tail.get(f"stability/sampled_weights/{role}/{stat}", [])
            cells.append(f"{stat.replace('_mean','')}={f(points[-1][1]) if points else 'n/a'}")
        print(f"    {role:12s} " + " ".join(cells))
    print("  branch amplification + gains (panel maxima; argmax can switch layers):")
    for key in ("stability/summary/residual_rms_max",
                "stability/summary/attention_ratio_max",
                "stability/summary/mlp_ratio_max",
                "stability/summary/gate_rms_max",
                "stability/summary/branch_norm_gain_max",
                "health/attn_qk_gain_max", "health/attn_qk_gain_mean",
                "health/ema_rel_distance"):
        pts = tail.get(key, []) or sampled.get(key, [])
        if pts:
            # Growth rate per 1000 probes-steps, so the trend can be read as a
            # rate instead of a size.
            rate = ""
            if len(pts) >= 2 and pts[-1][0] > pts[0][0]:
                span = pts[-1][0] - pts[0][0]
                first, last = pts[0][1], pts[-1][1]
                if first:
                    rate = (f"  x{(last / first) ** (1000.0 / span):.3f}/1k steps")
            print(f"    {key.split('/')[-1]:22s} {trend(pts, 4, 5)}  ({mono(pts)}){rate}")
    print("    note: attention_ratio_max / mlp_ratio_max are dominated by the "
          "first block, whose input RMS is ~0.5 (small denominator); a large "
          "ratio there is not branch explosion. Before concluding anything "
          "about a branch, read blocks/NN/{input,output}_rms in the run's "
          "stability.jsonl.")

    # ---- internal: conditioning path, features, response ----------------
    print("\n[internal] conditioning path / feature statistics / response")
    print("  conditioning displacement (one optimizer step, fixed panel):")
    for key in ("stability/conditioning/update_rms",
                "stability/conditioning/shared_update_rms",
                "stability/conditioning/centered_update_rms",
                "stability/conditioning/c2_shared_shift_rms",
                "stability/conditioning/c2_bias_shift_rms"):
        pts = tail.get(key, [])
        if pts:
            print(f"    {key.split('/')[-1]:22s} {trend(pts, 4, 5)}  ({mono(pts)})")
    print("  conditioning features (pre-update; ratios are centered/RMS):")
    for key in ("stability/conditioning/condition_rms",
                "stability/conditioning/condition_caption_ratio",
                "stability/conditioning/condition_time_ratio",
                "stability/conditioning/hidden_rms",
                "stability/conditioning/hidden_caption_ratio",
                "stability/conditioning/hidden_time_ratio",
                "stability/conditioning/negative_tail_fraction"):
        pts = tail.get(key, [])
        if pts:
            print(f"    {key.split('/')[-1]:22s} {trend(pts, 4, 5)}  ({mono(pts)})")
    print("  output movement:")
    for key in ("stability/response/prediction_change_rms",
                "stability/response/conditioning_prediction_change_rms",
                "stability/response/conditioning_gain"):
        pts = tail.get(key, [])
        if pts:
            print(f"    {key.split('/')[-1]:34s} {trend(pts, 4, 5)}  ({mono(pts)})")

    ref = tail.get("stability/conditioning/update_rms", [])
    print("  conditioning relative signal (bf16 relative eps = 0.0039):")
    cm = {s: v for s, v, _ in tail.get("stability/conditioning/condition_rms", [])}
    um = {s: v for s, v, _ in tail.get("stability/conditioning/update_rms", [])}
    rel = [(s, um[s] / cm[s], None) for s in um if s in cm and cm[s]]
    if len(rel) >= 2:
        print(f"    update_rms/condition_rms   {trend(rel, 3, 5)}  "
              f"last={f(rel[-1][1], 3)} = {rel[-1][1] / EPS:.1f}x bf16 eps")
        if rel[-1][1] < 2 * EPS:
            signals.append(f"COND-QUANT: a single step moves the condition "
                           f"vector by {rel[-1][1] / EPS:.1f}x bf16 eps "
                           f"({f(rel[-1][1], 3)})")
    for key in ("stability/conditioning/condition_caption_ratio",
                "stability/conditioning/hidden_caption_ratio",
                "stability/conditioning/condition_time_ratio",
                "stability/conditioning/hidden_time_ratio"):
        pts = tail.get(key, [])
        if pts:
            print(f"    {key.split('/')[-1]:26s} {trend(pts, 4, 5)}  ({mono(pts)})")
    cc = tail.get("stability/conditioning/condition_caption_ratio", [])
    if cc and cc[-1][1] < 0.01:
        signals.append(f"COND-CAPTION-FLOOR: caption-borne part of the condition "
                       f"vector is {f(cc[-1][1], 3)} of its RMS "
                       f"(~{cc[-1][1] / EPS:.1f} bf16 levels)")

    # Captured event described in notes/ascend_pretraining.md: the matrix update
    # acting on the shared hidden feature caused a harmful conditioning shift.
    # Update RMS 0.206 versus spread 0.196 characterized that event; the ratio
    # is a diagnostic reference, not a calibrated failure boundary.
    rat = {s: v for s, v, _ in tail.get("stability/conditioning/condition_caption_ratio", [])}
    marg = [(s, (cm[s] * rat[s]) / um[s], None)
            for s in um if s in cm and s in rat and um[s]]
    if len(marg) >= 2:
        print(f"    spread/update margin       {trend(marg, 3, 5)}  "
              f"last={f(marg[-1][1], 3)}  (1.0 = the H200 spike state)")
        if marg[-1][1] < 2.0:
            signals.append(f"COND-MARGIN: one conditioning update moves "
                           f"{1 / marg[-1][1]:.2f}x the between-sample spread "
                           f"(margin {f(marg[-1][1], 3)}; H200 spiked at ~0.95)")

    print(f"  response table (last {min(args.probes, len(ref))} probes; "
          "loss_before/after/held are one-step counterfactuals on the panel):")
    for idx in range(max(0, len(ref) - args.probes), len(ref)):
        step_i = ref[idx][0]
        cell = []
        for t in ("t010", "t050", "t090"):
            def at(m):
                pts = tail.get(f"stability/response/{t}/{m}", [])
                return pts[idx][1] if idx < len(pts) else None
            before, after = at("loss_before"), at("loss_after")
            held, twice = at("loss_held_condition"), at("loss_twice_condition_update")
            d = None if before is None or after is None else after - before
            cdelta = at("conditioning_loss_delta")
            second = at("second_difference")

            def sci(x):
                return "n/a" if x is None else f"{x:+.2e}"

            cell.append(f"{t}[d={sci(d)} cond={sci(cdelta)} "
                        f"held={f(held, 4)} twice={f(twice, 4)} "
                        f"2nd={sci(second)}]")
        print(f"    step {step_i}: " + " ".join(cell))

    # ---- eval + data context -------------------------------------------
    print("\n[eval] EMA vs live (probe every 500; bands can be thin evidence)")
    for key in ("eval/loss", "eval_live/loss"):
        pts = sampled.get(key, [])
        if pts:
            print(f"  {key}: {trend(pts, 5, 4)}")
    for tag in ("t015", "t040", "t065", "t090"):
        a = sampled.get(f"eval/loss_{tag}", [])
        b = sampled.get(f"eval_live/loss_{tag}", [])
        if a or b:
            print(f"  {tag}: ema={f(a[-1][1]) if a else 'n/a'} "
                  f"live={f(b[-1][1]) if b else 'n/a'}")
    weak = []
    for b in BANDS:
        l = one(ctx, f"eval/loss/band_{b}")
        n = one(ctx, f"eval/samples/band_{b}")
        short = one(ctx, f"eval/shortfall/band_{b}")
        if n is None:
            weak.append(f"{b}: no key")
        elif short:
            weak.append(f"{b}: short by {f(short,3)} of budget")
        print(f"  band {b:11s} loss={f(l)} samples={f(n,4)} shortfall={f(short,4)}")
    paired = one(ctx, "eval/paired_images")
    print(f"  paired_images={f(paired,4)} "
          "(images contributing to >=2 bands; band-vs-band comparisons rest "
          "on these)")
    if weak:
        print("  note: thinner-than-budget bands above 128 tokens are expected "
              "while length_preference_beta is still near -1 (short captions "
              "preferred); read them only once beta has ramped and the bands "
              "fill. A missing band key means no evidence, not a perfect score.")
    if all(one(ctx, f"eval/samples/band_{b}") == 0 for b in BANDS):
        signals.append("EVIDENCE-EMPTY: every eval band empty")

    print("\n[context] data / caption / repetition (most of these drift by design)")
    for key in ("policy/length_preference_beta", "caption/selected_mean_tokens",
                "caption/selected_p50", "caption/selected_p90",
                "caption/weight_mean", "caption/grad_weighted_mean_tokens",
                "caption/dropout_rate", "exec/padding_fraction",
                "exec/samples_per_micro_batch", "repetition/repeat_rate",
                "repetition/unique_rows_cumulative"):
        pts = ctx.get(key, [])
        if pts:
            print(f"  {key:38s} {trend(pts, 4, 4)}")
    print("  dropout_rate should be flat at the configured 0.1; beta/token "
          "lengths/padding drift by design; ema_rel_distance rises during "
          "warmup and the LR ramp by construction")

    # ---- signals --------------------------------------------------------
    # Persistence triggers, mirroring the handoff's three-probe review rule.
    # Both are relative to the panel loss and must clear a noise floor: bf16
    # rounding on tiny conditioning displacements produces sign flips and
    # ~1e-4 relative excursions that must not page anyone.
    FLOOR = 2e-3  # 0.2 % of the panel loss
    for t in ("t010", "t050", "t090"):
        full, cond = [], []
        for idx in range(len(ref)):
            def at(m):
                pts = tail.get(f"stability/response/{t}/{m}", [])
                return pts[idx][1] if idx < len(pts) else None
            before, after = at("loss_before"), at("loss_after")
            cd = at("conditioning_loss_delta")
            if before is not None and after:
                full.append((after - before) / after)
            if cd is not None and after:
                cond.append(cd / after)
        if full:
            print(f"  response persistence {t}: full-step rel "
                  f"{', '.join(f'{x:+.1e}' for x in full[-3:])} | cond rel "
                  f"{', '.join(f'{x:+.1e}' for x in cond[-3:])}")
        if len(full) >= 3 and all(x > FLOOR for x in full[-3:]):
            signals.append(f"FULL-STEP-3@{t}: net panel loss rose by "
                           f"{', '.join(f'{x:.2%}' for x in full[-3:])} in the "
                           "last 3 probes")
        if len(cond) >= 3 and all(x > FLOOR for x in cond[-3:]) \
                and cond[-3] < cond[-2] < cond[-1]:
            signals.append(f"COND-PATH-3@{t}: conditioning change hurt the loss "
                           f"with the trunk held fixed, growing over 3 probes "
                           f"({', '.join(f'{x:.2%}' for x in cond[-3:])})")

    # Structural rise is only actionable together with a functional or loss
    # signal, per the handoff.
    risen = [k.split("/")[-1] for k in ("stability/summary/residual_rms_max",
                                        "stability/summary/gate_rms_max",
                                        "stability/summary/attention_ratio_max",
                                        "stability/summary/mlp_ratio_max")
             if mono(tail.get(k, []), 5) == "up"]
    harmed = [s for s in signals if s.startswith(("FULL-STEP-3", "COND-PATH-3",
                                                  "LOSS-RISE", "LOSS-TAIL",
                                                  "GRAD-TAIL"))]
    if risen and harmed:
        signals.append("STRUCT-RISE-CORROBORATED: "
                       + "/".join(risen) + " rising monotonically alongside "
                       + harmed[0].split(":")[0])
    elif risen:
        print(f"\nnote: {', '.join(risen)} rising monotonically across the last "
              "probes (watch item; structural growth alone is not an alarm, and "
              "the argmax can switch layers between probes)")

    print("\n[signals]" + ("" if signals else " none"))
    for s in signals:
        print(f"  - {s}")
    if signals:
        print("  (triage leads only. If one fires: read "
              "notes/ascend_pretraining.md 'Reading the monitoring signals' "
              "and 'Review and intervention', then per-block/per-parameter detail in "
              "the run's stability.jsonl on sj-ssd3 via the job shell.)")


if __name__ == "__main__":
    main()

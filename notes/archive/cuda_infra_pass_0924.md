# Infrastructure pass

> Archived record. Current pretraining decisions and status are in
> [the Ascend plan](../ascend_pretraining_0924.md); old launch commands and
> configuration switches require the source revision used for that run.

Current update, September 24, 2026: the user chose **pure Ascend pretraining**;
subsequent pretraining debugging and optimization target the Ascend stack.
See [the decision and its evidence boundaries](../ascend_pretraining_0924.md).
The four-H200 hero stopped after recurrent spikes and EMA sample collapse.
Its 256p infrastructure qualification remains a measured historical reference,
but did not establish long-run model stability. H200 measurements and separate
allocation records remain in [H200 qualification and launch](h200_smoke_0923.md).
The ledger below is historical.

## September 18 qualification ledger

Updated September 18, 2026. **Stage 5 readiness is not established.** The user
approved moving the hero run to eight H100 or H200 GPUs with a total budget of
800 GPU-hours, plus a separate 16-GPU-hour migration pilot. Production T remains
the user's decision. H200 pilot results and RTX 4090 reference results are
separated below; neither establishes final H200 acceptance. The pilot allowance
closed with 0.38444 GPU-hours remaining. On September 18 the user approved a
separate **48 additional H200 GPU-hours** for qualification. Track new allocations
against that extension without borrowing from the 800-hour hero budget.

**Scheduling reality check (user, 2026-09-18):** hero hardware is not fixed to
H100/H200. Both Hopper queues have been effectively unschedulable for about two
days (V79/V80/V81 never observed a container start; 2026-09-18 availability
shows 0 free GPUs in every H100/H200 group). The launch goes to whichever side
delivers an allocation. **Budget rule per path (user, 2026-09-18):** on 4090,
drop the GPU-hour budget — fix a total step count and run at priority 1
(preemptible idle-fill) until done; 8× 4090 is preferred if a node is
schedulable, otherwise 4× 4090. On H200, keep the 800-GPU-hour budget, and the
H200 execution plan is not yet optimally tuned (e.g. batch-size re-screening);
the measured ~2.45× per-GPU speedup over 4090 is treated as a lower bound, not
the final figure. H200 qualification continues while queued (queue-only time
spends no budget), but hero launch must not block on Hopper indefinitely. The
4090 reference pass is already nearly complete (V76–V78), so the fallback path
needs mainly a 4-rank execution check: per-rank memory is unchanged
(accumulation 2/10/14 preserves the frozen effective-batch targets at four
ranks), while wall time and NCCL behavior differ from the eight-rank
reference.

The bounded migration pilot measures all three resolutions with the reference
bucket/accumulation settings, fresh compiler caches, eight ranks and profiler
windows. It must establish runtime compatibility and actual interconnect,
memory and critical-path evidence before hardware-specific re-screening. Its
phase completion alone cannot establish final acceptance or an all-in cost.

Migration pilot `infra-0917-h200-train-8g-v82` uses `分布式训练空间`, project
`具有能动性的科学大模型`, `训练区-H200-1号机房`, eight H200 GPUs, requested
priority 4. Runtime is capped at 1.75 hours (14 GPU-hours), with no automatic
retry or container retention. Monitor cadence is two hours. Source is frozen
`infra-0917-code-v78`; runner/launcher pins SHA-256:
`b6db2f97e14a8769bea1020daf2992edd869ffd35f3987b87dac065cc5c89ac6`.
It reuses V80's pinned launcher and output tag after confirming that neither
the output nor its compiler cache existed. Pilot results are reported below.
The preceding V81 H100 request reported `20 / NORMAL`, so effective
high priority was **not verified**. The installed CLI forwards request 4 as
`task_priority`, while status prefers a separate `priority` field over
`task_priority`; this is a possible reporting ambiguity, not proof of high priority.

The initial development-zone submission V79 was stopped before this replacement.
The user clarified that development-zone or sub-eight-GPU requests are effectively
low priority even when requesting high priority. V79 reported NORMAL and never
had an observed container start. Its closed conservative creation-to-finish bound
is 0.64889 H100 GPU-hours (includes queue time), with zero observed runtime.
Training-zone V80 was then stopped because scheduling events showed the primary
project using 430 of its 432-GPU parent quota; another eight GPUs could not fit.
Its additional conservative creation-to-finish bound is 0.22000 GPU-hours,
again with no observed container start. Both jobs are terminal before V81 is
submitted. Counting both whole bounds, V81's 14-hour runtime reservation leaves
1.13111 of the approved 16 H100/H200 GPU-hours for provisioning/termination margin.
This is a reservation bound, not an invoice or proof of actual zero-cost queueing.

At the user's request V81 was stopped before switching to H200. Its final
state is stopped; 169 retained events contain failed scheduling and no Scheduled,
Pulling or Started event, scheduled monitor checks remained queued, and its
output plus both compiler caches are absent. The raw ledger's 35.32222 GPU-hours
is its creation-to-finish queue-inclusive bound, not measured allocation. Treat
this as a documented queue-only exclusion, retaining the raw ledger rather than
overwriting it. V82 keeps the 14-GPU-hour runtime cap and uses the verified-unused
V80 output/cache paths; no H100/H200 pilot training was observed before V82.

V82 succeeded September 17 at 15:32:32 UTC on `qb-prod-gpu615`, with eight
NVIDIA H200 devices and NV18 links between every GPU pair. Source pins, three
config renders and eight-rank CUDA guard primitives passed. Each resolution
completed 128 updates on every rank, with aligned steps and finite losses.

The following windows use the slowest rank per update, exclude the first 50
updates and flagged profiler updates, and retain 75 updates per resolution.
They are reference-bucket pilot measurements, not final accepted H200 rates.

| Resolution | Median s/update | Mean s/update | Maximum s/update | Mean global batch | Peak allocated / reserved GB | Whole subprocess s |
|---|---:|---:|---:|---:|---:|---:|
| 256p | 0.52460 | 2.72222 | 163.81671 | 682.453 | 46.771 / 46.982 | 1190.995 |
| 640p | 1.91241 | 1.99204 | 7.65625 | 586.240 | 44.363 / 44.892 | 1111.531 |
| 896p | 3.11670 | 3.17174 | 6.67354 | 426.680 | 43.752 / 44.281 | 1233.297 |

Memory is decimal GB; reserved memory does not include all external CUDA/NCCL
allocations. Compiler stalls, sampling-window variation, profiler-export effects
outside flagged updates, bucket re-screening and all production overheads remain
unresolved. The 75:20:5 weighted median is 0.93177 s/update, already above the
0.900 all-in benchmark; it is not a final cost estimate. Do not cost the hero run
from the median alone or extrapolate sampled memory to untested caption/shape tails.

V82's closed scheduled-to-finish bound is 8.08444 H200 GPU-hours, versus 7.92444
observed container GPU-hours. The platform wait command incorrectly reports
running time 0s; it is not used for accounting. Including the earlier 0.86889
closed conservative bound and excluding documented queue-only V81 leaves
**7.04667 H100/H200 GPU-hours** of the approved 16-hour migration allowance.

The completed 256p three-update traces on all eight H200 ranks show activity
spans 1.617–1.660 s and GPU busy unions of 85.96–90.06% of those spans. Internal
idle is 161–229 ms; the largest recurring gaps are approximately 22–39 ms.
NCCL union time is 293–604 ms, with only 52–132 ms not temporally overlapping
other GPU events. These are profiler-window figures, not additive savings or
proof of saturation. Rank zero has about 13,640 name-classified pointwise GPU
events (446 ms summed work); fused matmuls also occur in the `other` class, so
the class table is not a reliable compute/bandwidth split. DataLoader CPU scopes
total 7.03 ms on rank zero; CPU operator events alone do not explain the largest
35.63-ms idle gap. Further attribution is needed before selecting any host or
pointwise optimization. No new GPU trial is authorized by this observation alone.

The 640p rank-zero three-update trace has 588 ms internal idle across a 5.986-s
activity span, 44,261 pointwise launches (1.280 s summed work), and 30.5 ms of
NCCL not overlapping other GPU work. This, together with the sampled memory
headroom, motivates a bounded larger-micro-batch/lower-accumulation comparison;
it does not prove that batching will improve end-to-end performance.

CPU-only proposals are in `$W/bucket_plans/hero/hopper-proposal-v84/{640p,896p}`,
generated by `scripts/bench/resize_bucket_plan.py`. They preserve existing caption
boundaries, stage intervals, dataset weights and beta-policy probabilities.
640p proposes accumulation 2 with estimated global mean 512.00162; 896p proposes
accumulation 3 with mean 400.00003. Reference means are 584.36461 and 426.04249,
not exactly the recipe's minimum targets 512 and 400. Batch sizes are scaled
proportionally, rounded and trimmed using sample-share-weighted harmonic means.
These are **unvalidated screening candidates**, not replacements for hero inputs.
Compare realized sample exposure and sample-normalized cost as well as update
latency; reducing the reference overshoot is not purely an execution speedup.
Memory tails, finite-queue effects and full eight-rank execution still need testing.
The candidate generator's seven tests pass; the combined candidate/cost suite has
26 passing tests. No additional GPU allocation was made for proposal generation.

Bounded screen V85 (`infra-0918-h200-batch-8g-v85`) failed before any training
phase on September 17 at 18:02:46 UTC. The original source-tree pin rejected
214 newly generated SwanLab files from V82. No originally pinned file changed
or disappeared. Those new logs are preserved in V82's output directory;
all pre-existing pinned entries are retained and the original manifest verifies
unchanged. No manifest was regenerated to accept drift. This failure adds a
0.18000-GPU-hour conservative scheduled-to-finish bound, leaving 6.86667 of the
16 pilot GPU-hours. It is an operational failure, not a batch-size result.

Retry V86 (`infra-0918-h200-batch-8g-v86`) keeps the same screen inputs and
uses a separate working directory with only a config-directory link back to
the frozen source. Default relative telemetry outputs and explicit SwanLab
logs now remain outside that source; Python bytecode writes stay disabled.
The corrected launcher is separately pinned; source and V85 input pins remain
unchanged. No training-code or scientific-policy change is involved.
V86 launcher SHA-256:
`961963971d2506caccbccb2774e220767b2a129c9375b4a9e48743ec2675a091`;
its manifest SHA-256:
`6f18bf71355152fddf719eee220450c622979b4f0c09759a100703592fa0bfab`.

V86 failed September 17 at 20:07:59 UTC after its 640p reference exceeded
the 320-second training deadline: it completed 27 updates with substantial
shape/startup stalls despite cache reuse. All three pin checks passed. No
candidate phase started; this is not an OOM or candidate performance result.
Its scheduled-to-finish bound is 0.90889 GPU-hours. Adjusted migration spend
is now 10.04222 GPU-hours, leaving 5.95778 of 16.

V87 screens only the two 96-update candidates, using V82 as separately collected
reference evidence (not a same-node A/B claim). Each trainer gets 1,140 seconds
within a 1,170-second phase bound; runner/outer/platform bounds are 2,400/2,460/
2,520 seconds. The platform reservation is 5.6 GPU-hours, leaving 0.35778 for
provisioning/termination. Failure remains fail-fast, with no automatic retry.
The original source and V85 input pins are unchanged; V87's runner/launcher
manifest SHA-256 is
`e5d1a90ae52a7cd16e06c6ac084284013cbcb6971fd5ba0d4fe7da26fbb76a1b`.
The candidate-only plan tests pass. The earlier focused integration suite passed
85 tests, but neither result substitutes for hardware acceptance.

V87 ended September 17 at 22:47:35 UTC. The 640p candidate completed all 96
updates on all eight ranks. The 896p candidate reached 70 updates on every rank
before its 1,140-second trainer timeout; losses were finite throughout both
recorded runs. There was no reported OOM. Whole phase durations were 1,131.60 s
and 1,142.71 s respectively. The 896p result is **partial**, not a passed screen.

| Candidate | Recorded updates/rank | Post-50 unprofiled updates | Mean / median s/update | Mean global batch | Samples/s | Peak allocated / reserved GB |
|---|---:|---:|---:|---:|---:|---:|
| 640p, accumulation 2 | 96 | 46 | 1.68708 / 1.66882 | 514.913 | 305.210 | 83.497 / 84.498 |
| 896p, accumulation 3, partial | 70 | 20 | 8.24434 / 2.82758 | 402.750 | 48.852 | 82.125 / 83.095 |

The 640p common step-index window 71–96, which excludes V82's profiler and
export spillover, gives reference/candidate 304.041/311.509 samples/s: only
about **2.46%** sample-normalized improvement. Mean update latency falls from
1.92231 to 1.66300 s, but realized batch also falls from 584.462 to 518.038.
These are different runs and sample streams, not a paired speedup proof.
Most apparent update-speed improvement is reduced batch overshoot, not better
execution efficiency. No new bucket plan is accepted on this evidence alone.

896p still has 56.828- and 57.272-second stalls at steps 53 and 69. The median
must not hide them: its post-50 mean is 8.24434 s. Using the three diagnostic
medians (V82 256p and V87 candidates) would give 0.86859 s/update, or 772.08
training-only GPU-hours at illustrative 400k steps. That leaves only 27.92 of
800 for **all** overheads, ignores the observed stalls and incomplete coverage,
and is not a valid budget forecast or a passed throughput gate. Final all-in
costing remains unproven; production T is not selected.

V87's scheduled-to-finish bound is 5.57333 H200 GPU-hours. The closed migration
ledger totals **15.61556 of 16**, including V79/V80's conservative bounds and
excluding only the documented queue-only V81 entry. No migration allocation
remains open. The 0.38444 remaining hours cannot cover a meaningful eight-rank
qualification run. Request additional explicit infra authority before any new
allocation; the 800-hour hero allowance remains untouched.

Approved qualification extension (September 18): **48 additional H200 GPU-hours**, bounded
and accounted per allocation, for selected-plan tail-memory tests, representative
curriculum/health-cadence timing and stall attribution, state/resume/transitions,
resident-state evaluation and complete overhead costing. This is a planning cap,
not a guarantee of completion; readiness still requires the stated evidence.
No broad optimizer/kernel search is proposed. New jobs use a separately recorded
qualification prefix and ledger; the initial pilot's figures above remain closed.

The first qualification allocation is `infra-0918-h200-qual-896-v88`, capped
at one allocated hour (8 GPU-hours), with 3,000-second trainer and 3,300-second
outer deadlines. It repeats the unchanged 896p candidate for 256 updates at
the stage midpoint, retains the H200 compiler cache, enables `TORCH_LOGS=recompiles`,
and reaches the production health cadence at step 250. Profiler capture and
optional CPU-wall instrumentation are off. All output stays outside pinned
source. Recompilation logging is diagnostic; assess its overhead before treating
rates as production estimates. No tail-memory or final-acceptance claim follows.

Existing V87 logs contain extensive matmul autotuning output; its stalls at
steps 53 and 69 coincide with first-seen shapes on a subset of ranks and nearly
equal 56.8/57.3-second waits across ranks. This supports compilation/autotuning
as a hypothesis, not a proven attribution or steady-state bound. V88 tests
whether guards/new shapes explain those stalls and whether they subside over
a longer window; no compiler policy is changed before that evidence.
Launcher/manifest SHA-256 respectively:
`d40364f44fb03a549764ca0b4d03dbf31a4c30b3dd920027cdc15ecfb31cc39a`,
`46af365df3b68c89937012512a458c1cb860123f39b87f43f12c5c2f701a21e8`.
The extension ledger uses prefix `infra-0918-h200-qual` and must count this
allocation's provisioning and termination as well as its training runtime.

V88 succeeded September 18 at 00:50:46 UTC: all eight ranks completed 256
finite updates, including the normal health event at 250 (slowest rank 2.87595 s).
Peak allocated/reserved memory was 82.12643/83.91334 decimal GB. Whole trainer
duration was 1,205.65966 s; the closed scheduled-to-finish bound is **2.76 H200
GPU-hours**, leaving **45.24 of the new 48-hour allowance**. No allocation remains
open at this checkpoint.

Recompilation logs identify guard failures in the compiled double-stream block,
including RoPE text-length shapes. The repeated stalls at 53/69 are now
19.303/19.567 s, with no matmul autotuning messages in this warm-cache run.
After update 69 there are no recorded updates over ten seconds. Updates 101–200
average 2.81177 s (142.789 samples/s, mean batch 401.49); 201–256 average
2.81148 s (142.356 samples/s, mean batch 400.232). The complete post-50 mean
including remaining stalls is 2.98397 s. This supports amortizable guard-driven
compilation in this midpoint window, not a guarantee about unvisited shapes or
other curriculum slices. Keep startup/recompilation excess in all-in costs;
do not change compiler semantics merely to eliminate bounded startup costs.

V89 ended September 18 at 10:29:27 UTC with exit code 0 and all three
resolutions verified: 8 ranks, 100 boundary cases, 2 cycles each, production
accumulation (reference 256p plan plus V84 640p/896p candidates at accumulation
1/2/3), health snapshots every update. Maximum allocated/reserved memory across
ranks: 256p 46.83/47.46 GB, 640p 83.53/83.91 GB, 896p 82.13/82.53 GB (decimal).
Estimated minimum headroom against the 141-GB H200: about 100 GB at 256p and
64–65 GB at 640p/896p. Headroom is sampled at update ends, not at transient
peaks, so it is an inspection input, not an automatic safety verdict. The
large margins support re-screening larger micro-batches at lower accumulation
on H200 while holding the frozen effective-batch targets; the V84 candidates
were conservative. V89's closed scheduled-to-finish bound is 18.24444 H200
GPU-hours (2h06m50s × 8), versus 6.10 observed container GPU-hours
(09:43:42–10:29:27 UTC). Adjusted qualification spend is 21.00444 of 48,
leaving **26.99556 H200 GPU-hours**. No allocation remains open.

Following the user's 2026-09-18 batch re-screen approval, more aggressive
candidates were drafted with `resize_bucket_plan.py` under
`$W/bucket_plans/hero/hopper-proposal-v92/`: 896p at accumulation 2 (estimated
global mean 400.00178 against the 400 target) and 640p at accumulation 1
(512.00167 against 512). A linear memory extrapolation from the V82/V89
measurements (about 4.24 GB per 896p micro-slot, intercept 11.5 GB; 2.25 GB
per 640p micro-slot, intercept 11.5 GB) predicts roughly 117 GB peak for 896p
accumulation 2 — feasible with margin — but roughly 155 GB for 640p
accumulation 1, which exceeds the 141-GB device and is dropped. 256p is already
at accumulation 1 with its target pinning the micro-batch, so it has no
re-screen freedom. The extrapolation is a screening gate, not a memory model.

V92 (`infra-0918-h200-qual-batch896-v92`) screens the 896p accumulation-2
candidate: phase A exercises every plan boundary twice on eight ranks
(memory/finite-update evidence only), phase B measures 96 updates at
accumulation 2, phase C measures 64 updates of the incumbent accumulation-3
candidate on the same node for a direct comparison. Per-phase bounded timeouts
cap the whole job near 1.5 allocated hours (about 12 GPU-hours worst case
against the remaining 26.99556-hour allowance). Launcher/config/plan pins:
`$W/infra-hopper-qual-pins-v92.json`. This is a screening comparison, not
final plan acceptance.

**Hero hardware fixed (user, 2026-09-18):** `hero-256p-4g-v2` scheduled and
passed the former OOM point (step 117) with the v93 plan; the user cancelled
`hero-256p-8g` and fixed the hero run to **4× RTX 4090** (priority 1,
preemptible, auto-restart; T=480,000; no GPU-hour budget). Early steps show
compile-stall rates around 10–19 s/update (new v93 bucket shapes are not in
the warm compiler cache); these are expected to subside like the V88
guard-driven stalls. H100/H200 remains only as the qualification path for the
800-hour budget variant; no further 4090 dual-queue is planned.

**H200 path dropped (user, 2026-09-18):** with the 4×4090 hero running, the
user cancelled V92 and ended Hopper qualification — no further H100/H200
queueing. Before the stop, V92 completed its phase A: the 896p
accumulation-2 candidate passed boundary memory stress (8 ranks, 100 cases,
2 cycles) with peak allocated/reserved 116.68/121.82 GB and estimated
headroom ~27 GB — memory-feasible, matching the ~117 GB linear extrapolation
to within 1 GB. The rate comparison (phases B/C) did not complete and is not
needed. V92's creation-to-finish bound is 25.58889 H200 GPU-hours
(18:47:30–21:59:25 local × 8). Final qualification ledger: **46.59333 of 48**
GPU-hours; the remaining ~1.41 hours lapse unused and the 800-hour hero
budget was never touched. The 896p acc2 plan and V92 phase-A evidence remain
on GPFS should an H200 variant ever be revisited.

**Resume preflight bug found by the first preemption (September 19).** The
running 4-rank hero was preempted around 23:45 local (priority 1) with a
complete step-2,000 checkpoint, but every auto-restart then failed in seconds:
`jobs/hero_stage_4090.sh` invoked `src.train.stage_control` without
`--world-size`, whose CLI default is 8, so the four-rank checkpoint record
(world size 4) was rejected with "checkpoint world size does not match". The
launcher now passes `--world-size "$RANKS"` on both the same-stage and
cross-stage preflight calls (synced to `$W/repo-0918`, md5-verified); the next
auto-restart read the fixed script and resumed from the step-2,000
checkpoint. Loss of training time was about 45 minutes; no checkpoint or
sampler state was affected. Same-stage resume retains the saved sampler
queues, so the continuation is the designed path, not a fresh start.

### 4-rank 4090 fallback chain smoke (V91) and hero launch

`infra-0918-chain-4g-v91` ran the diagnostic stage chain (T=40, warmup=5,
256p 0→30 with 28→30 replay, 640p 30→38, 896p 38→40) at four 4090 ranks with
fallback accumulation 2/10/14 on September 18, finishing 19:05:07 local. All
five subprocesses verified: exact stage stops, complete endpoint checkpoints
(world size 4, two schedulers, EMA), replay contract (scheduler and all rank
RNG states exact; model/optimizer/EMA differ only by the documented native
attention-backward nondeterminism), and continuous caption progression. The
scheduler step fix held at four ranks: checkpoint step 30 records
`last_epoch=30` (a pre-fix four-rank run would show 120), and its muon group
learning rate 0.004576846882342032 matches the closed-form warmup-5 cosine
value at step 30 of T=40 to all printed digits. Its creation-to-finish bound
is 4.06667 RTX 4090 GPU-hours; adjusted reference-pass spend is 111.025 of
120, leaving **8.975**. The H200 qualification ledger above is unchanged.

With the chain verified, the user-approved fallback launch followed:
`hero-256p-4g` (4×4090, accumulation 2/10/14) and `hero-256p-8g` (8×4090,
accumulation 1/5/7), both at **T=480,000**, priority 1 (preemptible
idle-fill) with platform auto-restart, using `jobs/hero_stage_4090.sh` from
the repo-0918 snapshot. Per the user's rule, whichever schedules first runs
and the other is cancelled; the launcher's writer lock prevents any
double-start. The 4090 path has no GPU-hour budget — fixed step count only.

**Four-rank 256p OOM and fix (September 18).** `hero-256p-4g` scheduled first
but crashed deterministically at optimizer step 117 across five auto-restarts
(same seed → same bucket draw): CUDA OOM in a compiled backward, PyTorch peak
about 45.5 GiB with only ~0.3 GiB card headroom. Realized global batch was
plan-consistent (~700 vs planned 652, the usual finite-queue overshoot), so
the plan itself — not sampler mis-scaling — exceeds 48-GB capacity on this
four-rank/accumulation-2 path even though the eight-rank stress passed. The
swanlab side effect: six identically named `hero-256p` runs (initial plus
restarts), crashed containers left in zombie RUNNING state; there was never a
second concurrent writer (lock held; `hero-256p-8g` never started).
`hero-256p-4g` was stopped after the fifth crash. Fix: a four-rank-specific
256p plan `bucket_plans/hero/fallback4r-proposal-v93/256p-acc3` (micro
≈0.654× reference, **accumulation 3**, estimated global mean 640.002 ≥ the
640 target; ~33% activation-memory cut), selected automatically by
`jobs/hero_stage_4090.sh` only for the 4-rank 256p case; 640p/896p keep
accumulation 10/14 on the standard plans (their eight-rank 48-GB stress
passed with much larger margins). Relaunched as `hero-256p-4g-v2`; the 8-card
job remains queued and the first-to-schedule rule still applies. The crash
attempts wrote no checkpoints (first checkpoint is step 2,000), so the
relaunch starts cleanly.

Next, V89 (`infra-0918-h200-qual-memory-v89`) exercises every plan boundary
twice on eight H200 GPUs at 256p/640p/896p, using accumulation 1/2/3,
the reference 256p plan and the V84 high-resolution candidate plans. Health
snapshots run at every synthetic update; periodic allocator cleanup stays off.
Synthetic stress is memory/finite-update evidence only, never throughput or
sampling evidence. The job is fail-fast, with 240-second preparation,
2,400-second training and 60-second verification caps per resolution, an
8,500-second outer timeout, and a 2.5-hour platform cap (20 GPU-hours).
Its reservation leaves 25.24 qualification hours before further provisioning;
actual closed spend determines later allocations, not the reservation alone.
V89 launcher/manifest SHA-256 respectively:
`685d2000aed05a619747b8a0b35b32ce51beec030241c88c68c53860b7805c43`,
`5af1640fd0a87a4a0c7f42ab6c78711f7566bd1fca23551948156c2846c6c907`.

Local validation orchestration now accepts explicit hardware-specific stage
configs and per-stage accumulation, propagated consistently to rates, memory
stress, state transitions and replay. Thirteen focused tests pass. This does
not change the frozen trainer or V89 inputs; the updated orchestration must be
separately pinned before it is used for subsequent acceptance jobs.

The V85/V86 screen design used eight H200 GPUs in the same
training-zone group and backup project, with a 0.8-hour platform runtime cap
(6.4 GPU-hours), 2,760-second outer timeout and 2,650-second runner deadline.
The intended order is a 64-update 640p reference, 96-update 640p candidate,
96-update 896p candidate, then a 64-update 896p reference only if bounded time
remains. Failure stops the screen; no automatic retry or final acceptance claim.
The H200 compiler cache from V82 is reused; new batch shapes may still compile.
Profiler capture is disabled to avoid its export spillover. Report cold subprocess
time, all-rank warm timing, sample-normalized costs and actual batches separately.

The trainer snapshot stays V78. Only the standalone benchmark harness gains
single-stage plan/accumulation overrides. Focused harness/sizing tests: 33 passed.
Screen source/launcher/candidate-plan pins SHA-256:
`df30e8e4c4eee3045963d211ec402095904b6eef95b3f020ae6b5032c98222bc`.
V85/V86 are now closed; V87's reservation and the remaining allowance are stated
above. No reservation authorizes consuming the hero budget if final validation
needs more time.

The six-working-hour threshold has been established and all three final
opportunities have been evaluated. No additional promising lever was identified.
That closes the 4090 optimization selection. The approved hardware migration
requires fresh measurements and execution-plan qualification. Do not launch
the hero run in this pass.

## 1. Frozen execution candidate

The scientific policy remains [hero_recipe.md](cuda_hero_recipe_0924.md): 532,706,812
parameters, 256p → 640p → 896p, 75:20:5 of global updates,
continuous caption curriculum, fixed mixtures and loss weighting. 1024p is
dropped. BF16 activations and FP32 parameters, gradients, EMA and communication
are unchanged. No activation checkpointing, quantization or model-size change.
Accumulation 1/5/7 is the 4090 reference; re-screen it with bucket sizes on the
selected H100/H200 hardware while preserving the effective-batch targets.

[The hero launcher](../jobs/hero_stage.sh) selects dynamic block compilation,
matmul autotuning at all resolutions, disabled DDP compiler splitting, hoisted
double-stream RoPE, real-valued RoPE, native varlen flash attention and compiled
square Newton–Schulz iterations in Muon. All attention queries remain; only
invalid padded K/V keys are removed. Muon conversion, norm reduction and
normalization remain eager; rectangular paths and optimizer policy are unchanged.

Health snapshots stay on GPU. Periodic allocator cleanup is disabled;
evaluation's explicit cleanup remains. Compiler workers are capped at two per
rank, OpenMP at one thread, and GEMM candidates at ATEN/TRITON. Optional
CPU-wall/breakdown and per-micro shape logging are off. The cleanup-on benchmark
explicitly requests cadence 100 instead of inheriting the zero-cleanup hero
config. Reference paths remain available for diagnostics.

These are selected settings, **not yet accepted eight-GPU rates or memory limits**.

## 2. Completed final opportunity comparisons

All results here are lower-rank selection evidence using the real trainer,
fixed stage-midpoint caption policy and near-initial training. Matching recorded
shapes, sample counts and progression is not bitwise trajectory or quality
equivalence. Warm windows and full subprocess times have different meanings.

| Change | Physical GPUs | Reference → candidate | Decision |
|---|---:|---|---|
| Remove periodic cleanup | 2 | Update 100: 3.14248 → 1.29659 s; update 200: 3.27350 → 1.27127 s | Select |
| GPU health snapshots | 2 | Health update 250: 3.52611 → 1.40972 s | Select |
| 256p autotuning | 2 | Warm updates 51–199: 1.32176 → 1.29891 s | Select |
| 640p autotuning | 2 | Warm updates 51–128: 4.80127 → 4.58938 s; repeated reference 4.80034 s | Select |
| 896p autotuning | 1 | Warm updates 51–128: 7.05191 → 6.80480 s | Select |

### Cleanup and health

The cleanup pair completed 256 finite updates on both ranks. Ordinary windows
51–99 and 101–199 were essentially unchanged: 1.31800/1.32685 seconds with
cleanup versus 1.31876/1.32517 without. The two boundary savings average
1.92406 seconds per invocation, approximately 19.24 ms/update at cadence 100.
Maximum paired loss difference was 0.00535214.

Full subprocess time was 689.77/534.14 seconds, but compiler-cache/order and
late-shape compilation confound that larger difference; it is not the cleanup-
only gain. The user identified cleanup as an unsuccessful legacy diagnostic
for spikes subsequently explained by long captions. It has no remaining
historical justification, but its removal still needs high-resolution/eight-GPU
shape-transition, fragmentation and tail-memory checks.

The CPU/GPU-health pair also completed 256 finite updates per rank without
cleanup. Ordinary updates 101–199 stayed at 1.32517/1.32527 seconds. The single
health event saved 2.11639 seconds, approximately 8.47 ms/update at cadence 250.
Maximum paired loss difference was 0.00834382. Full times 534.14/487.42 seconds
also contain compilation effects, not just the health benefit.

A separate component comparison measured about 961.38 → 68.26 ms, checked
snapshot/metric agreement and returned post-operation allocation to baseline.
Two-rank 256p stress subsequently exercised GPU snapshots every update and
verified their release before the next forward. Final high-resolution/eight-rank
memory and production-cadence checks remain required.

### Autotuning: warm gain is not short-run gain

| Resolution | Reference whole subprocess | Autotuned whole subprocess | Limitation |
|---|---:|---:|---|
| 256p | 485.78 s | 918.14 s | Warmed-cache continuation after timeout; late shapes still compiled |
| 640p | 919.04 s; repeat 783.24 s | 1,712.42 s | Completed A/B/A; fresh stage cache, short-run cost worse |
| 896p | 1,013.85 s | 1,067.10 s | One-GPU warmed-cache fallback, not an independent cold A/B/A |

Principal warm-throughput gains are approximately 1.76%, 4.62% and 3.63%.
They must pay for complete cold geometry compilation and retained memory over
the eventual horizon. Do not use lower-rank timings as eight-GPU cost inputs.

256p completed 256 updates/case on both ranks; maximum paired loss difference
was 0.00936794. Peak allocated bytes were 46,805,526,528 / 46,714,104,832
(reference/autotuned). 640p completed all three 128-update cases on both ranks;
autotuned/reference-repeat maximum loss differences against the first reference
were 0.00474715/0.00586951. Peak allocated bytes were
44,384,477,696 / 44,304,823,808 / 44,384,477,696.

896p completed both 128-update cases on one GPU; maximum paired loss difference
was 0.00402510. Reference/autotuned windows were 7.05615/6.81473 seconds
(51–99), 7.04253/6.78546 (101–128) and 7.03655/6.79143 (116–128).
Peak allocation was 41,639,671,296 / 41,561,586,176 bytes and reservation
43,023,073,280 / 43,274,731,520. These are observed-case peaks, not boundary
or eight-rank memory acceptance.

The original two-rank 896p reference completed 128 updates. Its autotuned case
was preempted after 75 finite updates, and the continuation after 74. Explicit
platform eviction events identify both failures. Another replacement stayed
queued for approximately 29 minutes and was stopped before the one-GPU fallback
was submitted. The fallback succeeded at 13:06:56 UTC on September 15.
Partial trials are retained, not counted as complete pairs. No automatic retry
or overlapping replacement was launched.

## 3. Other measured decisions

| Method | Evidence | Disposition |
|---|---|---|
| RoPE hoisting | Two-rank whole subprocess 627.25 → 353.58 s; late warm window 1.31506 → 1.31977 s | Startup benefit, not warm-rate gain |
| Native attention / real RoPE | Component output/gradient checks and four-rank high-resolution gains | Selected; final eight-rank proof pending |
| Muon square-iteration compilation | 35 exact finite complete optimizer updates; about 195.59 → 182.77 ms | Selected; normalization stays eager |
| Foreach gradient/EMA operations | Division about 4.98 versus 5.03 ms; EMA 7.76 versus 13.63 ms | Rejected |
| DDP gradient bucket views | Component memory benefit did not survive full training; no useful speed gain | Rejected |
| Force NCCL Simple | Automatic/Simple around 134 ms, no meaningful gain | Keep automatic selection |
| Skip repeated buffer broadcasts | About 0.15% component gain; exact gradient verification interrupted by preemption | Not adopted |
| BF16 communication | Faster isolated hook but larger temporary memory and Muon update error | Not adopted; keep FP32 |
| FFN padding / custom kernels | Padding component did not justify a production change; precision-matched autotuning offered a measured alternative | No dimension change; no custom kernels or TileLang |

Broader Muon compilation initially diverged after ten exact updates. The
corrected version compiles only the square iteration loop; the later 35-update
test passed. An earlier nonfinite block diagnostic affected both the efficient
reference and candidate gradients. Follow-ups did not reproduce it; its cause
remains unidentified and is not described as fixed. Finite guards and final
GPU validation remain mandatory.

A real two-rank FP32/BF16 communication-hook comparison measured 141.19 →
80.05 ms, but increased temporary allocation by about 2.33 GB. A separate
one-GPU test used the same captured DiT gradients for simulated reductions:

| Simulated ranks | Gradient relative L2 error | Muon update error | AdamW update error |
|---|---:|---:|---:|
| 2 | 0.23079% | 6.51707% | 1.32905% |
| 8 | 0.31714% | 8.58075% | 2.09456% |

Synthetic inputs, early smoke weights and fresh optimizer states do not predict
quality or actual eight-rank NCCL ordering. They do show why small gradient
error does not establish small Muon update error. Reopening compression requires
explicit numerical/quality-risk acceptance, not just faster timing.

## 4. Profiler evidence and its limits

Earlier three-update traces assigned attention 5.745 of 17.807 seconds at
640p and 14.093 of 29.639 seconds at 896p. The memory-efficient attention
backward kernel alone occupied approximately 24%/35% of those GPU spans,
justifying backend/layout investigation. GEMM and pointwise costs were measured
separately before testing autotuning, RoPE and optimizer changes.

The completed four-rank composite improved warmed 640p/896p times
5.88309 → 4.73168 and 10.39480 → 7.33092 seconds. Three-update attention work
fell 5.8408 → 2.6432 and 14.4769 → 6.2250 seconds, while GEMM work remained
broadly unchanged. This composite included foreach operations later rejected;
it is not the final frozen candidate.

An earlier actual eight-rank composite was preempted after both 256p cases
and the 640p reference. Its 256p warm window improved 1.37281 → 1.31303
seconds, but whole subprocess time worsened 340.57 → 366.65 seconds.
Realized batch mean 717.55 differed from planned 652.28, illustrating short-
queue initialization bias. The 640p candidate and all 896p cases did not run.

In its three-update 256p trace, exposed NCCL was approximately 0.338 → 0.327 s;
communication duration sums were 1.989 → 1.974 s and overlap compute. The
640p reference exposed about 0.833 s on rank zero, with all-rank values
0.323–1.099 s. Earlier actual eight-GPU FP32 all-reduce for about 2.13 GB took
196.09 ms as one tensor and 200.03 ms in 82 chunks. The ring-normalized
17.71-GiB/s figure and RING_LL kernel names do not prove algorithm/protocol.

Earlier rank-zero data-fetch means were about 0.9–4.4 ms/micro in inspected
windows, not proof about all-rank tails or cold startup. A roughly 460-ms host
encoding interval coexisted with about 24 ms of GPU text-side work because host
calls waited for prior asynchronous GPU work. No encoder rewrite was justified
by those host timings. A diagnostic CPU residual's double subtraction was
corrected as reporting, not booked as a speedup.

**Hardware saturation is unproven.** High busy percentages, “time accessing
memory,” VRAM occupancy and a rejected candidate are insufficient evidence.
48-GB capacity does not establish compute versus bandwidth binding. Nsight
Compute counters were denied with ERR_NVGPUCTRPERM; no privileged counter or
clock change was attempted.

## 5. Correctness and final validation

Already demonstrated, with limited scope:

- Local regression: **762 passed, one skipped, 28 warnings**, 41.33 seconds.
  Existing DataLoader/Gloo tests need host IPC/loopback access and bounded CPU
  threads. Sandboxed attempts stalled and were interrupted, not counted as
  passing. Local Torch is not the target CUDA runtime.
- Coordinated finite guards reject rank-local nonfinite loss or pre-clip
  gradient norm before either optimizer updates. CPU distributed tests and
  earlier eight-GPU primitive tests passed; final combined execution is pending.
- Two-rank stage rehearsal exercised T=40, warmup=5, 256p 0→30, replay 28→30,
  640p 30→38 and 896p 38→40 with accumulation 1/5/7. Live model, optimizers,
  schedulers, EMA and rank RNG restored exactly; replay row/caption identities,
  shapes and progression matched.
- Continued replay parameters were not bitwise identical. Same-input probes
  demonstrated native attention-backward nondeterminism with deterministic
  algorithms off, and exact gradients with it on. This does not attribute every
  replay difference to attention. Exact restoration and exact future trajectory
  are distinct contracts.
- Two-rank 256p stress visited 100 bucket/aspect cases twice at exact caption
  limits, dropout off, cleanup off and health snapshots every update. All 200
  updates/rank were finite and snapshots released before the next forward.
  Peak allocation/reservation was 46,865,912,832 / 49,037,705,216 bytes.
  Approximately 1.20-GiB estimated headroom used an external-memory sample,
  not a guaranteed transient minimum. Autotuning/high resolutions were not
  covered by that run.
- Earlier fresh-process 256p evaluation produced the panel, loss bands and a
  128-fake KID diagnostic. It does not prove resident-training memory or the
  final 2,000-fake KID workflow.

### Eight-GPU validation job

**infra-0915-acceptance-8g-v76** was created at **14:18:59 UTC, September 15**.
It started on September 16 at approximately 01:44 UTC on
`qb-prod-4090-gpu078`. The job finished FAILED at 07:00:37 UTC on September 16.
All three config-render phases and all nine rate phases exited successfully;
256p and 640p memory stress and their verifiers passed. The 896p stress phase
received only 1,408 seconds from the remaining whole-workload deadline and
exited 124 after 162/200 recorded updates on every rank. Logs show SIGTERM from
the timeout, not a CUDA OOM or nonfinite-update exception. This is a validation
time-allocation failure, not evidence that 896p cannot fit. Its verifier, CUDA
guard phase and stage-chain/evaluation phase were not reached. Preserve all
results; the incomplete 896p run does not establish full memory acceptance.
The local monitor's observation timeout was renewed without restarting the job.
This is progress,
not acceptance of the uninspected rate artifacts or remaining phases.
It requests one eight-GPU 4090 node,
110 CPUs, 800 GiB host memory and 16 GiB shared memory, requested priority 4
(platform-assigned 20 / NORMAL), no automatic retries. The user approved the
priority increase; V75 was stopped while still queued at 14:17:09 UTC, before
submitting this replacement. Its workload output and cache paths did not exist,
and source pins reverified, so V76 uses the unchanged V75 launcher, source and
artifact tag. No duplicate allocation or workload restart was launched.
Cap: 5.5 hours / 44 GPU-hours; outer command 19,600 seconds, inner
workload 19,000 seconds. No hero run is launched. Caps do not guarantee every
phase completes.

The acceptance run serialized 23 phases (one-off harness removed after
evidence archival):

1. Render exact versioned inputs without regenerating weights.
2. Measure all resolutions at three equal-width curriculum-slice midpoints.
   Stage-midpoint trials run 256 updates, including health cadence 250; early
   and late trials run 128 each. Cost slices equally despite unequal trial
   lengths. Midpoint runs include bounded three-update traces.
3. Exercise every bucket/aspect boundary twice at each resolution, exact
   caption limits, production accumulation and health snapshots every update.
4. Run eight-rank CUDA guard primitives.
5. Run actual stage transitions, same-stage resume, complete checkpoints,
   monitoring panels, loss probes and final KID with selected execution flags.

Evaluation runs after actual updates with optimizer, EMA and DDP state resident:
48 prompts, EMA/Euler-50/BF16/no-CFG; 512 loss samples per available caption band;
2,000 generated images and the full held-out real set for final KID. T=40 is
diagnostic only. Per-rank evaluation setup/runtime/memory and checkpoint
completion-barrier time are recorded separately from update timing.

Exit codes alone are not acceptance. Inspect rank counts, complete records,
finite values, restoration/numerics, realized exposure and batches, tail memory,
exact stops, artifacts and pins. Synthetic stress is not throughput/sampling
evidence. At the user's request, observation now sleeps for two hours between
job checks. The earlier five-minute job/resource polling loops are stopped;
the remote validation job is untouched. Do not submit duplicates or restart
on observation timeout.

## 6. Reproducible source and inputs

W denotes explicit ARTFLOW_ROOT. Frozen source is separate from the canonical
repository and must not be edited while queued/running. The selected image is
artflow-base:torch29-cu128, targeting Torch 2.9.1+cu128 / NCCL 2.27.5.
A mutable image tag alone is not an immutable runtime pin: preserve the actual
eight-GPU environment.

The snapshot includes trainer, benchmarks, tests, base/hero configs, three
input templates, bucket plans and the approved monitoring panel.
[render_hero_stage.py](../scripts/bench/render_hero_stage.py) only substitutes W.
All rendered configs matched measured remote configs as parsed TOML; all plans
and the panel matched referenced files byte for byte.

| Artifact | SHA-256 |
|---|---|
| Frozen code-v75.tar.gz | 01ea370f97af5864703bff3861839a44e38cff4ba83f96d546eca9c1eb426df8 |
| Eight-GPU launcher | 04f51f66382724f63c8ae7c7f1d65befe1f1ce8816dd33ccbf8137a080408976 |
| Source manifest: 265 source/input files plus archive and launcher | c0c4f8a1ac2a211fea7aeb9a822f57ef9cc5564964a5f4176febdddf63e0877e |
| Text-encoder/VAE manifest | 75540bf56154da55b6027906e2a14bf96b0a589ababb3dd800a11f0384673df5 |
| Full dataset-content manifest | 35ff7acd7f7b2449bc58c38fd8468c372c571bf616bf10357c2c5542be55eab3 |
| Inception weights / snapshot panel manifest | cf4ebbe2c0f8c03cb70c9398eb59d6426b9df0d5bdc4872ed920177f9006353d |
| Seven-manifest bundle, infra-v75-pins.tar.gz | 359cedb3f3474c6237fd8608e8978e7ec2e022c1dbd2310b71944b42ed56b5ce |

The full content scan covered 47 trees, 2,660 files and 1,246,687,199,919 bytes.
At approximately 13:22 UTC on September 15, model trees, 147-file dataset/
config/plan metadata, runtime-freeze file and panel manifests reverified.
This did not repeat the complete content scan. Inception weights were separately
pinned. Source and manifest archives were transferred and hash-verified.

Raw manifests/ledgers contain operational paths and stay outside public commits.
Public plans/hashes are linked in the hero recipe. Canonical remote source still
needs synchronization to the finally accepted snapshot before production.

### Retained evidence archives

| Evidence | SHA-256 |
|---|---|
| Cleanup pair | 448b9a8d27c50301d96f2238d8a2d0c730f7a7b331fe8471e34dad0fd5887b15 |
| CPU/GPU health pair | 04e14c07a375bca028358998d2faf88753ff4e5a5b5405d8dda80644977d9d7c |
| 256p autotuning continuation | a83577280554adb77a1d6f941ff5bfb85a365a3dada9ccc7e3f01b80e282f255 |
| 640p A/B/A | 3a6baa93ae643a261502e21a4a7e034b6e3d25b5ca0234cf1cfb1dcc52fdfe6b |
| One-GPU 896p pair | 059e4896c08048e8190facdc35b253d0280eb3d9fc2e66b19b7c266ce9728b37 |
| Four-rank high-resolution composite | 835adc10d534bc4cf257d241746e770996a0e9d82a29babf5a6692628a65c9f0 |
| Preempted eight-rank composite | b09de2c5d95715bb72aa857bbfc7586184fbd9d6449216d8d1260948597a700e |
| Two-rank state-chain rehearsal | 55da33b24c4b444695f83213ec127e4541ac7ca8821cfe9b3967054fd3328300 |
| 256p memory/snapshot stress | 8d45db7b3ea87b80d063b4b7c29418431966153e0f9c505b70db5af22b035cc3 |
| BF16 communication numerical follow-up | 6f7deaf0890d4b1ea9ad7da7377bd6daf3ad370f1c2480ee7bc7e1d64b39f6b6 |

The detailed journal remains in ignored local
infra-pass-journal-pre-validation.tar.gz, SHA-256
30b9a5c07d310d92569c9fbd43bbc30e409e8a1b215d461e9371a275f7eda57d.
This report replaces stale status statements, not failed-trial evidence or
checkpoints.

## 7. Time, budget and optimization exit

Work began September 14 at 10:41:49 UTC. The budget-decision pause
10:50:38–10:54:53 is excluded. A conservative working-time lower bound was
4.8222 hours through 23:10:48; continuing measurement/implementation/analysis
through September 15 00:21:50 added 1 hour 11 minutes 2 seconds:
**6.0061 working wall hours**. Queue-only waiting is excluded. This is a proven
lower bound, not a claim that all elapsed wall time was work.

The exhausted earlier Stage-4 allowance is separate. This pass now has **120
additional RTX 4090 GPU-hours** (96 originally, plus 24 approved September 16
for unfinished validation), separate from the hero budget (now 800 H100/H200
GPU-hours, not interchangeable with RTX 4090 hours).
Include failed allocations, provisioning, termination and allocated idle time
regardless of project billing. Rank fallback does not waive eight-rank evidence.

Closed accounting before the final job, September 15 at 13:07:41 UTC:

| View | GPU-hours |
|---|---:|
| Primary-project conservative bound | 12.06361 |
| Backup-project conservative bound | 37.04806 |
| Combined conservative bound | **49.11167** |
| Combined observed container intervals | **29.52639** |
| Conservative remaining allowance | **46.88833** |
| Final eight-GPU runtime reservation | 44.00000 |
| Headroom beyond reservation | **2.88833** |

Bounds include creation-to-finish time where allocation evidence is missing,
including 0.95167 GPU-hours for the stopped queued two-GPU comparison; they are
not invoices. Observed container intervals omit provisioning/grace and are not
all-in spend. The successful final one-GPU pair contributed 0.58139 bounded /
0.57889 observed GPU-hours.

The priority replacement adds a queue-only ledger entry: V75's creation-to-stop
bound is 5.85778 GPU-hours, but platform wait reports running time 0s, repeated
FailedScheduling continued until immediately before stop, and neither workload
output nor cache was created. The raw ledger retains this bound; it is excluded
from the allocation reservation calculation, not claimed as GPU spend. Thus the
44-GPU-hour replacement reservation does not duplicate an allocated V75 run.
At 14:19:50 UTC the backup ledger's raw bound was 42.97500 GPU-hours, including
V75 and 0.06917 queue-inclusive hours for the open V76; observed closed container
intervals remained 26.88667. These are different accounting views, not invoices.

The queued final job is additional, not closed spend. Refresh its scheduling/
finish evidence; do not double-count reservation and actual runtime or treat
missing events as zero cost. Optimization exits on six hours plus disposition
of all visible levers, **not** on a saturation claim. Mandatory validation and
handoff remain.

## 8. Hero cost and remaining handoff

### September 16 eight-rank observations (not final costing)

All nine rate trials have eight aligned rank records and finite recorded losses:
256 updates per midpoint trial and 128 per early/late trial. Numeric inspection
uses the slowest recorded rank interval at each update. After the first 50
updates, excluding explicitly profiled updates:

| Stage | Early seconds/update | Mid seconds/update | Late seconds/update |
|---|---:|---:|---:|
| 256p | 1.26948 | 5.42236 | 4.17476 |
| 640p | 4.65162 | 4.73349 | 4.68617 |
| 896p | 7.11252 | 7.16568 | 7.14535 |

These are not clean steady-state estimates: 256p midpoint/late retain large
timing spikes well beyond update 50, interspersed with approximately 1.3-second
windows. Separate compiler/profiler/startup effects using the retained logs and
traces before any extrapolation; do not silently discard slow windows. Realized
post-50 mean global batches were 635.9–704.3, 584.5–587.4 and 426.6–426.9 across
the respective stages. These short-window observations require exposure/queue
analysis, not a claim that target effective batches were achieved.

The September 16 07:41 UTC ledger has no open allocations. V76 used 42.33333
scheduled-to-finish GPU-hours. Combined raw primary/backup bound is 97.30278,
including the documented 5.85778 queue-only V75 bound. Excluding only that
queue-only entry gives 91.44500 against the 96-hour allowance, leaving
4.55500 GPU-hours conservatively before the approved extension. The user approved
24 more GPU-hours, bringing remaining conservative allowance to 28.55500.
`infra-0916-completion-8g-v77` was created at 07:48:56 UTC, initially QUEUING,
requested priority 4 (assigned 20 / NORMAL), with a 3.4-hour / 27.2-GPU-hour cap,
an 11,900-second workload deadline and a 12,100-second outer timeout. It repeats
the interrupted 896p stress from fresh state, then guards and the full
state/evaluation chain; no completed rate trials are repeated. The source-v75
trainer and compiler cache are reused unchanged. Only the orchestration runner
adds explicit completion-phase selection, tested locally (three plan tests
passed) and dry-run verified remotely. All three configs render into fresh paths.
Runner SHA-256: 394b15c167fd313cff7abd3a5d9b9a82f03e4440e2e04cd644edb07a25428102.
Launcher SHA-256: 92677ad3456702885d9fac8f71f63aa03ec2a094bbd749715474f2e12284a468.

V77 terminated on September 17 at 00:58:53 UTC, consuming 11.08 scheduled-to-
finish GPU-hours. The 896p 200-update stress and verifier, CUDA guard primitives,
and all four state-chain training subprocesses completed. The final verifier
failed on KID sample counts: 896p has 1,038 held-out rows, and the evaluator
silently capped 2,000 requested fakes to 1,038. KID was finite (mean 0.64520890,
std 0.01062597), but this did not meet the required generated count. The verifier
also incorrectly assumed at least 2,000 real rows. Earlier passing checks are
retained; no inference-quality claim follows from this diagnostic checkpoint.

The correction generates the requested number using deterministic cycling of
conditioning rows with distinct global fake-ID seeds. It retains every real
row and checks the exact resolution-specific real count; it does not duplicate
the real set or change training data. Thirty focused CPU tests passed, including
more fakes than real rows, multi-rank/empty shards, exact fake-seed coverage and
rejection of count mismatches. Only the KID evaluator and state-chain verifier
change in the separate source-v78 snapshot.

Targeted job `infra-0917-evalfix-8g-v78` restores V77's step-38 checkpoint, performs
two 896p updates, and validates resident-state panel/loss/2,000-fake KID, exact
restoration and the step-40 checkpoint. It has a 1.5-hour / 12-GPU-hour cap and
5,200-second command timeout. Before this allocation, conservative adjusted
spend is 102.525 GPU-hours, leaving 17.475 of the approved 120. No memory/rate
trials are repeated. Source/config/launcher manifest SHA-256:
00c201429fb7851553dd046692e93128ab43c8b1989f3629e9990a7dc7c6c4e7.
V78 succeeded September 17 at 20:52:05 UTC. Its platform log contains the
explicit `EVALFIX_VERIFIED` marker after every launcher assertion and exit code
zero: 2,000 fakes, full real set, eight-rank resident-state evaluation, exact
restoration and global endpoint. This closes the targeted correction on RTX
4090, not on H200. Its scheduled-to-finish bound is 4.43333 RTX 4090 GPU-hours;
adjusted cumulative reference-pass spend is 106.95833 of 120, leaving 13.04167.
Seventeen newly generated tracking files were preserved outside the source;
all originally pinned files remained unchanged and the original manifest
verified again. No GPU time was spent re-running this verification.
Artifact inspection confirms KID used 1,038 real / 2,000 generated images
(mean 0.64387852, std 0.01008584), all eight restoration reports are exact,
and the checkpoint completion record has global step/max_steps 40/40,
world size eight, two schedulers and EMA. These are execution checks on a
diagnostic checkpoint, not quality evidence. Slowest-rank evaluation durations
were 22.825 s for loss setup, 243.753/243.532 s for the two loss probes,
84.449 s for the panel and 1,089.191 s for final KID. They are RTX 4090
reference overheads; do not enter them as measured H200 costs.

**No final eight-rank cost input has been populated.** Lower-rank times and the
4090 eight-rank measurements do not establish the revised 400k/800-H100-or-H200-hour gate.

    weighted_step_seconds = 0.75*t256 + 0.20*t640 + 0.05*t896
    GPU_hours(T) = 8*T*weighted_step_seconds/3600 + overhead_GPU_hours(T)

At illustrative T=400k, the all-in equivalent must be at most 0.900 seconds/
update. Startup/compilation, evaluation, complete checkpoint writes, allocated
idle and bounded recovery must fit inside that number.

[cost_hero_run.py](../../scripts/bench/cost_hero_run.py), invoked with explicit
`--budget-gpu-hours 800` and selected-hardware measurements, accounts for exact integer
endpoints, checkpoints every 2k, grids every 10k plus endpoint/incoming/+2k
triggers, loss probes every 500 plus entry baselines, and final KID. Overlapping
triggers count once. Integer horizon search does not assume local monotonicity
under rounded boundaries and overlapping triggers.

Still required before Stage 5 readiness:

- Actual eight-rank rates/traces for three equally weighted curriculum slices
  at every resolution, realized batches/exposure and sampling/loss semantics.
  Diagnose short-queue bias, input tails and exposed communication.
- Boundary-memory acceptance, exact global-T stops, complete loadable checkpoints,
  same-stage resume and both transitions with model/optimizer/scheduler/EMA/RNG
  and continuous curriculum. Reject incomplete newest checkpoints rather than
  silently selecting an older one.
- Coordinated nonfinite stopping and resident-training panel/loss/final KID
  workflow. No numerical capability or inference-latency gate is added.
- Measured startup/compilation excess and full-allocation checkpoint/grid/loss/
  KID costs, including waiting ranks. Do not silently exclude late compilation,
  health events or unexplained stalls to improve rates.
- Explicit recovery/allocated-idle allowances without double-counting; nominal
  and conservative GPU-hours/1k updates, steps per wall/GPU-hour and feasible
  budget-supported horizons. Queue wall time is not allocated wall time.
- Final source/runtime/input pins, canonical-source synchronization, validated
  launch/resume commands, writer locking, retention, storage and recovery checks.
- Refreshed time/spend ledger and reconciled recipe/Stage-4 checklist. Every
  costing horizon remains illustrative; **the user selects production T**.

Earlier checkpoints were about 6,461,975,336 bytes each: 200 writes would be about
1.29 TB before other artifacts. Retain checkpoints unless the user authorizes a
different policy. Preparatory shared-filesystem free space was
28,435,443,351,552 bytes, not a verified project quota or final reservation.
Refresh capacity and validate retention/failure handling in the final handoff.

**Pass completion means Stage 5 infrastructure readiness pending the user's T,
not merely a frozen candidate, passing local tests or a completed profiling job.**

## 2026-09-19 eval_interval 2500 + 手动重启验证

- `configs/hero.toml`: `eval_interval` 10000 → 2500（grid 触发为绝对步数 `step % interval == 0`，resume 安全）。已同步 `$W/repo-0918/configs/hero.toml`（md5 67a3341faa034b4b005cb2d6213cc1c2）。
- 通过 `inspire job shell` + `pkill -f src.train` 触发容器 exit 1，auto-fault-tolerance 自动拉起 worker-6（01:00:53，qb-prod-4090-gpu085），从 checkpoint 2000 正常恢复（EMA/scheduler/RNG 还原）。**这同时实证了抢占场景的全自动续训链路**。
- 成本：丢失 ~150 步未 checkpoint 训练 + 一次 triton autotune 重编译（约 20 分钟）。

## 2026-09-19 step 2500 grid 崩溃：assets/ 未同步进 repo-0918

- 现象：eval_interval 改为 2500 后，worker-6 在 step 2500 首次触发 grid eval 时崩溃（rank 2 exit 1）。根因：`configs/hero.toml` 的 `prompts_file = assets/eval/hero_monitor_v1.jsonl` 是 repo 相对路径，而 repo-0918 快照同步时漏了 `assets/` 目录。
- 修复：本地 `assets/`（仅 eval/prompts_v1.jsonl 与 eval/hero_monitor_v1.jsonl，104K）打包 scp 至 `$W/repo-0918/` 解包，md5 双侧一致（13a4a50bae53a0bb165160237310f58d / 7171f32f9f441aacb28d1fa1ad0097bf）。
- 排查确认 configs 与 launcher 中只有这两个 repo 相对路径引用，无其他缺失资产；eval-loss probe、vae、text encoder 均走 $ARTFLOW_ROOT 绝对路径，启动期已验证正常。
- 当前：worker-7 处于 FailedScheduling 排队（priority 1 等卡），文件已就位，调度成功后从 checkpoint 2000 恢复即可越过 2500。后台 watcher（bash-q8yqtw0s）每 10 分钟检查 grid_step_002500 产出，最长 24h。
- 教训：repo 快照同步清单必须包含 `assets/`；改 eval_interval 这类"首次触发点才暴露依赖"的改动，应在改动后立即在 smoke 中触发一次对应事件。

## 2026-09-19 EMA 监控假象事件（重要运维教训）

- 现象：hero-256p 的 eval/loss（probe）在 step ~7500 触底 1.58 后回升至 1.87@10k；grid 在 2500/5000/7500/10000 几乎不变、纯噪声、跨 prompt 趋同。train/loss 同期完全正常（0.88，与健康 run 一致）。
- 排查排除项：grid 的 "zh / short / seed=" 前缀是展示标签（prompt_grid.py:310），不进 text encoder；数据无错位（256p precompute 与健康 s4 run 同批）；采样重复非因（m-15k 同样 step 4000 起 repeat_rate=1.0 且 eval 持续下降）。
- 根因：probe 与 grid 均评 **EMA 权重**。hero 用 ema_decay=0.9999（s4 用 0.999），init 残留 0.9999^t：step 10k 时仍有 36.8% 随机初始化权重；且 lr=0.02 长期恒定（cosine 周期 480k），权重高速移动使 EMA 持续偏离路径（ema_rel_distance 0.72 vs m-15k 0.09），平均值落在 loss 曲面路径外侧导致 probe 非单调。
- 决定性验证（hero-diag-liveprobe，1 卡，scripts/diag/live_probe.py）：checkpoint_step_010000 的 **live 权重 probe eval/loss=0.90014**（band: le128=0.924, 129_256=0.924, 257_512=0.655, 513_1024=0.666；t015=0.835, t040=0.751, t065=0.903, t090=1.110），与健康 m-15k 同步数 0.889 相当。**训练健康，无止损必要**。
- 后续：probe/grid 需改为评 live 权重（或 live+EMA 双路），待 grid 对比图出来后与用户确认再动训练代码；EMA decay 是否在长跑后期收敛到 live，到 640p 阶段前再评估。
- 验证补完：离线 EMA probe=1.86884 与训练日志 1.86865 一致（EMA 评估链路无 bug）；live/EMA 同 seed grid 对比图在 $W/runs/hero-diag-liveprobe/（live 为结构完整静物画，EMA 为纯噪声）。用户决定（2026-09-19）：不改训练代码，继续评 EMA；cron 巡检已改用 train/loss + grad_norm 作健康判据。

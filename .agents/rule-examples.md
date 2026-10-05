# Rule calibration examples

Dated incidents that calibrated the rules in [AGENTS.md](../AGENTS.md). These are evidence for future supersession audits, not authority; the rules themselves are the only current standard. Recorded 2026-10-05 from the rules interview.

## 1. Discussion ends only when the user ends it

A discuss-first task received a second-round revision and counter-arguments; the agent read the reply as "discussion complete" and implemented from the half-baked design, producing code later deleted in the cleanup. The reply was evidence the discussion was deepening, not concluding.

## 2. No fix without a mechanism

A posttrain fix was declared successful when the loss went from exploding to under 10; the generated grids were still broken. Also observed: investigations ending with several sacrifice-options and "you decide" — deference used to stop an unfinished investigation.

## 3. Green is not a correctness signal

Most frequent species: pytest all-green reported as "the change is correct". Also observed: import succeeds or the script runs → the feature is claimed to work; eval/loss declining → the model is claimed to be improving. The true correctness signals (grids, benchmark scores) were never fetched.

## 4. Existing code is not evidence of intent

Stated as a principle during the 2026-10 cleanup: code and docs present in the tree may be deliberate decisions, missed leftovers, or a previous agent's output. Consequence drawn at the time: no compatibility interfaces, resolutely none.

## 5. One-off code is deleted once its evidence is extracted

The cleanup's most-deleted species: one-off engineering scripts kept "just in case", including untracked wrapper variants accumulating in scratch directories. Code is disposable in agent-era development; the durable asset is the evidence recorded in notes/.

## 6. Test expectations must be independent

A test imported a constant from src and asserted `CONSTANT == 10`; a trial decision ("try 30 epochs first") was written into a test. Positive calibration: `tests/test_checkpoint_retention.py` passes `keep_last` as an input and asserts the mechanism — the (N+1)-th newest is pruned, N survive, and damaged replacements block pruning.

## 7. tests/ verifies mechanism, never effect

Calibration: `tests/test_posttrain_loop_cpu.py` was judged in-bounds — it verifies loop mechanics (rollout, buffer, save/resume) with externals stubbed, not training quality. The line: effects are unverifiable, mechanics are verifiable.

## 8. The repo is written for a reader who has only HEAD

Tracked notes referencing "1005决策", "规则 7", "用户要求", "0927 讨论" — labels and citations unresolvable without the session. Internal facts (credential storage locations) appearing in tracked docs. `scripts/ascend/` accumulating pretrain and monitor scripts until the name predicted nothing. `src/evaluation/concept_benchmark.py`: a one-off, script-shaped facility file in a library location — deleted in the cleanup.

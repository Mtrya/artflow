# AGENTS.md

ArtFlow: flow-matching text-to-image DiT. This file is **behavioral guidance
for agents** — durable ways of working, not repo status. Run state and stage
progress live in `notes/`; consult them before acting, never from here.

## How to do infra optimization

- Measure before and after, on the real workload, one variable at a time.
  Microbenchmarks justify a probe, never a recipe change.
- Adopt a lever only when its measured end-to-end gain is material; revert
  levers that aimed at the wrong segment (profile the step breakdown first:
  data / forward / backward / sync / optimizer).
- Platform-specific code must be opt-in flags, default-off, with zero cost
  and zero behavior change on the main (NVIDIA) path.
- One-off probes are deleted once their evidence is written into `notes/`.
  The note is the archive; never keep a script "just in case".
- Don't ship speculative generality: three similar lines beat a premature
  abstraction, but a proven bottleneck fix ships clean.

## How to debug

- Get the real traceback before theorizing. Training stdout may not reach
  platform logs — know where the actual log lives and how to read it; make
  crashes self-report (first traceback, error files) instead of eyeballing
  teardown noise.
- Fix root causes, not symptoms; if a failure is probabilistic, treat past
  successes as luck until the mechanism is proven, not as evidence of health.
- Change one thing per iteration and record what each attempt showed.
  A fix whose efficacy is unverified is a hypothesis — say so when reporting.
- Separate "model bug" from "platform bug" by bisecting the environment
  (same code on another node/card/OS) before touching model code.

## How to write documentation

- `notes/` is the project memory: record decisions with their reasoning and
  the measured evidence, dated; update living docs when a decision changes
  rather than appending contradictions.
- When a stage closes, move its records to `notes/archive/` and fix links.
- Comments and docstrings describe the code as it is now — sweep stale ones
  in the same change that makes them stale.
- No secrets in any artifact. Keys come from env vars or platform secret
  files; internal-only operational detail goes in gitignored docs.

## How to keep the repo clean

- Delete dead code and its tests in the same change; don't leave superseded
  paths flag-disabled "for later".
- Commit in small logical chunks, English messages stating what and why;
  the test suite (`.venv/bin/python -m pytest tests/ -q`) is green before
  committing; add tests for new src code.
- `jobs/` is untracked local scratch — never rely on its contents; repeatable
  tooling belongs in `scripts/` under the right area subdirectory.
- Large data operations run on the compute platform, not the local machine;
  local disk is scarce.

## When to consult which doc

- `notes/redesign_plan.md` — master plan, stage definitions, frozen decisions.
  Check before proposing anything that touches a frozen item.
- `notes/hero_recipe.md` — the signed-off training recipe and launch records.
- `notes/infra_pass.md` — measured monitoring/throughput facts; check before
  re-deriving known baselines or re-diagnosing known phenomena.
- `INSPIRE.md` — platform operations (Inspire): job submission, storage
  layout, quotas, gotchas. Read before any platform action.
- `notes/ascend_probe_0921.md` — Ascend bring-up chronicle; check before
  touching NPU code or re-running a concluded probe.
- `git log` — recent decisions and their rationale; check before undoing or
  re-adding something.

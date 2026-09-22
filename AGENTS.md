# AGENTS.md

ArtFlow: flow-matching text-to-image DiT (bilingual zh/en, multi-resolution
curriculum). This file holds **durable** conventions only — current run state,
stage progress, and monitoring setup live in `notes/`, not here.

## Layout

- `src/pretrain/` — training loop, sampler, telemetry, precompute
- `src/posttrain/` — last-mile post-training (DMD2 / DiffusionNFT / rewards)
- `src/models/`, `src/flow/`, `src/dataset/`, `src/evaluation/`, `src/pipeline/`
- `scripts/` — repeatable tooling only, grouped by area: `bench/` (bucket-plan
  and screening pipeline), `data/` (harvest/publish), `caption/` (VLM captioning
  production flow), `posttrain/`, `ascend/` (Ascend platform operations)
- `configs/`, `bucket_plans/`, `tests/`, `notes/` (living docs at top level,
  finished stage records in `notes/archive/`)
- `jobs/` — local scratch launchers, gitignored; never rely on its contents

## Conventions

- Tests: `.venv/bin/python -m pytest tests/ -q`. Add tests for new src code;
  every test must pass before committing.
- One-off probes and benchmark harnesses are deleted once their evidence is
  written into `notes/`. If a result matters, the note is the archive — do not
  keep the script "just in case".
- Keep dead code out: when a module/path is superseded, delete it and its
  tests in the same change rather than leaving it flag-disabled.
- Platform-specific code (e.g. Ascend 910B paths in `src/pretrain/train.py`,
  `src/models/`) must be opt-in flags, default-off, zero cost on the NVIDIA
  path.
- Inspire platform: remote jobs get unique names (retries multiply duplicate
  rows); large data operations run on the platform, not the local machine;
  platform-specific operational knowledge goes in `INSPIRE.md`.
- Don't commit secrets (tokens, netrc). Keys come from environment variables
  or platform secret files.
- Commit in small logical chunks; messages state what and why in English.
- Documentation: living documents (plans, recipes, probe chronicles) are kept
  current as decisions change; once a stage closes, its record moves to
  `notes/archive/` and links are updated. Comments and docstrings must
  describe the code as it is now — sweep stale references in the same change.

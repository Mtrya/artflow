"""Bounded H200 candidate screen, not memory-tail or final hero acceptance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.bench.infra_acceptance import BENCH_FLAGS


def plan(root, tag, *, candidate_only=False):
    phases = []
    for stage, label, accumulation, steps, limit, optional in (
        ("640p", "reference", 5, 64, 350, False),
        ("640p", "candidate", 2, 96, 1050, False),
        ("896p", "candidate", 3, 96, 1050, False),
        ("896p", "reference", 7, 64, 450, True),
    ):
        if candidate_only:
            if label != 'candidate':
                continue
            limit = 1170
        cmd = [sys.executable, str(root / 'infra-baseline-v85.py'),
               '--root', str(root), '--tag', f'{tag}-{stage}-{label}',
               '--stage-config-dir', str(root / 'runs/infra-0917-hopper-train-8g-v80/configs'),
               '--stage', stage, '--ranks', '8', '--steps', str(steps),
               '--stage-timeout', str(limit-30), '--accumulation', str(accumulation),
               '--curriculum-fraction', '.5', '--record-metrics', '--trace-start', '-1',
               '--no-cpu-wall-profile', '--no-cache-clear', '--autotune-blocks', *BENCH_FLAGS]
        if label == 'candidate':
            cmd += ['--bucket-plan', str(root / f'bucket_plans/hero/hopper-proposal-v84/{stage}/plan.json')]
        phases.append(dict(name=f'{stage}-{label}', command=cmd,
                           timeout_seconds=limit, optional=optional))
    return phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--max-seconds', type=int, default=2650)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--candidate-only', action='store_true',
                        help='Use separately completed reference evidence; no same-node comparison')
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in ('', '.', '..') or args.max_seconds < 1:
        parser.error('single-component tag and positive deadline required')
    root = args.root.resolve()
    report = dict(phases=plan(root, args.tag, candidate_only=args.candidate_only),
                  candidate_only=args.candidate_only, complete=False, final_acceptance=False,
                  caveat='Different realized sample counts; report per-sample cost and exposure, '
                         'not only per-update speed. No tail-memory or production-rate claim.')
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    import torch
    if torch.cuda.device_count() != 8 or not all('H200' in torch.cuda.get_device_name(i) for i in range(8)):
        raise ValueError('screen requires eight H200 GPUs; cache/runtime are H200-specific')
    out = root/'runs'/args.tag
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.max_seconds
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1')

    def save():
        (out/'screen.json').write_text(json.dumps(report, indent=2)+'\n')

    save()
    for phase in report['phases']:
        remaining = int(deadline-time.monotonic())-30
        if phase['optional'] and remaining < phase['timeout_seconds']:
            phase['skipped'] = 'insufficient bounded time for optional same-node reference'
            save()
            continue
        if remaining <= 0:
            report['error'] = 'whole-screen deadline reached'
            save()
            return 1
        limit = min(remaining, phase['timeout_seconds'])
        phase.update(started_unix=time.time(), effective_timeout_seconds=limit)
        save()
        with (out/(phase['name']+'.log')).open('w') as log:
            result = subprocess.run(['timeout', '--signal=TERM', '--kill-after=30s',
                                     f'{limit}s', *phase['command']], env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        phase.update(ended_unix=time.time(), returncode=result.returncode)
        save()
        if result.returncode:
            return 1
    report['complete'] = True
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

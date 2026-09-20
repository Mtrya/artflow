"""Exercise the actual launcher lock with a harmless fake training command."""

import os
from pathlib import Path
import signal
import subprocess
import time


def test_launcher_rejects_concurrent_writer_and_releases_on_exit(tmp_path):
    root = tmp_path / "workspace"
    (root / "repo").mkdir(parents=True)
    (root / "jobs").mkdir()
    (root / "jobs/swanlab_login.sh").write_text("# No credentials needed in this test.\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake_python = binaries / "python3"
    fake_python.write_text(
        '#!/bin/sh\n: > "$ARTFLOW_ROOT/started"\nexec sleep 20\n'
    )
    fake_python.chmod(0o755)
    launcher = Path(__file__).resolve().parents[1] / "jobs/hero_stage.sh"
    env = dict(os.environ, ARTFLOW_ROOT=str(root), PATH=f"{binaries}:{os.environ['PATH']}")
    command = ["bash", str(launcher), "256p", "400000"]
    first = subprocess.Popen(command, env=env, start_new_session=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (root / "started").exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert (root / "started").exists(), first.communicate(timeout=1)
        second = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
        assert second.returncode == 1
        assert "another hero launcher holds the writer lock" in second.stderr
    finally:
        if first.poll() is None:
            os.killpg(first.pid, signal.SIGTERM)
        first.communicate(timeout=5)
    lock = root / "runs/hero-256p/.writer.lock"
    assert lock.exists()  # File persistence must not block recovery.
    released = subprocess.run(["flock", "-n", str(lock), "true"], timeout=5)
    assert released.returncode == 0

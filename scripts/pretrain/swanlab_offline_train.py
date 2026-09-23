"""Run training with timely SwanLab offline writes on a shared filesystem.

SwanLab 0.9.7 leaves its binary writer buffered until close. GPFS chooses a
4 MB Python buffer, hiding early metrics from the CPU relay. Flush at most
once per ten seconds of writes, including the initial run record.
"""

import runpy
from pathlib import Path
import sys
import threading
import time


def enable_periodic_flush(writer_type, interval=10.0, clock=time.monotonic):
    original = writer_type.write
    lock = threading.Lock()

    def write(self, data):
        # Training and the hardware monitor call the same SDK writer from
        # different threads. Header, payload, indices and flush must be atomic.
        with lock:
            original(self, data)
            now = clock()
            previous = getattr(self, "_artflow_last_flush", None)
            if previous is None or now - previous >= interval:
                self.ensure_flushed()
                self._artflow_last_flush = now

    writer_type.write = write


if __name__ == "__main__":
    import swanlab
    from swanlab.sdk.internal.core_python.store import DataStoreWriter

    if swanlab.__version__ != "0.9.7":
        raise RuntimeError("Offline writer flush is qualified for swanlab==0.9.7")
    enable_periodic_flush(DataStoreWriter)
    module = sys.argv.pop(1)
    # Match `python -m` resolution although torchrun invokes this file by path.
    sys.path.insert(0, str(Path.cwd()))
    runpy.run_module(module, run_name="__main__", alter_sys=True)

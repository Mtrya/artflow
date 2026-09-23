"""A log writer may be interrupted anywhere within a binary record."""

import pytest
from pathlib import Path
import subprocess
import sys
import struct
import threading
import time

from swanlab.proto.swanlab.record.v1.record_pb2 import Record
from swanlab.sdk.internal.core_python.store import DataStoreReader, DataStoreWriter

from scripts.pretrain.swanlab_live_relay import read_available
from scripts.pretrain.swanlab_offline_train import enable_periodic_flush


def test_training_wrapper_resolves_module_from_source_working_directory(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/pretrain/swanlab_offline_train.py"
    (tmp_path / "wrapped_train.py").write_text("import sys\nprint('ARGS', sys.argv[1:])\n")
    result = subprocess.run([sys.executable, str(script), "wrapped_train", "--config", "recipe.toml"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "ARGS ['--config', 'recipe.toml']" in result.stdout


def test_writer_flushes_initial_record_then_only_at_interval():
    class Writer:
        writes = 0
        flushes = 0

        def write(self, data):
            self.writes += 1

        def ensure_flushed(self):
            self.flushes += 1

    ticks = iter([100, 101, 109, 110])
    enable_periodic_flush(Writer, clock=lambda: next(ticks))
    writer = Writer()
    writer.write(b"start")
    assert writer.flushes == 1
    writer.write(b"a")
    writer.write(b"b")
    assert writer.flushes == 1 and writer.writes == 3
    writer.write(b"c")
    assert writer.flushes == 2 and writer.writes == 4


def test_concurrent_writer_records_remain_readable(tmp_path):
    class SlowWriter(DataStoreWriter):
        def _write_record(self, data, *args):
            time.sleep(.001)
            return super()._write_record(data, *args)

    enable_periodic_flush(SlowWriter, interval=0)
    path = tmp_path / "concurrent.swanlab"
    writer = SlowWriter()
    writer.open(path)
    barrier = threading.Barrier(4)

    def produce(rank):
        barrier.wait()
        for step in range(20):
            record = Record()
            record.scalar.key = f"rank/{rank}/" + "x" * (1000 * (step % 2))
            record.scalar.step = step
            writer.write(record.SerializeToString())

    threads = [threading.Thread(target=produce, args=(rank,)) for rank in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    writer.close()
    records, cursor = read_available(path, None, DataStoreReader, Record)
    assert len(records) == 80 and cursor == path.stat().st_size
    assert len({(r.scalar.key, r.scalar.step) for r in records}) == 80


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_interleaved_pair_recovers_without_mutating_source(tmp_path, reverse):
    path = tmp_path / "interleaved.swanlab"
    writer = DataStoreWriter()
    writer.open(path)
    for step in (1, 2):
        record = Record()
        record.scalar.key = "train/loss"
        record.scalar.step = step
        record.scalar.value.number = 1 / step
        writer.write(record.SerializeToString())
    writer.close()
    data = path.read_bytes()
    size_a = struct.unpack("<IHB", data[7:14])[1]
    a, head_b, b = data[14:14 + size_a], data[14 + size_a:21 + size_a], data[21 + size_a:]
    broken = data[:14] + head_b + (b + a if reverse else a + b)
    # Incomplete pairs must wait, even if the first payload already exists.
    path.write_bytes(broken[:-1])
    records, cursor = read_available(path, None, DataStoreReader, Record)
    assert records == [] and cursor == 7
    path.write_bytes(broken)
    records, cursor = read_available(path, cursor, DataStoreReader, Record)
    assert [(r.scalar.step, r.scalar.value.number) for r in records] == [(1, 1.0), (2, .5)]
    assert cursor == len(broken) and path.read_bytes() == broken
    # A changed payload with a mismatching CRC is never "repaired" or skipped.
    damaged = bytearray(broken)
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    records, cursor = read_available(path, None, DataStoreReader, Record)
    assert records == [] and cursor == 7


@pytest.mark.parametrize("header_bytes", [0, 3])
def test_unflushed_header_waits_for_writer(tmp_path, header_bytes):
    full = tmp_path / "full.swanlab"
    writer = DataStoreWriter()
    writer.open(full)
    record = Record()
    record.start.name = "h200-hero-256p"
    writer.write(record.SerializeToString())
    writer.close()
    path = tmp_path / "growing.swanlab"
    path.write_bytes(full.read_bytes()[:header_bytes])
    records, cursor = read_available(path, None, DataStoreReader, Record)
    assert records == [] and cursor is None
    path.write_bytes(full.read_bytes())
    records, _ = read_available(path, cursor, DataStoreReader, Record)
    assert records[0].start.name == "h200-hero-256p"


def test_partial_record_is_retried_without_losing_or_repeating_metrics(tmp_path):
    full = tmp_path / "full.swanlab"
    writer = DataStoreWriter()
    writer.open(full)
    for step in (1, 2):
        record = Record()
        record.scalar.key = "train/loss"
        record.scalar.step = step
        record.scalar.value.number = 1 / step
        writer.write(record.SerializeToString())
    writer.close()
    data = full.read_bytes()
    partial = tmp_path / "partial.swanlab"
    partial.write_bytes(data[:-3])
    records, cursor = read_available(partial, None, DataStoreReader, Record)
    assert [r.scalar.step for r in records] == [1]
    with partial.open("ab") as handle:
        handle.write(data[-3:])
    records, cursor = read_available(partial, cursor, DataStoreReader, Record)
    assert [r.scalar.step for r in records] == [2]
    records, again = read_available(partial, cursor, DataStoreReader, Record)
    assert records == [] and again == cursor


def test_fragmented_record_is_retried_from_its_start(tmp_path):
    full = tmp_path / "full.swanlab"
    writer = DataStoreWriter()
    writer.open(full)
    record = Record()
    record.start.description = "x" * 100000
    writer.write(record.SerializeToString())
    writer.close()
    data = full.read_bytes()
    path = tmp_path / "partial.swanlab"
    path.write_bytes(data[:-50000])
    records, cursor = read_available(path, None, DataStoreReader, Record)
    assert not records
    path.write_bytes(data)
    records, _ = read_available(path, cursor, DataStoreReader, Record)
    assert len(records) == 1 and records[0].start.description == "x" * 100000

"""A log writer may be interrupted anywhere within a binary record."""

import pytest
from pathlib import Path
import subprocess
import sys

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

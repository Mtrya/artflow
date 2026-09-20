import pytest

from scripts.bench.pin_artifacts import record, verify


def test_manifest_detects_same_size_content_change(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("aaa")
    manifest = record([config], [])
    verify(manifest)
    config.write_text("bbb")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify(manifest)


@pytest.mark.parametrize("change", ["add", "remove"])
def test_tree_membership_is_pinned(tmp_path, change):
    (tmp_path / "weights").write_bytes(b"a")
    manifest = record([], [tmp_path])
    verify(manifest)
    if change == "add":
        (tmp_path / "extra").write_bytes(b"b")
    else:
        (tmp_path / "weights").unlink()
    with pytest.raises(ValueError):
        verify(manifest)


def test_symlinked_files_are_hashed_but_directory_links_are_rejected(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    target = tmp_path / "target"
    target.write_bytes(b"weights")
    (tree / "weight").symlink_to(target)
    manifest = record([], [tree])
    verify(manifest)
    (tree / "directory").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        record([], [tree])


def test_empty_manifest_is_not_a_pin(tmp_path):
    with pytest.raises(ValueError):
        record([], [])
    with pytest.raises(ValueError):
        record([], [tmp_path])
    with pytest.raises(ValueError):
        verify(dict(version=1, files={}, trees={}))


def test_manifest_cannot_add_itself_to_a_pinned_tree(tmp_path, monkeypatch):
    import sys
    from scripts.bench.pin_artifacts import main

    monkeypatch.setattr(sys, "argv", ["pin_artifacts", "--tree", str(tmp_path),
                                     "--out", str(tmp_path / "manifest.json")])
    with pytest.raises(SystemExit):
        main()
    assert not (tmp_path / "manifest.json").exists()

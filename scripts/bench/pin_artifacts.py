"""Hash explicit files/trees and verify their identity before a training launch.

Run bulk dataset hashing outside throughput measurements: it reads every byte.
Manifests can contain private paths; keep them with local operational artifacts.
Inputs must remain immutable while recording, verifying, and using a manifest.
"""

import argparse
import hashlib
import json
from pathlib import Path


def fingerprint(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"artifact is not a regular file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"artifact changed during hashing: {path}")
    return dict(bytes=after.st_size, sha256=digest.hexdigest())


def tree_files(path):
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"artifact tree is not a directory: {root}")
    # File symlinks are hashed through their target. Reject directory symlinks:
    # rglob does not follow them, which would silently omit nested model files.
    members = sorted(root.rglob("*"))
    if any(p.is_symlink() and (not p.exists() or p.is_dir()) for p in members):
        raise ValueError(f"tree contains a dangling or directory symlink: {root}")
    return [str(p.relative_to(root)) for p in members if p.is_file()]


def record(files, trees):
    result = dict(version=1, files={}, trees={})
    for path in files:
        path = str(Path(path).absolute())
        result["files"][path] = fingerprint(path)
    for path in trees:
        root = Path(path).absolute()
        names = tree_files(root)
        if not names:
            raise ValueError(f"artifact tree is empty: {root}")
        result["trees"][str(root)] = {name: fingerprint(root / name) for name in names}
        if names != tree_files(root):
            raise ValueError(f"artifact tree membership changed during hashing: {root}")
    if not result["files"] and not result["trees"]:
        raise ValueError("at least one explicit file or tree is required")
    return result


def verify(manifest):
    if manifest.get("version") != 1 or not (manifest.get("files") or manifest.get("trees")):
        raise ValueError("invalid/empty artifact manifest")
    # Re-enumeration rejects added/removed files as well as changed content.
    actual = record(manifest["files"], manifest["trees"])
    if actual != manifest:
        raise ValueError("artifact identity mismatch; do not launch with changed inputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--tree", action="append", default=[])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path)
    mode.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.check:
        if args.file or args.tree:
            parser.error("--check uses only the manifest's recorded inputs")
        verify(json.loads(args.check.read_text()))
        print("Artifact hashes and tree membership match.")
    else:
        output = args.out.resolve()
        if any(output.is_relative_to(Path(tree).resolve()) for tree in args.tree):
            parser.error("manifest output must be outside the trees being pinned")
        manifest = record(args.file, args.tree)
        # Do not silently overwrite a previous pin.
        with args.out.open("x") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        print("Artifact manifest written; keep inputs immutable after verification.")


if __name__ == "__main__":
    main()

"""Retain actual compiler-input timestamps after content/provenance validation.

Cargo fingerprints include source mtimes. A new checkout must recover the
producer's original times to reuse its compiled artifacts. This restores
recorded metadata only after every tracked input matches the same Git revision
and SHA-256 digest; any mismatch fails before the first timestamp write.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import os
import subprocess


INPUTS = ("Cargo.toml", "Cargo.lock", "crates", "xtask")


def inputs(repository: Path) -> tuple[list[str], list[str]]:
    changed = subprocess.check_output(
        ["git", "-C", str(repository), "diff", "--name-only", "HEAD", "--", *INPUTS], timeout=30,
    ).decode().strip()
    if changed:
        raise ValueError("compiler inputs differ from HEAD: " + changed)
    files = subprocess.check_output(
        ["git", "-C", str(repository), "ls-files", "-z", "--", *INPUTS], timeout=30,
    ).decode().rstrip("\0").split("\0")
    if not files or files == [""]:
        raise ValueError("compiler input inventory is empty")
    directories = set()
    for name in files:
        path = repository / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"compiler input is not a regular tracked file: {name}")
        parent = Path(name).parent
        while parent != Path("."):
            directories.add(str(parent))
            parent = parent.parent
    return files, sorted(directories)


def record(repository: Path) -> dict:
    files, directories = inputs(repository)
    result = {"files": {}, "directories": {name: (repository / name).stat().st_mtime_ns for name in directories}}
    for name in files:
        path = repository / name
        before = path.stat().st_mtime_ns
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_mtime_ns != before:
            raise ValueError(f"compiler input changes during retention: {name}")
        result["files"][name] = {"sha256": digest, "mtime_ns": before}
    return result


def restore(repository: Path, saved: dict) -> None:
    files, directories = inputs(repository)
    if set(saved) != {"files", "directories"} or set(saved["files"]) != set(files) or set(saved["directories"]) != set(directories):
        raise ValueError("compiler input inventory differs")
    # Validate the whole manifest before writing any file or directory time.
    for name in files:
        entry = saved["files"][name]
        if set(entry) != {"sha256", "mtime_ns"} or hashlib.sha256((repository / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"compiler input content differs: {name}")
        if type(entry["mtime_ns"]) is not int or not 0 <= entry["mtime_ns"] <= (1 << 63) - 1:
            raise ValueError(f"invalid retained source timestamp: {name}")
    for name in directories:
        if type(saved["directories"][name]) is not int or not 0 <= saved["directories"][name] <= (1 << 63) - 1:
            raise ValueError(f"invalid retained directory timestamp: {name}")
        if (repository / name).is_symlink():
            raise ValueError(f"compiler input directory is a symlink: {name}")
    for name in files:
        path = repository / name
        os.utime(path, ns=(path.stat().st_atime_ns, saved["files"][name]["mtime_ns"]))
    for name in sorted(directories, key=lambda name: len(Path(name).parts), reverse=True):
        path = repository / name
        os.utime(path, ns=(path.stat().st_atime_ns, saved["directories"][name]))

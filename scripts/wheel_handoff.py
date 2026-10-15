#!/usr/bin/env python3
"""Bind a complete built wheel to its exact source head and locked graph."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


KEYS = {"head", "lock_sha256", "filename", "wheel_sha256"}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def source_identity(repository: Path) -> dict:
    subprocess.run(["git", "-C", str(repository), "diff", "--exit-code", "HEAD", "--", "Cargo.lock"],
                   check=True, capture_output=True, timeout=15)
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                   text=True, timeout=15).strip()
    return {"head": head, "lock_sha256": digest(repository / "Cargo.lock")}


def wheel_path(directory: Path) -> Path:
    entries = list(directory.iterdir())
    if (len(entries) != 1 or entries[0].is_symlink() or not entries[0].is_file()
            or not re.fullmatch(r"[A-Za-z0-9_.+!-]+\.whl", entries[0].name)):
        raise ValueError("artifact must contain exactly one regular wheel")
    return entries[0]


def describe(repository: Path, directory: Path) -> dict:
    wheel = wheel_path(directory)
    return dict(source_identity(repository), filename=wheel.name, wheel_sha256=digest(wheel))


def verify(repository: Path, directory: Path, expected: dict) -> Path:
    if not isinstance(expected, dict) or set(expected) != KEYS:
        raise ValueError("invalid wheel identity schema")
    for key, length in (("head", 40), ("lock_sha256", 64), ("wheel_sha256", 64)):
        if not isinstance(expected[key], str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, expected[key]):
            raise ValueError("invalid " + key)
    if not isinstance(expected["filename"], str) or not re.fullmatch(r"[A-Za-z0-9_.+!-]+\.whl", expected["filename"]):
        raise ValueError("invalid wheel filename")
    actual = describe(repository, directory)
    for key in sorted(KEYS):
        if actual[key] != expected[key]:
            raise ValueError("wheel handoff mismatch: " + key)
    return directory / actual["filename"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("describe", "verify"))
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "describe":
        print("identity=" + json.dumps(describe(args.repository, args.directory), separators=(",", ":")))
    else:
        verify(args.repository, args.directory, json.loads(os.environ["WHEEL_IDENTITY"]))
        print("Complete wheel, source head and lockfile identities verified")


if __name__ == "__main__":
    main()

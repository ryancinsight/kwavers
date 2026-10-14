#!/usr/bin/env python3
"""Persist only k-Wave reference data with complete CI environment provenance.

The bridge already keys each NPZ by grid, medium, source, sensor and options.
This outer namespace binds those inputs to actual reference code, the CPU
executable, dependency records, Python, OS packages and execution environment.
An exact-key restore must match its sealed byte manifest before comparisons.
Kwavers outputs and test outcomes are never stored here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


MANIFEST = "manifest.json"
REFERENCE_NAME = re.compile(r"[0-9a-f]{64}\.npz")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(128 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(files: dict[str, Path], environment: dict) -> str:
    payload = {
        "files": {name: file_digest(path) for name, path in sorted(files.items())},
        "environment": environment,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def reference_identity(repository: Path) -> str:
    import kwave
    from kwave.options.simulation_execution_options import SimulationExecutionOptions

    options = SimulationExecutionOptions(
        is_gpu_simulation=False, verbose_level=0, show_sim_log=False,
    )
    binary = Path(options.binary_path)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"k-Wave CPU executable is unavailable: {binary}")
    package = Path(kwave.__file__).resolve().parent
    files = {
        "reference/" + str(path.relative_to(package)): path
        for path in package.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and "bin" not in path.relative_to(package).parts
    }
    files["cpu-executable"] = binary
    for relative in (
        "scripts/kwave_reference_cache.py",
        ".github/workflows/python-wheel-smoke.yml",
        "crates/kwavers-python/python/pykwavers/comparison.py",
        "crates/kwavers-python/python/pykwavers/kwave_python_bridge.py",
        "crates/kwavers-python/tests/test_kwave_comparison.py",
    ):
        files["source/" + relative] = repository / relative
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"].lower()
        if name == "kwavers-python":
            continue
        record = distribution.read_text("RECORD")
        if record is None:
            raise ValueError(f"Installed reference dependency has no RECORD: {name}")
        packages[name] = {
            "version": distribution.version,
            "record": hashlib.sha256(record.encode()).hexdigest(),
        }
    environment = {
        "schema": 1,
        "python": sys.version,
        "system": platform.system(),
        "machine": platform.machine(),
        "libc": platform.libc_ver(),
        "num_threads": options.num_threads,
        "variables": {
            name: value for name, value in os.environ.items()
            if name.startswith(("OMP_", "MKL_", "OPENBLAS_", "KWAVE_"))
        },
        "packages": packages,
        "system_packages": subprocess.check_output(
            ["dpkg-query", "--show", "--showformat=${binary:Package}=${Version}\n"],
            text=True, timeout=30,
        ),
    }
    return identity(files, environment)


def snapshot(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_symlink():
            raise ValueError(f"Reference cache contains a symlink: {path}")
        if path.name == MANIFEST:
            continue
        if not path.is_file() or not REFERENCE_NAME.fullmatch(path.name):
            raise ValueError(f"Unexpected reference cache entry: {path}")
        result[path.name] = file_digest(path)
    if not result:
        raise ValueError("Reference cache contains no numerical results")
    return result


def seal(directory: Path, expected: str) -> None:
    manifest = {"identity": expected, "files": snapshot(directory)}
    (directory / MANIFEST).write_text(json.dumps(manifest, sort_keys=True) + "\n")


def verify(directory: Path, expected: str) -> None:
    actual = {"identity": expected, "files": snapshot(directory)}
    manifest = json.loads((directory / MANIFEST).read_text())
    if manifest != actual:
        raise ValueError("k-Wave reference cache provenance or contents differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    namespace = subparsers.add_parser("identity")
    namespace.add_argument("--repository", type=Path, required=True)
    namespace.add_argument("--output", type=Path, required=True)
    for name in ("seal", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--directory", type=Path, required=True)
        command.add_argument("--identity", required=True)
    arguments = parser.parse_args()
    if arguments.command == "identity":
        value = reference_identity(arguments.repository)
        with arguments.output.open("a") as output:
            output.write(f"identity={value}\n")
    elif arguments.command == "seal":
        seal(arguments.directory, arguments.identity)
    else:
        verify(arguments.directory, arguments.identity)


if __name__ == "__main__":
    main()

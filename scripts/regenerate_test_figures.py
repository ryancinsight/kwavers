"""Regenerate the committed PNG figures owned by four integration binaries.

Each binary runs in its own locked, serial Nextest subprocess. The command
stops on the first failed or timed-out binary and leaves generated deltas for
explicit review.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
REGENERATE_ENV = "KWAVERS_REGENERATE_TEST_FIGURES"
TEST_BINARIES = (
    "absorption_validation_test",
    "fdtd_pstd_comparison",
    "imaging_literature_validation",
    "test_pstd_kwave_comparison",
)
# Each selected binary is smaller than the repository's complete integration
# suite. Five minutes matches the committed heavy-test per-test ceiling while
# bounding a Cargo, Nextest, or test-harness hang at the subprocess boundary.
BINARY_TIMEOUT_SECONDS = 5 * 60


def command_for(binary: str) -> list[str]:
    """Return the locked command selecting exactly one integration binary."""
    return [
        "cargo",
        "nextest",
        "run",
        "--color",
        "never",
        "--locked",
        "-p",
        "kwavers",
        "--no-default-features",
        "--features",
        "full",
        "--test",
        binary,
        "--test-threads=1",
        "--no-fail-fast",
    ]


def regenerate() -> int:
    """Regenerate all selected binaries serially, returning the first failure."""
    environment = os.environ.copy()
    environment[REGENERATE_ENV] = "1"

    for binary in TEST_BINARIES:
        command = command_for(binary)
        print(f"regenerating figures from {binary}", flush=True)
        try:
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                shell=False,
                timeout=BINARY_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            print(
                f"error: {binary} exceeded {BINARY_TIMEOUT_SECONDS} seconds",
                file=sys.stderr,
            )
            return 124

        if completed.returncode != 0:
            print(
                f"error: {binary} exited with status {completed.returncode}",
                file=sys.stderr,
            )
            return completed.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(regenerate())

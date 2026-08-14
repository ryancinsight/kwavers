#!/usr/bin/env python3
"""Regenerate or check `Cargo.lock` against the source set CI actually resolves.

# The trap this exists for

This repository is normally worked on inside the Atlas stack, whose
`.cargo/config.toml` carries a `[patch]` overlay redirecting every first-party
dependency to a local working tree. Cargo discovers that config by walking up
from the *current directory*, so any `cargo` command run from inside the stack
picks it up -- including anything that rewrites the lock.

A lock written with the overlay active has every `source = "git+..."` line
**stripped**, because those dependencies resolved to local paths rather than to
git. Committing it replaces all 87 git sources with nothing. CI has no overlay,
so it re-resolves, and every `--locked` job fails with

    error: cannot update the lock file ... because --locked was passed

which names neither the cause nor the fix. That message is also what a merely
*stale* lock produces -- one pinning first-party revisions whose versions no
longer satisfy the manifests -- so the two failures are indistinguishable from
the log alone (KW-CI-087).

This is not limited to deliberate regeneration. *Any* cargo invocation that
updates the lock while the overlay is active flattens it -- an ordinary
`cargo check` inside the stack is enough, which is how it happens in practice:
nobody sets out to rewrite the lock. Treat a modified `Cargo.lock` after routine
work as suspect and run `--check` before staging it.

Both are fixed the same way: regenerate from outside the overlay. This script
does that by running cargo from a temporary directory that is not underneath the
stack root, which is the whole mechanism -- there is no flag that disables config
discovery.

# Usage

    scripts/lockfile.py --check        # verify the committed lock, offline
    scripts/lockfile.py --regenerate   # rewrite it correctly (needs network)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
LOCKFILE = REPOSITORY / "Cargo.lock"
MANIFEST = REPOSITORY / "Cargo.toml"

# Any first-party dependency resolves through one of these. A lock with none of
# them has been flattened by the overlay.
FIRST_PARTY_SOURCE = re.compile(r'^source = "git\+https://github\.com/ryancinsight/', re.M)


def run_outside_the_overlay(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run cargo with a working directory outside the stack root.

    Cargo resolves `.cargo/config.toml` by walking up from the working
    directory, never from `--manifest-path`, so this is what excludes the
    overlay. Running from the repository itself would silently include it.
    """
    with tempfile.TemporaryDirectory() as neutral_directory:
        return subprocess.run(
            ["cargo", *arguments, "--manifest-path", str(MANIFEST)],
            cwd=neutral_directory,
            capture_output=True,
            text=True,
            check=False,
        )


def check() -> int:
    if not LOCKFILE.is_file():
        print(f"error: {LOCKFILE} does not exist", file=sys.stderr)
        return 1

    sources = len(FIRST_PARTY_SOURCE.findall(LOCKFILE.read_text(encoding="utf-8")))
    if sources == 0:
        print(
            "error: Cargo.lock contains no first-party git sources.\n"
            "\n"
            "It was regenerated with the Atlas stack overlay active, which\n"
            "resolves those dependencies to local paths and drops their git\n"
            "sources. CI has no overlay and will fail every --locked job.\n"
            "\n"
            "Fix: scripts/lockfile.py --regenerate",
            file=sys.stderr,
        )
        return 1

    completed = run_outside_the_overlay(
        ["metadata", "--locked", "--format-version", "1", "--all-features"]
    )
    if completed.returncode != 0:
        print(
            f"error: the committed Cargo.lock does not resolve under --locked "
            f"({sources} first-party git sources present, so it is stale rather "
            f"than flattened).\n"
            f"\n"
            f"The pinned first-party revisions no longer satisfy the manifests'\n"
            f"version requirements, so cargo must re-resolve and --locked\n"
            f"refuses. This is what blocks the benchmark baseline alignment.\n"
            f"\n"
            f"Fix: scripts/lockfile.py --regenerate\n"
            f"\n"
            f"cargo said:\n{completed.stderr.strip()}",
            file=sys.stderr,
        )
        return 1

    print(f"Cargo.lock resolves under --locked; {sources} first-party git sources.")
    return 0


def regenerate() -> int:
    completed = run_outside_the_overlay(["generate-lockfile"])
    if completed.returncode != 0:
        print(f"error: regeneration failed:\n{completed.stderr.strip()}", file=sys.stderr)
        return 1
    print("Cargo.lock regenerated outside the overlay.")
    return check()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify the committed lock")
    mode.add_argument("--regenerate", action="store_true", help="rewrite the lock correctly")
    arguments = parser.parse_args()
    return regenerate() if arguments.regenerate else check()


if __name__ == "__main__":
    raise SystemExit(main())

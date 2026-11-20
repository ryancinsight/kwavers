#!/usr/bin/env python3
"""Regenerate or check `Cargo.lock` against the source set CI actually resolves.

# The trap this exists for

This repository is normally worked on inside the Atlas stack, whose
`.cargo/config.toml` carries a `[patch]` overlay redirecting every first-party
dependency to a local working tree. Cargo discovers that config by walking up
from the *current directory*, so any `cargo` command run from inside the stack
picks it up -- including anything that rewrites the lock.

A lock written with the overlay active records those dependencies as
source-less local packages. In a standalone workspace, only the
declared workspace members may appear without a registry or Git source.
This invariant also holds when the whole external graph uses crates.io.

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

    scripts/lockfile.py --check         # verify the committed lock
    scripts/lockfile.py --check-staged  # fast index-only check, for pre-commit
    scripts/lockfile.py --regenerate    # rewrite it correctly (needs network)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
LOCKFILE = REPOSITORY / "Cargo.lock"
MANIFEST = REPOSITORY / "Cargo.toml"


def external_local_packages(
    lock_text: str, read_manifest: Callable[[str], str]
) -> list[str]:
    """Return source-less packages outside the declared workspace members."""
    workspace = tomllib.loads(read_manifest("Cargo.toml"))["workspace"]
    members = set()
    for member in workspace["members"]:
        package = tomllib.loads(read_manifest(f"{member}/Cargo.toml"))["package"]
        version = package["version"]
        if version == {"workspace": True}:
            version = workspace["package"]["version"]
        members.add((package["name"], version))

    packages = tomllib.loads(lock_text)["package"]
    return sorted(
        f"{package['name']} v{package['version']}"
        for package in packages
        if "source" not in package and (package["name"], package["version"]) not in members
    )


def check_package_sources(
    lock_text: str, read_manifest: Callable[[str], str]
) -> int:
    external = external_local_packages(lock_text, read_manifest)
    if not external:
        return 0
    print(
        "error: Cargo.lock contains local packages outside this workspace: "
        + ", ".join(external)
        + "\nRegenerate it outside the Atlas overlay with scripts/lockfile.py --regenerate.",
        file=sys.stderr,
    )
    return 1


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
            # `text=True` alone decodes with the locale codepage. Cargo emits
            # UTF-8, so on a Windows console (cp1252) subprocess's reader thread
            # dies on the first byte it cannot map and the captured stream is
            # lost. The verdict survives -- it comes from `returncode` -- but the
            # message explaining a failure does not, which is the one moment it
            # is needed.
            encoding="utf-8",
            errors="replace",
            check=False,
        )


def check() -> int:
    if not LOCKFILE.is_file():
        print(f"error: {LOCKFILE} does not exist", file=sys.stderr)
        return 1

    if check_package_sources(
        LOCKFILE.read_text(encoding="utf-8"),
        lambda name: (REPOSITORY / name).read_text(encoding="utf-8"),
    ):
        return 1

    completed = run_outside_the_overlay(
        ["metadata", "--locked", "--format-version", "1", "--all-features"]
    )
    if completed.returncode != 0:
        print(
            "error: Cargo.lock does not resolve under --locked "
            "outside the Atlas overlay.\n"
            "Regenerate it with scripts/lockfile.py --regenerate.\n"
            f"Cargo said:\n{completed.stderr.strip()}",
            file=sys.stderr,
        )
        return 1

    print("Cargo.lock resolves under --locked outside the Atlas overlay.")
    return 0


def check_staged() -> int:
    """Check staged lock and manifest blobs without running Cargo."""
    def index_file(path: str) -> str:
        return subprocess.check_output(
            ["git", "show", f":{path}"], cwd=REPOSITORY, encoding="utf-8"
        )

    staged = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--", "Cargo.lock"],
        cwd=REPOSITORY,
        encoding="utf-8",
    )
    if not staged.strip():
        return 0
    return check_package_sources(index_file("Cargo.lock"), index_file)


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
    mode.add_argument(
        "--check-staged",
        action="store_true",
        help="fast structural check of the staged lock, for pre-commit",
    )
    arguments = parser.parse_args()
    try:
        if arguments.regenerate:
            return regenerate()
        if arguments.check_staged:
            return check_staged()
        return check()
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: unable to verify the lockfile inputs: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

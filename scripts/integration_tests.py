"""Run the kwavers integration suite against a committed baseline of failures.

CI compiles every `crates/kwavers/tests/*.rs` -- the strict clippy step passes
`--all-targets` -- but runs almost none of them: the test-coverage job is `--lib`
plus four named integration binaries. 686 integration tests were therefore
compiled and never executed, and 17 of them had been failing unnoticed.

Fixing all 17 before running any of them would leave the other 669 unprotected
for however long that takes. So this enforces the shape the repository already
uses for the clippy floor (ADR 116): a committed baseline that may shrink and
may not grow.

The baseline is a *set*, not a count. A count says "17 failures" and cannot tell
a fixed test from a newly broken one; the set names both. Two ways to fail:

  - a test fails that is not in the baseline -- a regression, named;
  - a test in the baseline passes -- the entry is stale and must be deleted, so
    a fixed test cannot silently leave room for a new failure to hide in.

Usage:
    python scripts/integration_tests.py            # enforce
    python scripts/integration_tests.py --update   # rewrite the baseline
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

# CI sets CARGO_TERM_COLOR=always, so nextest wraps every field in escape
# sequences and a regex written against plain output silently matches nothing.
# `--color never` below is the fix; this strips anything that still arrives
# coloured, because a parser reporting "no failures" from output it could not
# read is the exact failure this check exists to prevent.
ANSI = re.compile("\x1b\\[[0-9;]*m")

BASELINE = pathlib.Path(".config/integration-test-baseline.txt")
# nextest prints "        FAIL [   0.010s] (85/686) <binary> <test path>", and
# "     TIMEOUT [  60.097s] (535/683) ..." for one that hit the termination
# bound. A timed-out test is a failed test: matching only FAIL left two SWE
# timeouts invisible here while nextest itself reported them plainly.
FAIL_LINE = re.compile(
    r"^\s+(?:FAIL|TIMEOUT)\s+\[[^\]]*\]\s+\(\s*\d+/\d+\)\s+(\S+)\s+(\S+)\s*$"
)
SUMMARY = re.compile(r"^\s+Summary\s+\[[^\]]*\]\s+(\d+) tests run")

COMMAND = [
    "cargo", "nextest", "run",
    "--color", "never",
    "-p", "kwavers", "--tests",
    "--no-default-features", "--features", "full",
    "--test-threads=1", "--no-fail-fast",
]


def read_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    return {
        line.strip()
        for line in BASELINE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def write_baseline(failures: set[str]) -> None:
    header = (
        "# Integration tests known to fail, one per line, as\n"
        "# `<binary> <test path>`. Enforced by scripts/integration_tests.py:\n"
        "# a failure absent from this list is a regression, and an entry that\n"
        "# passes is stale and must be removed. The list may shrink, never grow.\n"
        "#\n"
        "# Every entry is a defect owed a fix, not an accepted behaviour. See\n"
        "# backlog.md KW-INTEGRATION-TESTS-UNRUN for the inventory.\n"
    )
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(header + "".join(f"{f}\n" for f in sorted(failures)), encoding="utf-8")


def run(locked: bool) -> tuple[set[str], int]:
    # CI resolves against the committed lockfile. A tree under the Atlas
    # development overlay cannot: the overlay redirects first-party crates to
    # local paths, so `--locked` refuses before the suite starts.
    command = COMMAND[:5] + (["--locked"] if locked else []) + COMMAND[5:]
    result = subprocess.run(command, capture_output=True, text=True)
    output = ANSI.sub("", result.stdout + result.stderr)
    failures = {f"{m.group(1)} {m.group(2)}" for line in output.splitlines()
                if (m := FAIL_LINE.match(line))}
    ran = 0
    for line in output.splitlines():
        if m := SUMMARY.match(line):
            ran = int(m.group(1))
    if ran == 0:
        # No summary means the suite did not run -- a compile error or a
        # harness failure. Reporting "no new failures" there would be the
        # check passing because it never executed.
        print(output[-4000:], file=sys.stderr)
        raise SystemExit("integration suite produced no summary; it did not run")
    return failures, ran


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="rewrite the baseline from this run")
    parser.add_argument("--unlocked", action="store_true",
                        help="drop --locked, for a tree under the Atlas overlay")
    args = parser.parse_args()

    failures, ran = run(locked=not args.unlocked)
    print(f"integration suite: {ran} tests run, {len(failures)} failed")

    if args.update:
        write_baseline(failures)
        print(f"baseline rewritten with {len(failures)} entries")
        return 0

    baseline = read_baseline()
    regressions = sorted(failures - baseline)
    fixed = sorted(baseline - failures)

    for test in regressions:
        print(f"::error::integration regression: {test}")
    for test in fixed:
        print(f"::error::{test} now passes; remove it from {BASELINE}")

    if regressions or fixed:
        print(f"\n{len(regressions)} regression(s), {len(fixed)} stale entry(ies).")
        print(f"Refresh with: python {__file__.replace(chr(92), '/')} --update")
        return 1

    print(f"no regressions; {len(baseline)} known failures unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

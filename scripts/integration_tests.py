"""Run the kwavers integration suite against a committed baseline of failures.

CI compiles every `crates/kwavers/tests/*.rs` -- the strict clippy step passes
`--all-targets` -- but runs almost none of them: the test-coverage job is `--lib`
plus four named integration binaries. Most integration tests were therefore
compiled and never executed, and multiple failures went unnoticed.

Fixing every known failure before running any of them would leave the passing
majority unprotected for however long that takes. So this enforces the shape
the repository already uses for the clippy floor (ADR 116): a committed
baseline that may shrink and may not grow.

The baseline is a *set*, not a count. A count cannot distinguish a fixed test
from a newly broken one; the set names both. Two ways to fail:

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
# The complete command took 17m28s on the 4-core hosted runner. Twenty-five
# minutes retains 43% headroom while ensuring a compile or harness hang cannot
# consume the enclosing 45-minute job without a specific diagnostic.
INTEGRATION_RUN_TIMEOUT_SECONDS = 25 * 60
# nextest prints a timed status, ordinal/total pair, binary, and test path, and
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
    "--locked",
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


def run() -> tuple[set[str], int]:
    try:
        result = subprocess.run(
            COMMAND,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=INTEGRATION_RUN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        for label, stream in (("stdout", error.stdout), ("stderr", error.stderr)):
            if isinstance(stream, bytes):
                stream = stream.decode("utf-8", errors="replace")
            if stream:
                print(f"{label} tail:\n{stream[-4000:]}", file=sys.stderr)
        raise SystemExit(
            f"integration suite exceeded {INTEGRATION_RUN_TIMEOUT_SECONDS} seconds"
        ) from error
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
    args = parser.parse_args()

    failures, ran = run()
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

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
from dataclasses import dataclass
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
from typing import BinaryIO

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
PROCESS_TREE_TERMINATION_TIMEOUT_SECONDS = 10
DIAGNOSTIC_TAIL_CHARACTERS = 4000
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
    "-p", "kwavers", "--tests",
    "--no-default-features", "--features", "full",
    "--test-threads=1", "--no-fail-fast",
]

WINDOWS_PROCESS_BOOTSTRAP = """
import subprocess
import sys

if sys.stdin.buffer.read(1) != b"1":
    raise SystemExit("process-tree bootstrap was not released")
raise SystemExit(subprocess.run(sys.argv[1:], check=False).returncode)
"""


@dataclass(frozen=True)
class ExecutionResult:
    """Bounded output and status from one nextest process tree."""

    returncode: int
    timed_out: bool
    termination_error: str | None
    failures: frozenset[str]
    tests_run: int
    stdout_tail: str
    stderr_tail: str
    failure_status_tail: str


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


def _create_windows_kill_job(process: subprocess.Popen[bytes]) -> int:
    """Assign a Windows process to a job that owns every descendant."""

    import ctypes
    from ctypes import wintypes

    class LargeInteger(ctypes.Structure):
        _fields_ = [("quad_part", ctypes.c_longlong)]

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("per_process_user_time_limit", LargeInteger),
            ("per_job_user_time_limit", LargeInteger),
            ("limit_flags", wintypes.DWORD),
            ("minimum_working_set_size", ctypes.c_size_t),
            ("maximum_working_set_size", ctypes.c_size_t),
            ("active_process_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority_class", wintypes.DWORD),
            ("scheduling_class", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("read_operation_count", ctypes.c_ulonglong),
            ("write_operation_count", ctypes.c_ulonglong),
            ("other_operation_count", ctypes.c_ulonglong),
            ("read_transfer_count", ctypes.c_ulonglong),
            ("write_transfer_count", ctypes.c_ulonglong),
            ("other_transfer_count", ctypes.c_ulonglong),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("basic_limit_information", BasicLimitInformation),
            ("io_info", IoCounters),
            ("process_memory_limit", ctypes.c_size_t),
            ("job_memory_limit", ctypes.c_size_t),
            ("peak_process_memory_used", ctypes.c_size_t),
            ("peak_job_memory_used", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    information = ExtendedLimitInformation()
    information.basic_limit_information.limit_flags = 0x00002000
    if not kernel32.SetInformationJobObject(
        job, 9, ctypes.byref(information), ctypes.sizeof(information)
    ):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(job)
        raise error

    process_handle = getattr(process, "_handle", None)
    if process_handle is None or not kernel32.AssignProcessToJobObject(
        job, wintypes.HANDLE(int(process_handle))
    ):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(job)
        raise error
    return int(job)


def _close_windows_job(job: int) -> str | None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    if kernel32.CloseHandle(wintypes.HANDLE(job)):
        return None
    return f"job handle close failed: {ctypes.WinError(ctypes.get_last_error())}"


def _terminate_process_tree(
    process: subprocess.Popen[bytes], windows_job: int | None
) -> str | None:
    """Force-stop the process and descendants, returning any cleanup error."""

    error: str | None = None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        if windows_job is None:
            error = "process was not assigned to a Windows kill-on-close job"
        else:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel32.TerminateJobObject.restype = wintypes.BOOL
            if not kernel32.TerminateJobObject(wintypes.HANDLE(windows_job), 1):
                error = f"job termination failed: {ctypes.WinError(ctypes.get_last_error())}"
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as caught:
            error = f"process-group kill failed: {caught}"

    if process.poll() is None and error is not None:
        process.kill()
    try:
        process.wait(timeout=PROCESS_TREE_TERMINATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=PROCESS_TREE_TERMINATION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            error = "process did not exit after tree termination and direct kill"
    return error


def _scan_stream(stream: BinaryIO) -> tuple[set[str], int, str, str]:
    failures: set[str] = set()
    ran = 0
    tail = ""
    failure_status_tail = ""
    stream.seek(0)
    for raw_line in stream:
        line = ANSI.sub("", raw_line.decode("utf-8", errors="replace"))
        tail = (tail + line)[-DIAGNOSTIC_TAIL_CHARACTERS:]
        if match := FAIL_LINE.match(line):
            failures.add(f"{match.group(1)} {match.group(2)}")
            failure_status_tail = (failure_status_tail + line)[
                -DIAGNOSTIC_TAIL_CHARACTERS:
            ]
        if match := SUMMARY.match(line):
            ran = max(ran, int(match.group(1)))
    return failures, ran, tail, failure_status_tail


def _execute(command: list[str], timeout_seconds: float) -> ExecutionResult:
    if sys.platform == "win32":
        launch_command = [sys.executable, "-c", WINDOWS_PROCESS_BOOTSTRAP, *command]
        process_options = {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
            "stdin": subprocess.PIPE,
        }
    else:
        launch_command = command
        process_options = {"start_new_session": True}
    with tempfile.TemporaryFile(mode="w+b") as stdout, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr:
        process = subprocess.Popen(
            launch_command,
            stdout=stdout,
            stderr=stderr,
            **process_options,
        )
        timed_out = False
        termination_error = None
        windows_job = None
        if sys.platform == "win32":
            try:
                windows_job = _create_windows_kill_job(process)
            except OSError as caught:
                termination_error = f"failed to establish process-tree ownership: {caught}"
                process.kill()
                try:
                    process.wait(timeout=PROCESS_TREE_TERMINATION_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    termination_error += "; bootstrap did not exit after direct kill"
            else:
                if process.stdin is None:
                    termination_error = "Windows process-tree bootstrap has no control pipe"
                    process.kill()
                else:
                    process.stdin.write(b"1")
                    process.stdin.close()
        if termination_error is None:
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                termination_error = _terminate_process_tree(process, windows_job)
        if windows_job is not None:
            close_error = _close_windows_job(windows_job)
            if close_error is not None:
                termination_error = termination_error or close_error

        stdout_failures, stdout_ran, stdout_tail, stdout_failure_status = _scan_stream(
            stdout
        )
        stderr_failures, stderr_ran, stderr_tail, stderr_failure_status = _scan_stream(
            stderr
        )
        return ExecutionResult(
            returncode=process.returncode if process.returncode is not None else -1,
            timed_out=timed_out,
            termination_error=termination_error,
            failures=frozenset(stdout_failures | stderr_failures),
            tests_run=max(stdout_ran, stderr_ran),
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            failure_status_tail=(stdout_failure_status + stderr_failure_status)[
                -DIAGNOSTIC_TAIL_CHARACTERS:
            ],
        )


def _print_diagnostic_tails(result: ExecutionResult) -> None:
    for label, tail in (
        ("stdout", result.stdout_tail),
        ("stderr", result.stderr_tail),
    ):
        if tail:
            print(f"{label} tail:\n{tail}", file=sys.stderr)


def _print_failure_statuses(result: ExecutionResult) -> None:
    if result.failure_status_tail:
        print(
            f"failure status lines:\n{result.failure_status_tail}",
            file=sys.stderr,
        )


def run(locked: bool = True) -> tuple[set[str], int]:
    # CI resolves against the committed lockfile. A tree under the Atlas
    # development overlay cannot: the overlay redirects first-party crates to
    # local paths, so `--locked` refuses before the suite starts and the gate
    # is unrunnable on the machine where it would catch a failure earliest.
    command = COMMAND[:5] + (["--locked"] if locked else []) + COMMAND[5:]
    result = _execute(command, INTEGRATION_RUN_TIMEOUT_SECONDS)
    if result.termination_error is not None:
        _print_failure_statuses(result)
        _print_diagnostic_tails(result)
        raise SystemExit(
            "integration suite process-tree cleanup failed: "
            f"{result.termination_error}"
        )
    if result.timed_out:
        _print_failure_statuses(result)
        _print_diagnostic_tails(result)
        raise SystemExit(
            f"integration suite exceeded {INTEGRATION_RUN_TIMEOUT_SECONDS} seconds"
        )

    failures = set(result.failures)
    if failures:
        _print_failure_statuses(result)
    ran = result.tests_run
    if ran == 0:
        # No summary means the suite did not run -- a compile error or a
        # harness failure. Reporting "no new failures" there would be the
        # check passing because it never executed.
        _print_diagnostic_tails(result)
        raise SystemExit("integration suite produced no summary; it did not run")
    return failures, ran


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlocked", action="store_true",
                        help="drop --locked, for a tree under the Atlas overlay")
    parser.add_argument("--update", action="store_true",
                        help="rewrite the baseline from this run")
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

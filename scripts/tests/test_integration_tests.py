"""Regression tests for the bounded integration-test process runner."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest

from scripts import integration_tests

if os.name != "nt":
    import fcntl


class IntegrationTestRunnerTests(unittest.TestCase):
    def test_output_parsing_retains_only_bounded_tails(self) -> None:
        output = (
            "x" * 5000
            + "\n        FAIL [   0.010s] (1/2) kwavers::sample test_case\n"
            + "     TIMEOUT [  60.000s] (2/2) kwavers::sample timed_out_case\n"
            + "     Summary [  60.010s] 2 tests run: 0 passed, 2 failed\n"
        )

        result = integration_tests._execute(
            [sys.executable, "-c", f"print({output!r})"], 5.0
        )

        self.assertFalse(result.timed_out)
        self.assertIsNone(result.termination_error)
        self.assertEqual(
            result.failures,
            frozenset(
                {
                    "kwavers::sample test_case",
                    "kwavers::sample timed_out_case",
                }
            ),
        )
        self.assertEqual(result.tests_run, 2)
        self.assertIn("FAIL", result.failure_status_tail)
        self.assertIn("TIMEOUT", result.failure_status_tail)
        self.assertIn("kwavers::sample test_case", result.failure_status_tail)
        self.assertLessEqual(
            len(result.stdout_tail), integration_tests.DIAGNOSTIC_TAIL_CHARACTERS
        )
        self.assertLessEqual(
            len(result.failure_status_tail),
            integration_tests.DIAGNOSTIC_TAIL_CHARACTERS,
        )

    @unittest.skipIf(os.name == "nt", "POSIX process-group contract")
    def test_timeout_kills_descendant_holding_inherited_pipe(self) -> None:
        child_code = textwrap.dedent(
            """
            import fcntl
            import os
            import sys
            import time

            lock = open(sys.argv[1], "a+", encoding="utf-8")
            fcntl.flock(lock, fcntl.LOCK_EX)
            print(os.getpid(), flush=True)
            time.sleep(60)
            """
        )
        parent_code = textwrap.dedent(
            """
            import subprocess
            import sys
            import time

            child = subprocess.Popen(
                [sys.executable, "-c", sys.argv[2], sys.argv[1]],
                stdout=subprocess.PIPE,
                text=True,
            )
            print(child.stdout.readline(), end="", flush=True)
            time.sleep(60)
            """
        )

        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "descendant.lock"
            result = integration_tests._execute(
                [sys.executable, "-c", parent_code, str(lock_path), child_code],
                1.0,
            )
            child_pid = int(result.stdout_tail.strip())

            try:
                with lock_path.open("a+", encoding="utf-8") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.fail("timed-out descendant retained its file lock")
            finally:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        self.assertTrue(result.timed_out)
        self.assertIsNone(result.termination_error)

    @unittest.skipUnless(os.name == "nt", "Windows process-tree contract")
    def test_timeout_kills_windows_descendant(self) -> None:
        child_code = "import os,time; print(os.getpid(), flush=True); time.sleep(60)"
        parent_code = textwrap.dedent(
            """
            import subprocess
            import sys
            import time

            child = subprocess.Popen(
                [sys.executable, "-c", sys.argv[1]],
                stdout=subprocess.PIPE,
                text=True,
            )
            print(child.stdout.readline(), end="", flush=True)
            time.sleep(60)
            """
        )

        result = integration_tests._execute(
            [sys.executable, "-c", parent_code, child_code], 1.0
        )
        child_pid = int(result.stdout_tail.strip())
        process_list = subprocess.run(
            ["tasklist", "/FI", f"PID eq {child_pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )

        self.assertTrue(result.timed_out)
        self.assertIsNone(result.termination_error)
        self.assertNotIn(f'"{child_pid}"', process_list.stdout)


if __name__ == "__main__":
    unittest.main()

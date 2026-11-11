"""Runs the mirror selection script against a stub `curl` on PATH.

The stub logs the URL and the arguments of each call. It fails with status 6 for
the hosts in `STUB_UNREACHABLE`, and with status 22 for those in
`STUB_NOT_FOUND` only when `--fail` is among its arguments, as curl does for a 404.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".github/actions/install-system-dependencies/select-mirrors.sh"
)
BASH = shutil.which("bash")
A, B, C = "http://a.test/ubuntu/", "https://b.test/ubuntu/", "https://c.test/ubuntu/"
DEAD = "http://dead.test/ubuntu/"
KEPT = "every mirror answered; the list is left as found"

STUB_CURL = """#!/usr/bin/env bash
url="${@: -1}"
printf '%s\\n' "$url" >> "$STUB_LOG"
printf '%s\\n' "$*" >> "$STUB_ARGS"
host="${url#*://}"
host="${host%%/*}"
if [[ " $STUB_UNREACHABLE " == *" $host "* ]]; then
  echo "curl: (6) Could not resolve host: $host" >&2
  exit 6
fi
if [[ " $STUB_NOT_FOUND " == *" $host "* && " $* " == *" --fail "* ]]; then
  echo "curl: (22) The requested URL returned error: 404" >&2
  exit 22
fi
"""


def probe(mirror: str, codename: str = "noble") -> str:
    return f"{mirror.rstrip('/')}/dists/{codename}-security/InRelease"


@unittest.skipIf(BASH is None, "bash is not on PATH")
class MirrorSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "bin").mkdir()
        stub = root / "bin" / "curl"
        stub.write_bytes(STUB_CURL.encode())
        stub.chmod(0o755)
        self.root, self.list = root, root / "apt-mirrors.txt"
        self.log, self.arguments = root / "curl.log", root / "curl.args"

    def run_script(self, content, unreachable="", not_found="", codename="noble", argv=None):
        if content is not None:
            self.list.write_bytes(content.encode())
        env = {**os.environ, "STUB_LOG": str(self.log), "STUB_ARGS": str(self.arguments)}
        env["PATH"] = str(self.root / "bin") + os.pathsep + env["PATH"]
        env["STUB_UNREACHABLE"], env["STUB_NOT_FOUND"] = unreachable, not_found
        argv = [str(self.list), codename] if argv is None else argv
        return subprocess.run(
            [BASH, str(SCRIPT), *argv], env=env, capture_output=True, text=True, timeout=60
        )

    def select(self, content, written=None, probes=None, codename="noble", **stub):
        """Runs the script and checks exit 0, the file afterwards (default: unchanged) and the probes."""
        result = self.run_script(content, codename=codename, **stub)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = content if written is None else written
        self.assertEqual(self.list.read_bytes(), expected.encode())
        if probes is not None:
            logged = self.log.read_text().splitlines() if self.log.exists() else []
            self.assertEqual(logged, [probe(m, codename) for m in probes])
        return result

    def test_a_list_whose_mirrors_all_answer_is_left_byte_identical(self) -> None:
        text = f"# fastest first\n{A}\tpriority:9\n\n{B}\tpriority:2\ttype:deb\n{C}\tpriority:5\n"
        self.assertIn(KEPT, self.select(text).stdout)

    def test_a_crlf_list_whose_mirrors_all_answer_is_left_byte_identical(self) -> None:
        self.assertIn(KEPT, self.select(f"{A}\r\n{B}\tpriority:2\r\n").stdout)

    def test_only_the_lines_of_unreachable_mirrors_are_dropped(self) -> None:
        text = f"# m\n{A}\tpriority:9\ttype:deb\n{DEAD}\tpriority:1\n\n{C}\tpriority:5\n"
        written = f"# m\n{A}\tpriority:9\ttype:deb\n\n{C}\tpriority:5\n"
        result = self.select(text, written, unreachable="dead.test")
        self.assertIn(f"unreachable {DEAD} ", result.stdout)

    def test_a_mirror_answering_not_found_is_dropped(self) -> None:
        self.select(f"{A}\tpriority:1\n{B}\tpriority:2\n", f"{A}\tpriority:1\n", not_found="b.test")

    def test_every_probe_carries_a_connect_and_a_total_time_limit(self) -> None:
        self.select(f"{A}\n{B}\n{DEAD}\n", f"{A}\n{B}\n", unreachable="dead.test")
        calls = [line.split() for line in self.arguments.read_text().splitlines()]
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertEqual(call[call.index("--connect-timeout") + 1], "5")
            self.assertEqual(call[call.index("--max-time") + 1], "10")

    def test_a_mirror_is_probed_once_whether_or_not_its_lines_end_in_a_slash(self) -> None:
        text = f"{A}\tpriority:1\n{A[:-1]}\tpriority:2\n{B}\tpriority:3\n{A}\tpriority:4\n"
        self.select(text, probes=[A, B])

    def test_a_repeated_mirror_keeps_or_loses_all_its_lines_together(self) -> None:
        text = f"{DEAD}\tpriority:1\n{A}\tpriority:2\n{DEAD[:-1]}\tpriority:3\n{A[:-1]}\tpriority:4\n"
        self.select(text, f"{A}\tpriority:2\n{A[:-1]}\tpriority:4\n", unreachable="dead.test")

    def test_a_comment_line_naming_a_dead_mirror_is_not_probed(self) -> None:
        text = f"# {DEAD}\n#{DEAD}\tpriority:1\n{A}\tpriority:2\n"
        self.select(text, probes=[A], unreachable="dead.test")

    def test_a_dead_mirror_on_a_last_line_without_a_newline_is_dropped(self) -> None:
        result = self.select(f"{A}\tpriority:1\n{DEAD}\tpriority:2", f"{A}\tpriority:1\n",
                             probes=[A, DEAD], unreachable="dead.test")
        self.assertIn(f"\nreachable {A} ", result.stdout)

    def test_a_live_mirror_on_a_last_line_without_a_newline_is_probed_and_kept(self) -> None:
        result = self.select(f"{DEAD}\tpriority:1\n{B}\tpriority:2", f"{B}\tpriority:2\n",
                             probes=[DEAD, B], unreachable="dead.test")
        self.assertIn(f"\nunreachable {DEAD} ", result.stdout)

    def test_two_live_mirrors_without_a_final_newline_are_both_probed_and_unchanged(self) -> None:
        result = self.select(f"{A}\tpriority:1\n{B}\tpriority:2", probes=[A, B])
        self.assertIn(f"\nreachable {A} ", result.stdout)

    def test_a_single_line_without_a_newline_names_a_mirror(self) -> None:
        self.assertIn(f"\nreachable {A} ", self.select(A, probes=[A]).stdout)

    def test_a_uri_only_line_ending_in_crlf_is_probed_without_the_carriage_return(self) -> None:
        self.select(f"{A}\r\n{DEAD}\r\n", f"{A}\r\n", probes=[A, DEAD], unreachable="dead.test")

    def test_a_line_holding_only_a_carriage_return_is_not_probed_and_is_kept(self) -> None:
        self.select(f"{A}\r\n\r\n{DEAD}\r\n", f"{A}\r\n\r\n", probes=[A, DEAD], unreachable="dead.test")

    def test_probe_url_names_the_codename_security_release_file(self) -> None:
        self.select(f"{A}\n", probes=[A], codename="jammy")

    def test_no_mirror_answering_exits_one_and_leaves_the_list_unchanged(self) -> None:
        text = f"{A[:-1]}\tpriority:1\n{B}\n"
        result = self.run_script(text, unreachable="a.test b.test")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.list.read_bytes(), text.encode())
        self.assertIn(A[:-1], result.stderr)
        self.assertIn(B, result.stderr)

    def test_unusable_input_exits_two(self) -> None:
        cases = {
            "missing list": (None, None),
            "empty list": ("", None),
            "comment-only list": (f"# {A}\n\n  \n", None),
            "no arguments": (f"{A}\n", []),
            "three arguments": (f"{A}\n", [str(self.list), "noble", "extra"]),
            "empty codename": (f"{A}\n", [str(self.list), ""]),
        }
        for label, (content, argv) in cases.items():
            with self.subTest(label):
                self.list.unlink(missing_ok=True)
                self.assertEqual(self.run_script(content, argv=argv).returncode, 2)

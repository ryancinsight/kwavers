"""Runs the mirror selection script against a stub `curl` on PATH.

The stub fails for the hosts in `STUB_UNREACHABLE` and logs each URL it gets.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".github/actions/install-system-dependencies/select-mirrors.sh"
)
BASH = shutil.which("bash")

STUB_CURL = """#!/usr/bin/env bash
url="${@: -1}"
printf '%s\\n' "$url" >> "$STUB_LOG"
host="${url#*://}"
host="${host%%/*}"
for dead in $STUB_UNREACHABLE; do
  if [[ "$host" == "$dead" ]]; then
    echo "curl: (6) Could not resolve host: $host" >&2
    exit 6
  fi
done
"""


@unittest.skipIf(BASH is None, "bash is not on PATH")
class MirrorSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        root = Path(self._dir.name)
        self.bin_dir = root / "bin"
        self.bin_dir.mkdir()
        stub = self.bin_dir / "curl"
        stub.write_bytes(STUB_CURL.encode())
        stub.chmod(0o755)
        self.log = root / "curl.log"
        self.log.write_bytes(b"")
        self.list = root / "apt-mirrors.txt"

    def run_script(
        self, text: str | None, unreachable: str = "", codename: str = "noble"
    ) -> subprocess.CompletedProcess[str]:
        if text is not None:
            self.list.write_bytes(text.encode())
        env = dict(os.environ)
        env["PATH"] = str(self.bin_dir) + os.pathsep + env["PATH"]
        env["STUB_LOG"] = str(self.log)
        env["STUB_UNREACHABLE"] = unreachable
        return subprocess.run(
            [BASH, str(SCRIPT), str(self.list), codename],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def probed(self) -> list[str]:
        return self.log.read_text().splitlines()

    def test_keeps_reachable_mirrors_in_order_with_renumbered_priorities(self) -> None:
        result = self.run_script(
            "http://a.test/ubuntu/\tpriority:9\nhttps://b.test/ubuntu/\tpriority:2\n"
            "https://c.test/ubuntu/\tpriority:5\n"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.list.read_text(),
            "http://a.test/ubuntu/\tpriority:1\nhttps://b.test/ubuntu/\tpriority:2\n"
            "https://c.test/ubuntu/\tpriority:3\n",
        )

    def test_drops_an_unreachable_first_mirror_and_renumbers(self) -> None:
        result = self.run_script(
            "http://dead.test/ubuntu/\tpriority:1\n"
            "https://b.test/ubuntu/\tpriority:2\n"
            "https://c.test/ubuntu/\tpriority:3\n",
            unreachable="dead.test",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.list.read_text(),
            "https://b.test/ubuntu/\tpriority:1\nhttps://c.test/ubuntu/\tpriority:2\n",
        )
        self.assertIn("unreachable http://dead.test/ubuntu/ ", result.stdout)

    def test_probes_a_repeated_mirror_once_and_keeps_its_first_position(self) -> None:
        result = self.run_script(
            "http://a.test/ubuntu/\tpriority:1\n"
            "https://b.test/ubuntu/\tpriority:2\n"
            "http://a.test/ubuntu/\tpriority:3\n"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.probed(),
            [
                "http://a.test/ubuntu/dists/noble-security/InRelease",
                "https://b.test/ubuntu/dists/noble-security/InRelease",
            ],
        )
        self.assertEqual(
            self.list.read_text(),
            "http://a.test/ubuntu/\tpriority:1\nhttps://b.test/ubuntu/\tpriority:2\n",
        )

    def test_discards_metadata_after_the_uri_and_blank_lines(self) -> None:
        result = self.run_script(
            "\nhttp://a.test/ubuntu/   priority:7 weight:3\n\n"
            "https://b.test/ubuntu\n"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.list.read_text(),
            "http://a.test/ubuntu/\tpriority:1\nhttps://b.test/ubuntu\tpriority:2\n",
        )

    def test_probe_url_names_the_codename_security_release_file(self) -> None:
        result = self.run_script("https://a.test/ubuntu/\n", codename="jammy")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.probed(), ["https://a.test/ubuntu/dists/jammy-security/InRelease"]
        )

    def test_no_reachable_mirror_exits_one_and_leaves_the_list_unchanged(self) -> None:
        original = "http://x.test/ubuntu/\tpriority:1\nhttps://y.test/ubuntu/\n"

        result = self.run_script(original, unreachable="x.test y.test")

        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.list.read_bytes(), original.encode())
        self.assertIn("http://x.test/ubuntu/", result.stderr)
        self.assertIn("https://y.test/ubuntu/", result.stderr)

    def test_missing_list_exits_two(self) -> None:
        result = self.run_script(None)

        self.assertEqual(result.returncode, 2)

    def test_empty_list_exits_two(self) -> None:
        self.assertEqual(self.run_script("").returncode, 2)


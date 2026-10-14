"""Regressions for the actual wheel workflow's build-script handoff.

For each consumer, restore-compiled < retain < Maturin cleanup < restore.
The retention/restoration commands must preserve the complete file tree and
executable modes and must fail when their source artifacts are missing.
These tests execute the workflow commands on the compiled Kwavers build script;
installed-wheel and numerical behavior remain the hosted wheel suites' contract.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ".github/workflows/python-wheel-smoke.yml"
CONSUMERS = ("base-wheel", "kwave-comparison")
PR_ONLY = "github.event_name == 'pull_request'"


def load_yaml(relative_path: str) -> dict:
    # BaseLoader preserves GitHub's YAML 1.2 `on` key as a string.
    return yaml.load((ROOT / relative_path).read_text(), Loader=yaml.BaseLoader)


def matching_step(job: dict, field: str, value: str) -> dict:
    matches = [step for step in job["steps"] if step.get(field) == value]
    if len(matches) != 1:
        raise AssertionError(f"expected one {field}={value!r}, found {len(matches)}")
    return matches[0]


def restoration_script(build: dict, workspace: Path) -> str:
    expression = build["with"]["before-script-linux"]
    match = re.fullmatch(
        r"\$\{\{ github\.event_name == 'pull_request' && "
        r"format\('([^']*)', github\.workspace\) \|\| '' \}\}",
        expression,
    )
    if match is None:
        raise AssertionError("wheel restoration must use the pull-request workspace")
    script = match[1].format(str(workspace))
    if not script.startswith("set -e; "):
        raise AssertionError("wheel restoration must fail on its first error")
    # Cargo's /root configuration is container-specific. Execute the unchanged
    # build-script portion locally; the real Docker path is covered in wheel CI.
    return "set -e; " + script[script.index("mkdir -p "):]


def tree_identity(root: Path) -> dict:
    identity = {}
    for item in sorted(root.rglob("*")):
        mode = stat.S_IMODE(item.lstat().st_mode)
        if item.is_symlink():
            value = ("symlink", os.readlink(item))
        elif item.is_file():
            value = ("file", hashlib.sha256(item.read_bytes()).hexdigest())
        else:
            value = ("directory",)
        identity[str(item.relative_to(root))] = (mode, value)
    return identity


class PythonWheelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = load_yaml(WORKFLOW_PATH)
        cls.jobs = cls.workflow["jobs"]
        cls.compiled_directory = tempfile.TemporaryDirectory(prefix="kwavers-build-script-")
        cls.addClassCleanup(cls.compiled_directory.cleanup)
        cls.build_script = Path(cls.compiled_directory.name) / "build-script-build"
        subprocess.run(
            ["rustc", str(ROOT / "crates/kwavers/build.rs"), "-o", str(cls.build_script)],
            check=True, capture_output=True, text=True, timeout=30,
        )
        cls.build_output = subprocess.check_output(
            [str(cls.build_script)], text=True, timeout=5,
        )

    def commands(self, consumer: str, workspace: Path) -> tuple[str, str]:
        job = self.jobs[consumer]
        retain = matching_step(job, "name", "Retain matching compiled build scripts")
        build = matching_step(job, "name", "Build stable-ABI wheel")
        return retain["run"], restoration_script(build, workspace)

    def execute(self, command: str, workspace: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-c", command], cwd=workspace, capture_output=True,
            text=True, timeout=10,
        )

    def test_both_consumers_retain_after_identity_check_before_maturin(self) -> None:
        native = matching_step(
            self.jobs["native-prerequisites"], "name",
            "Compile the real TLS prerequisite in the wheel's ABI image",
        )
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                job = self.jobs[consumer]
                restore = matching_step(job, "uses", "./source/.github/actions/restore-compiled")
                retain = matching_step(job, "name", "Retain matching compiled build scripts")
                build = matching_step(job, "name", "Build stable-ABI wheel")
                self.assertLess(job["steps"].index(restore), job["steps"].index(retain))
                self.assertLess(job["steps"].index(retain), job["steps"].index(build))
                self.assertEqual(restore["if"], PR_ONLY)
                self.assertEqual(retain["if"], PR_ONLY)
                self.assertEqual(retain["shell"], "bash")
                self.assertIn("set -euo pipefail", retain["run"])
                self.assertEqual(restore["with"]["profile"], "wheel-dev")
                self.assertEqual(restore["with"]["artifact"], "wheel-prerequisites-${{ github.sha }}")
                self.assertEqual(job["needs"], ["sources", "medium-prerequisites"])
                self.assertEqual(build["uses"], native["uses"])
                self.assertEqual(build["with"]["rust-toolchain"], native["with"]["rust-toolchain"])
                self.assertIn(native["with"]["container"], build["with"]["container"])
                self.assertIn("--locked", build["with"]["args"])
                self.assertIn("--target-dir {0}/target", build["with"]["args"])
                for step in (restore, retain, build):
                    self.assertNotIn("continue-on-error", step)
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})

    def test_regression_is_required_by_wheel_preflight_and_path_filters(self) -> None:
        test_path = "scripts/tests/test_python_wheel_artifacts.py"
        for event in ("push", "pull_request"):
            with self.subTest(event=event):
                self.assertIn(test_path, self.workflow["on"][event]["paths"])
        preflight = matching_step(
            self.jobs["native-prerequisites"], "name", "Verify wheel build-script handoff",
        )
        self.assertIn("set -euo pipefail", preflight["run"])
        self.assertIn("test_python_wheel_artifacts.py", preflight["run"])
        self.assertNotIn("continue-on-error", preflight)

    def test_round_trip_preserves_compiled_contents_modes_and_links(self) -> None:
        self.assertEqual(self.build_output, "cargo:rerun-if-changed=src/\n")
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer), tempfile.TemporaryDirectory() as temporary:
                workspace = Path(temporary) / "runner workspace"
                artifacts = workspace / "target/debug/build"
                package = artifacts / "kwavers"
                package.mkdir(parents=True)
                (workspace / ".ci-cargo").mkdir()
                binary = package / "build-script-build"
                shutil.copy2(self.build_script, binary)
                (package / "output").write_text(self.build_output)
                (package / "build-script-link").symlink_to(binary.name)
                expected = tree_identity(artifacts)
                retain, restore = self.commands(consumer, workspace)
                result = self.execute(retain, workspace)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(tree_identity(workspace / ".ci-cargo/wheel-build-scripts"), expected)
                # Reproduce the pinned Maturin action's pre-container cleanup.
                shutil.rmtree(artifacts)
                result = self.execute(restore, workspace)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(tree_identity(artifacts), expected)
                self.assertEqual(
                    subprocess.check_output([str(binary)], text=True, timeout=5),
                    self.build_output,
                )

    def test_missing_compiled_build_scripts_fail_retention(self) -> None:
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer), tempfile.TemporaryDirectory() as temporary:
                workspace = Path(temporary)
                (workspace / ".ci-cargo").mkdir()
                retain, _ = self.commands(consumer, workspace)
                result = self.execute(retain, workspace)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("target/debug/build", result.stderr)
                self.assertFalse((workspace / ".ci-cargo/wheel-build-scripts").exists())

    def test_missing_retained_build_scripts_fail_restoration(self) -> None:
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer), tempfile.TemporaryDirectory() as temporary:
                workspace = Path(temporary)
                _, restore = self.commands(consumer, workspace)
                result = self.execute(restore, workspace)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("wheel-build-scripts", result.stderr)
                self.assertEqual(tree_identity(workspace / "target/debug/build"), {})


if __name__ == "__main__":
    unittest.main()

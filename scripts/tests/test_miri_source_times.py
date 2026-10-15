"""Compiler inputs retain real timestamps only after complete content checks."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("source_times", ROOT / "scripts/miri_source_times.py")
TIMES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TIMES)


class MiriSourceTimesTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="kwavers-miri-inputs-")
        self.addCleanup(temporary.cleanup)
        self.repository = Path(temporary.name)
        paths = ["Cargo.toml", "Cargo.lock", "crates/kwavers/build.rs", "crates/kwavers/src/lib.rs",
                 "crates/kwavers-solver/build.rs",
                 "crates/kwavers-solver/src/inverse/fwi/frequency_domain/born_kernel/arithmetic/operations.rs.in"]
        for name in paths:
            destination = self.repository / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        for command in (["init", "--quiet"], ["add", "."],
                        ["-c", "user.name=Compiler input verification", "-c", "user.email=verification@example.invalid",
                         "commit", "--quiet", "-m", "Retain actual production compiler inputs"]):
            subprocess.run(["git", *command], cwd=self.repository, check=True, capture_output=True, timeout=15)
        self.saved = TIMES.record(self.repository)
        for name, entry in self.saved["files"].items():
            os.utime(self.repository / name, ns=(entry["mtime_ns"], entry["mtime_ns"] + 1_000_000_000))
        for name, timestamp in self.saved["directories"].items():
            os.utime(self.repository / name, ns=(timestamp, timestamp + 1_000_000_000))

    def times(self) -> dict:
        return {name: (self.repository / name).stat().st_mtime_ns
                for name in (*self.saved["files"], *self.saved["directories"])}

    def test_json_handoff_restores_exact_contents_and_nanosecond_times(self) -> None:
        retained = json.loads(json.dumps(self.saved))
        TIMES.restore(self.repository, retained)
        self.assertEqual(len(retained["files"]), 6)
        for name, entry in retained["files"].items():
            path = self.repository / name
            self.assertEqual(path.stat().st_mtime_ns, entry["mtime_ns"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
        self.assertEqual({name: (self.repository / name).stat().st_mtime_ns for name in retained["directories"]}, retained["directories"])

    def test_content_mismatch_fails_before_any_timestamp_write(self) -> None:
        source = self.repository / "crates/kwavers/build.rs"
        source.write_bytes(source.read_bytes() + b"\n")
        before = self.times()
        with self.assertRaisesRegex(ValueError, "compiler inputs differ from HEAD"):
            TIMES.restore(self.repository, self.saved)
        self.assertEqual(self.times(), before)

    def test_manifest_digest_mismatch_fails_before_any_timestamp_write(self) -> None:
        bad = copy.deepcopy(self.saved)
        bad["files"]["Cargo.toml"]["sha256"] = hashlib.sha256((self.repository / "Cargo.lock").read_bytes()).hexdigest()
        before = self.times()
        with self.assertRaisesRegex(ValueError, "content differs"):
            TIMES.restore(self.repository, bad)
        self.assertEqual(self.times(), before)

    def test_incomplete_extra_and_invalid_metadata_are_atomic_failures(self) -> None:
        variants = []
        bad = copy.deepcopy(self.saved); del bad["files"]["Cargo.lock"]; variants.append(bad)
        bad = copy.deepcopy(self.saved); bad["directories"]["../outside"] = 0; variants.append(bad)
        bad = copy.deepcopy(self.saved); bad["files"]["Cargo.toml"]["mtime_ns"] = -1; variants.append(bad)
        bad = copy.deepcopy(self.saved); bad["files"]["Cargo.toml"]["mtime_ns"] = 1 << 63; variants.append(bad)
        bad = copy.deepcopy(self.saved); bad["directories"]["crates"] = True; variants.append(bad)
        for bad in variants:
            before = self.times()
            with self.subTest(metadata=bad):
                with self.assertRaises(ValueError):
                    TIMES.restore(self.repository, bad)
                self.assertEqual(self.times(), before)

    def test_actual_miri_archive_retains_content_bound_source_times(self) -> None:
        scripts = self.repository / "scripts"
        scripts.mkdir()
        shutil.copy2(ROOT / "scripts/miri_source_times.py", scripts / "miri_source_times.py")
        actions = []
        for name in ("prepare-compiled", "restore-compiled"):
            action = yaml.load((ROOT / f".github/actions/{name}/action.yml").read_text(), Loader=yaml.BaseLoader)
            actions.append(next(step["run"] for step in action["runs"]["steps"] if step.get("shell") == "bash"))
        target = self.repository / "target/debug/build/kwavers/build-script-build"
        target.parent.mkdir(parents=True)
        subprocess.run(["rustc", "+nightly", str(self.repository / "crates/kwavers/build.rs"), "-o", str(target)],
                       check=True, capture_output=True, timeout=30)
        binary = target.read_bytes()
        runner = self.repository / "runner-temp"
        download = runner / "compiled-download"
        download.mkdir(parents=True)
        environment = dict(os.environ, SOURCE_REPOSITORY=str(self.repository), BUILD_PROFILE="miri", RUNNER_TEMP=str(runner))
        def execute(command: str) -> subprocess.CompletedProcess:
            return subprocess.run(["bash", "-c", command], cwd=self.repository, env=environment,
                                  capture_output=True, text=True, timeout=30)
        result = execute(actions[0])
        self.assertEqual(result.returncode, 0, result.stderr)
        info = json.loads((self.repository / "compiled-info.json").read_text())
        self.assertEqual(info["source_times"], TIMES.record(self.repository))
        archive = "compiled-prerequisites.tar.zst"
        shutil.copy2(runner / archive, download / archive)
        for name, entry in info["source_times"]["files"].items():
            os.utime(self.repository / name, ns=(entry["mtime_ns"], entry["mtime_ns"] + 1_000_000_000))
        target.unlink()
        result = execute(actions[1])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(target.read_bytes(), binary)
        self.assertEqual(subprocess.check_output([str(target)], text=True, timeout=5), "cargo:rerun-if-changed=src/\n")
        self.assertEqual(TIMES.record(self.repository), info["source_times"])
        # Same revision/compiler/profile must still reject a corrupt source digest.
        info["source_times"]["files"]["Cargo.toml"]["sha256"] = hashlib.sha256(binary).hexdigest()
        (self.repository / "compiled-info.json").write_text(json.dumps(info))
        result = execute("set -euo pipefail\n" + actions[0].splitlines()[-1])
        self.assertEqual(result.returncode, 0, result.stderr)
        shutil.copy2(runner / archive, download / archive)
        before = self.times()
        result = execute(actions[1])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("compiler input content differs: Cargo.toml", result.stderr)
        self.assertEqual(self.times(), before)


if __name__ == "__main__":
    unittest.main()

"""Computed identity and integrity contracts using a real k-Wave reference NPZ."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("reference_cache", ROOT / "scripts/kwave_reference_cache.py")
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)
REFERENCE = ROOT / "crates/kwavers/tests/reference/kwave/ivp_homogeneous_3d.npz"


class ReferenceCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="kwave-reference-contract-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.key = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
        self.result = self.directory / (self.key + ".npz")
        shutil.copyfile(REFERENCE, self.result)
        self.identity = CACHE.identity({"reference": REFERENCE}, {"num_threads": 4, "version": "0.6.2"})

    def test_identity_is_content_and_environment_sensitive(self) -> None:
        original = CACHE.identity({"reference": self.result}, {"version": "0.6.2", "num_threads": 4})
        self.assertEqual(original, self.identity)
        for field, value in (("version", "0.6.3"), ("num_threads", 2)):
            environment = {"version": "0.6.2", "num_threads": 4, field: value}
            self.assertNotEqual(CACHE.identity({"reference": self.result}, environment), original)
        self.result.write_bytes(self.result.read_bytes() + b"\n")
        self.assertNotEqual(
            CACHE.identity({"reference": self.result}, {"version": "0.6.2", "num_threads": 4}), original,
        )

    def test_real_reference_round_trip_has_exact_computed_manifest(self) -> None:
        CACHE.seal(self.directory, self.identity)
        expected = {"identity": self.identity, "files": {self.result.name: hashlib.sha256(REFERENCE.read_bytes()).hexdigest()}}
        self.assertEqual(json.loads((self.directory / CACHE.MANIFEST).read_text()), expected)
        CACHE.verify(self.directory, self.identity)
        self.assertEqual(self.result.read_bytes(), REFERENCE.read_bytes())
        self.assertEqual(CACHE.snapshot(self.directory), expected["files"])

    def test_modified_reference_is_rejected_before_reuse(self) -> None:
        CACHE.seal(self.directory, self.identity)
        self.result.write_bytes(self.result.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "provenance or contents differ"):
            CACHE.verify(self.directory, self.identity)

    def test_different_environment_is_rejected_before_reuse(self) -> None:
        CACHE.seal(self.directory, self.identity)
        with self.assertRaisesRegex(ValueError, "provenance or contents differ"):
            CACHE.verify(self.directory, "f" * 64)

    def test_missing_or_extra_results_are_rejected(self) -> None:
        CACHE.seal(self.directory, self.identity)
        extra = self.directory / ("e" * 64 + ".npz")
        shutil.copyfile(REFERENCE, extra)
        with self.assertRaisesRegex(ValueError, "provenance or contents differ"):
            CACHE.verify(self.directory, self.identity)
        extra.unlink()
        self.result.unlink()
        with self.assertRaisesRegex(ValueError, "no numerical results"):
            CACHE.verify(self.directory, self.identity)

    def test_unsealed_symlink_and_nonreference_entries_are_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            CACHE.verify(self.directory, self.identity)
        link = self.directory / CACHE.MANIFEST
        link.symlink_to(REFERENCE)
        with self.assertRaisesRegex(ValueError, "symlink"):
            CACHE.verify(self.directory, self.identity)
        link.unlink()
        (self.directory / "current-kwavers-output.npz").write_bytes(REFERENCE.read_bytes())
        with self.assertRaisesRegex(ValueError, "Unexpected reference cache entry"):
            CACHE.seal(self.directory, self.identity)

    def test_workflow_retains_fresh_assertions_and_exact_reference_namespace(self) -> None:
        workflow = yaml.load((ROOT / ".github/workflows/python-wheel-smoke.yml").read_text(), Loader=yaml.BaseLoader)
        steps = workflow["jobs"]["kwave-comparison"]["steps"]
        by_name = {step.get("name"): step for step in steps}
        restore = by_name["Restore k-Wave reference results"]
        self.assertEqual(restore["uses"], "actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9")
        self.assertEqual(restore["with"]["path"], "kwave_cache")
        self.assertEqual(restore["with"]["key"], "kwave-reference-${{ runner.os }}-${{ runner.arch }}-${{ steps.kwave-reference.outputs.identity }}")
        self.assertNotIn("restore-keys", restore["with"])
        self.assertNotIn("if", by_name["Build stable-ABI wheel"])
        parity = by_name["Run value-semantic k-Wave parity suite"]
        self.assertNotIn("if", parity)
        self.assertIn("--capture=tee-sys", parity["run"])
        order = [step.get("name") for step in steps]
        self.assertLess(order.index("Identify the k-Wave reference environment"), order.index("Restore k-Wave reference results"))
        self.assertLess(order.index("Verify restored k-Wave reference results"), order.index(parity["name"]))
        self.assertLess(order.index(parity["name"]), order.index("Seal validated k-Wave reference results"))
        for event in ("pull_request", "push"):
            self.assertIn("scripts/kwave_reference_cache.py", workflow["on"][event]["paths"])
            self.assertIn("scripts/tests/test_kwave_reference_cache.py", workflow["on"][event]["paths"])
        self.assertIn('"k-wave-python==0.6.2"', by_name["Install wheel and k-Wave comparison dependency"]["run"])
        preflight = [step for step in workflow["jobs"]["native-prerequisites"]["steps"]
                     if step.get("name") == "Verify wheel build-script handoff"]
        self.assertEqual(len(preflight), 1)
        self.assertIn("test_kwave_reference_cache.py", preflight[0]["run"])


if __name__ == "__main__":
    unittest.main()

"""Artifact transport uses a real wheel; hosted acceptance uses built Kwavers."""

import ensurepip
import json
from pathlib import Path
import shutil
import sys
import sysconfig
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import wheel_handoff


class WheelHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="wheel-handoff-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        # Use Python's declared wheel source: distributions can keep it in
        # WHEEL_PKG_DIR instead of ensurepip's bundled directory. Both contain
        # complete real wheels; transport never substitutes package behavior.
        wheel_directory = sysconfig.get_config_var("WHEEL_PKG_DIR")
        origin = Path(wheel_directory) if wheel_directory else Path(ensurepip.__file__).parent / "_bundled"
        wheels = list(origin.glob("pip-*.whl"))
        self.assertEqual(len(wheels), 1)
        self.wheel = self.directory / wheels[0].name
        shutil.copyfile(wheels[0], self.wheel)
        with zipfile.ZipFile(self.wheel) as archive:
            self.assertTrue(any(name.endswith(".dist-info/WHEEL") for name in archive.namelist()))
        self.identity = wheel_handoff.describe(ROOT, self.directory)

    def test_complete_wheel_and_source_identity_survive_transport(self):
        encoded = json.loads(json.dumps(self.identity))
        destination = self.directory / "downloaded"
        destination.mkdir()
        shutil.copyfile(self.wheel, destination / self.wheel.name)
        actual = wheel_handoff.verify(ROOT, destination, encoded)
        self.assertEqual(actual.read_bytes(), self.wheel.read_bytes())

    def test_changed_head_lock_digest_and_wheel_bytes_are_rejected(self):
        for key in ("head", "lock_sha256", "wheel_sha256"):
            with self.subTest(key=key):
                changed = dict(self.identity)
                first = "1" if changed[key][0] != "1" else "2"
                changed[key] = first + changed[key][1:]
                with self.assertRaisesRegex(ValueError, key):
                    wheel_handoff.verify(ROOT, self.directory, changed)
        self.wheel.write_bytes(self.wheel.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "wheel_sha256"):
            wheel_handoff.verify(ROOT, self.directory, self.identity)

    def test_missing_extra_and_symlink_artifacts_are_rejected(self):
        original = self.wheel.read_bytes()
        self.wheel.unlink()
        with self.assertRaisesRegex(ValueError, "one regular wheel"):
            wheel_handoff.verify(ROOT, self.directory, self.identity)
        source = self.directory.parent / (self.directory.name + ".whl")
        self.addCleanup(source.unlink, missing_ok=True)
        source.write_bytes(original)
        self.wheel.symlink_to(source)
        with self.assertRaisesRegex(ValueError, "one regular wheel"):
            wheel_handoff.verify(ROOT, self.directory, self.identity)
        self.wheel.unlink()
        self.wheel.write_bytes(original)
        (self.directory / "unexpected.txt").write_text("unexpected artifact")
        with self.assertRaisesRegex(ValueError, "one regular wheel"):
            wheel_handoff.verify(ROOT, self.directory, self.identity)

    def test_identity_schema_and_traversal_are_rejected(self):
        for changed in [
            dict(self.identity, filename="../" + self.wheel.name),
            dict(self.identity, wheel_sha256="invalid"),
            dict(self.identity, extra="unrecognized"),
        ]:
            with self.subTest(identity=changed):
                with self.assertRaises(ValueError):
                    wheel_handoff.verify(ROOT, self.directory, changed)


if __name__ == "__main__":
    unittest.main()

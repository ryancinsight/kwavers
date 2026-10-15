"""The installed-wheel consumer must execute after a verified producer."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]


class WheelValidationWorkflowTests(unittest.TestCase):
    def test_comparison_uses_a_checked_wheel_handoff_and_its_own_bound(self):
        jobs = yaml.load((ROOT / ".github/workflows/python-wheel-smoke.yml").read_text(),
                         Loader=yaml.BaseLoader)["jobs"]
        producer = jobs["kwave-comparison"]
        consumer = jobs["kwave-validation"]
        self.assertEqual(producer["timeout-minutes"], "${{ github.event_name == 'pull_request' && 5 || 30 }}")
        self.assertEqual(consumer["timeout-minutes"], producer["timeout-minutes"])
        self.assertEqual(consumer["needs"], "kwave-comparison")
        self.assertEqual(producer["outputs"]["identity"], "${{ steps.wheel-identity.outputs.identity }}")
        self.assertEqual(producer["outputs"]["artifact-id"], "${{ steps.wheel-upload.outputs.artifact-id }}")
        self.assertFalse(any("pytest" in step.get("run", "") for step in producer["steps"]))
        steps = consumer["steps"]
        download = next(s for s in steps if s.get("uses", "").startswith("actions/download-artifact@"))
        self.assertEqual(download["with"]["artifact-ids"], "${{ needs.kwave-comparison.outputs.artifact-id }}")
        self.assertEqual(download["with"]["digest-mismatch"], "error")
        verify = next(s for s in steps if s.get("name") == "Verify the complete built wheel")
        install = next(s for s in steps if s.get("name") == "Install wheel and k-Wave comparison dependency")
        self.assertLess(steps.index(download), steps.index(verify))
        self.assertLess(steps.index(verify), steps.index(install))
        self.assertEqual(verify["env"]["WHEEL_IDENTITY"], "${{ needs.kwave-comparison.outputs.identity }}")
        self.assertIn("wheel_handoff.py verify", verify["run"])
        commands = "\n".join(s.get("run", "") for s in steps)
        for suite in ("test_quantity_boundary.py", "test_kwave_comparison.py", "test_runtime_export_inventory.py"):
            self.assertIn(suite, commands)
        for step in steps:
            self.assertNotIn("continue-on-error", step)
            self.assertFalse(step.get("uses", "").startswith("PyO3/maturin-action@"))


if __name__ == "__main__":
    unittest.main()

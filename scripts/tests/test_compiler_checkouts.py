"""Bounded compiler jobs retain complete inputs without unrelated datasets."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
INPUTS = (".cargo", ".config", ".github", "crates", "scripts", "xtask")
JOBS = {
    "bounded-compile.yml": ("native", "foundation", "medium", "graphics"),
    "bounded-wheel-compile.yml": ("compile",),
    "benchmark-regression.yml": ("benchmark-compile",),
    "python-wheel-smoke.yml": ("native-prerequisites", "base-wheel", "kwave-comparison"),
    "ci.yml": ("full-feature-doctests",),
}


class CompilerCheckoutTests(unittest.TestCase):
    def test_every_bounded_compiler_retains_the_complete_workspace_inputs(self) -> None:
        for workflow, names in JOBS.items():
            jobs = yaml.load((ROOT / ".github/workflows" / workflow).read_text(), Loader=yaml.BaseLoader)["jobs"]
            for name in names:
                with self.subTest(workflow=workflow, job=name):
                    checkouts = [s for s in jobs[name]["steps"] if s.get("uses", "").startswith("actions/checkout@")]
                    self.assertEqual(len(checkouts), 1)
                    checkout = checkouts[0]
                    self.assertEqual(checkout["uses"].split("@", 1)[1], "3d3c42e5aac5ba805825da76410c181273ba90b1")
                    self.assertEqual(checkout["with"]["ref"], "${{ github.event.pull_request.head.sha || github.sha }}")
                    self.assertEqual(tuple(checkout["with"]["sparse-checkout"].split()), INPUTS)
                    self.assertNotIn("filter", checkout["with"])
                    self.assertNotIn("continue-on-error", checkout)


if __name__ == "__main__":
    unittest.main()

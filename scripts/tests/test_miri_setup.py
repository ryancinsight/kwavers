"""Miri preparation retains complete inputs and validates a compiler-keyed sysroot.

Sparse checkout changes file transport only: every workspace package, its full
source/test tree, and every literal compile-time include remains byte-identical.
The cache never substitutes for sysroot validation or any inventoried test.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
ACTION = ".github/actions/setup-miri/action.yml"
CACHE = "actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9"
INPUTS = (".cargo", ".config", ".github", "crates", "scripts", "xtask")


def load(relative: str) -> dict:
    return yaml.load((ROOT / relative).read_text(), Loader=yaml.BaseLoader)


class MiriSetupTests(unittest.TestCase):
    def test_both_consumers_keep_the_exact_inventory_and_bounds(self) -> None:
        jobs = load(".github/workflows/ci.yml")["jobs"]
        for name in ("miri-preparation", "memory-safety"):
            with self.subTest(job=name):
                job = jobs[name]
                self.assertEqual(job["timeout-minutes"], "${{ github.event_name == 'pull_request' && 15 || 30 }}")
                checkout = job["steps"][0]
                self.assertEqual(checkout["with"]["ref"], "${{ github.event.pull_request.head.sha || github.sha }}")
                self.assertEqual(tuple(checkout["with"]["sparse-checkout"].split()), INPUTS)
                self.assertNotIn("filter", checkout["with"])
                setup = [i for i, step in enumerate(job["steps"]) if step.get("uses") == "./.github/actions/setup-miri"]
                self.assertEqual(len(setup), 1)
                self.assertLess(setup[0], next(i for i, step in enumerate(job["steps"]) if "scripts/miri_gate.py" in step.get("run", "")))
                for step in job["steps"]:
                    self.assertNotIn("continue-on-error", step)
                    self.assertNotIn("MIRIFLAGS", step.get("env", {}))
        matrix = jobs["memory-safety"]
        self.assertEqual(matrix["strategy"], {"fail-fast": "false", "matrix": "${{ fromJSON(needs.miri-preparation.outputs.matrix) }}"})
        execute = matrix["steps"][-1]
        self.assertEqual(execute["run"], 'python3 scripts/miri_gate.py --package kwavers --filter "$MIRI_CASE" --exact')
        self.assertEqual(execute["env"], {"MIRI_CASE": "${{ matrix.case }}"})
        self.assertIn("memory-safety", jobs["ci-gate"]["needs"])

    def test_real_sparse_checkout_preserves_every_workspace_input(self) -> None:
        checkout = load(".github/workflows/ci.yml")["jobs"]["memory-safety"]["steps"][0]
        patterns = checkout["with"]["sparse-checkout"].split()
        with tempfile.TemporaryDirectory(prefix="kwavers-miri-sparse-") as temporary:
            sparse = Path(temporary) / "checkout"
            def git(*args: str) -> str:
                result = subprocess.run(["git", *args], cwd=sparse if sparse.exists() else ROOT, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout
            # A shallow partial source cannot serve an ordinary local clone:
            # upload-pack would need blobs that sparse checkout never fetched.
            # Share its existing object store without transport or source edits.
            objects = Path(git("rev-parse", "--git-path", "objects").strip())
            if not objects.is_absolute():
                objects = ROOT / objects
            shallow = Path(git("rev-parse", "--git-path", "shallow").strip())
            if not shallow.is_absolute():
                shallow = ROOT / shallow
            head = git("rev-parse", "HEAD").strip()
            git("init", "--quiet", str(sparse))
            (sparse / ".git/objects/info/alternates").write_text(str(objects.resolve()) + "\n")
            if shallow.exists():
                shutil.copyfile(shallow, sparse / ".git/shallow")
            git("update-ref", "HEAD", head)
            git("sparse-checkout", "set", "--cone", *patterns)
            git("checkout", "--quiet", "HEAD")
            tracked = git("ls-tree", "-r", "--name-only", "HEAD").splitlines()
            selected = [p for p in tracked if "/" not in p or p.split("/")[0] in INPUTS]
            self.assertGreater(len(selected), 1000)
            objects = {line.split("\t", 1)[1]: line.split()[2] for line in git("ls-tree", "-r", "HEAD").splitlines()}
            computed = subprocess.check_output(
                ["git", "hash-object", "--stdin-paths"], input="\n".join(selected) + "\n",
                cwd=sparse, text=True, timeout=30,
            ).splitlines()
            self.assertEqual(computed, [objects[p] for p in selected])
            metadata = json.loads(subprocess.check_output(
                ["cargo", "metadata", "--locked", "--offline", "--no-deps", "--format-version", "1"],
                cwd=sparse, text=True, stderr=subprocess.PIPE, timeout=30,
                env=dict(os.environ, CARGO_HOME=str(Path(temporary) / "cargo-home")),
            ))
            self.assertEqual(len(metadata["workspace_members"]), len(metadata["packages"]))
            for package in metadata["packages"]:
                self.assertTrue(Path(package["manifest_path"]).is_file(), package["name"])
            includes = 0
            for relative in selected:
                source = sparse / relative
                if source.suffix != ".rs":
                    continue
                for include in re.findall(r'include_(?:str|bytes)!\(\s*"([^"\n]+)"', source.read_text()):
                    # One audit tool documents the directive in a string literal.
                    if source.name == "readme_audit.rs":
                        continue
                    included = source.parent / include
                    self.assertTrue(included.is_file(), f"{relative}: {include}")
                    includes += 1
            self.assertGreater(includes, 30)
            self.assertFalse((sparse / "data").exists())
            self.assertFalse((sparse / "docs").exists())

    def test_cache_is_exact_and_setup_always_validates_it(self) -> None:
        steps = load(ACTION)["runs"]["steps"]
        cache = next(step for step in steps if step.get("uses") == CACHE)
        self.assertEqual(cache["with"]["path"], "~/.cache/miri")
        self.assertNotIn("restore-keys", cache["with"])
        self.assertIn("steps.identity.outputs.digest", cache["with"]["key"])
        for path in ("Cargo.lock", ".cargo/config.toml", ACTION, ".github/workflows/ci.yml"):
            self.assertIn(path, cache["with"]["key"])
        self.assertEqual(steps[-1]["run"], "cargo +nightly miri setup")
        self.assertNotIn("if", steps[-1])
        for step in steps:
            self.assertNotIn("continue-on-error", step)
            self.assertNotIn("MIRI_SYSROOT", step.get("env", {}))

    def test_real_compiler_and_interpreter_determine_the_digest(self) -> None:
        identity = next(step for step in load(ACTION)["runs"]["steps"] if step.get("id") == "identity")
        with tempfile.TemporaryDirectory(prefix="kwavers-miri-identity-") as temporary:
            output = Path(temporary) / "output"
            environment = dict(os.environ, GITHUB_OUTPUT=str(output))
            result = subprocess.run(["bash", "-c", identity["run"]], cwd=ROOT, env=environment, text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            values = [subprocess.check_output(command, text=True, timeout=5) for command in (
                ["rustc", "+nightly", "-vV"], ["cargo", "+nightly", "miri", "--version"],
            )]
            self.assertEqual(output.read_text(), "digest=" + hashlib.sha256("".join(values).encode()).hexdigest() + "\n")


if __name__ == "__main__":
    unittest.main()

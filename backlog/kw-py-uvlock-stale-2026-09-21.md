<a id="kw-py-uvlock-stale-2026-09-21"></a>

## KW-PY-UVLOCK-STALE-2026-09-21 — The Python dev lockfile is stale against its own pyproject [patch] — todo

priority: verification; needs: none; scope: `crates/kwavers-python/{pyproject.toml,uv.lock}`

- **Finding.** `uv lock --check --directory crates/kwavers-python` fails against the committed `pyproject.toml` and `uv.lock`: uv resolves 129 packages for the floor the pyproject declares, against 77 for a 3.10 floor, so the lockfile did not match its project before any floor change.
- **Impact.** `uv sync` re-locks on this package; CI does not invoke uv, so no hosted gate is red.
- **Acceptance:** a `uv lock` whose diff is reviewed as a dependency change in its own right, or an explicit decision that this lockfile is not maintained here.
- **Next step:** run `uv lock` outside the stack overlay and review the diff.

# ADR 006 — Workspace architecture for PyO3 Python bindings

- **Status:** Implemented; partially superseded by [ADR 011](011-workspace-crate-split.md)
- **Date:** 2026-02-04 · **Audited:** 2026-06-03
- **Change class:** [arch]
- **Relates:** extended and partially superseded by [ADR 011](011-workspace-crate-split.md)

## Context

kwavers needs Python bindings for k-Wave / k-wave-python validation, the NumPy/
SciPy/Matplotlib ecosystem, and notebook workflows. The question was *where* the
bindings live. Embedding them as a `kwavers::python` module would leak PyO3 into
the core and violate bounded-context separation; a fully separate non-workspace
crate would lose shared profiles and lints.

## Decision

Adopt a Cargo **workspace** with the PyO3 bindings in a dedicated crate that
depends one-directionally on the core (`bindings → core`, never the reverse).
The binding crate is a thin presentation/infrastructure layer: it converts types,
maps Rust errors to Python exceptions, and releases the GIL around compute; it
holds no domain logic. PyO3 dependencies never enter the core crates.

## Current state (audited 2026-06-03)

Implemented, with the crate renamed and relocated since this ADR was written:

- The PyO3 crate is **`kwavers-python`** at `crates/kwavers-python/` (this ADR
  originally named it `pykwavers` at repo root). The name `pykwavers` survives as
  the **published Python module/package** only — `crates/kwavers-python/pyproject.toml`
  declares `name = "pykwavers"`, `module-name = "pykwavers._pykwavers"`, with the
  pure-Python helpers at `crates/kwavers-python/python/pykwavers/`.
- Dependency direction holds downward: no layer crate depends on `kwavers-python`.
- No PyO3 dependency appears in any domain/layer crate.
- **As of 2026-06-03, `kwavers-python` depends on the layer crates directly**
  (`kwavers-core`, `kwavers-domain`, `kwavers-solver`, …), not on a `kwavers`
  facade. The facade was removed (see [ADR 011 Amendment](011-workspace-crate-split.md));
  binding source uses `kwavers_<layer>::…` paths. The original
  `kwavers = { path = "../../kwavers" }` dependency no longer exists.

The original two-crate `members = ["kwavers", "pykwavers", "xtask"]` workspace was
superseded by the multi-crate workspace of
[ADR 011](011-workspace-crate-split.md): the layers are separate crates, and
`kwavers` is now only a thin top-level app/integration crate (no re-exports).

## Consequences

- Clean bounded-context separation; the core stays PyO3-free and testable.
- The "thin binding layer" rule is now codified project-wide (domain crates must
  not depend on `pyo3`).
- ADR superseded in part: the single-`kwavers`-core topology is replaced by the
  layered crate split; the workspace + thin-binding decision itself stands.

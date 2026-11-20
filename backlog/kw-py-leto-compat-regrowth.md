<a id="kw-py-leto-compat-regrowth"></a>

## KW-PY-LETO-COMPAT-REGROWTH — A local numpy-Leto conversion module serves 25 binding files, contradicting its closed gap record [patch] [arch] — verified locally

- Decision: follow Leto ADR 0017. `crates/kwavers-python/src/array_utils/owned.rs`
  owns the third-party boundary. The six rank-specific functions and the
  breast FWI module are removed. All callers use rank-inferred conversions.
- Evidence: six array-boundary tests pass with GPU and plotting bindings
  compiled. The regressions check allocation identity in both directions,
  reversed strides, an offset slice, Fortran storage, and an empty extent.

priority: architecture; needs: none; basis: 38eaf2f90eccd74d11ce3d0e836cbad7d4606ae0; scope: `crates/kwavers-python/src/breast_fwi_bindings/complex_compat.rs` and every file that imports it; milestone: M-imaging-capability

- **Finding.** `complex_compat.rs:4-35` defines six converters (`nd_to_leto1`, `nd_to_leto2`, `nd_to_leto3`, `leto1_to_nd1`, `leto2_to_nd2`, `leto3_to_nd3`), each ending in `.expect("valid shape")`. `git grep -E '\b(nd_to_leto[123]|leto[123]_to_nd[123])\b' origin/main -- crates` names them in 26 files, and 127 lines call one, in unrelated binding modules (`pam_bindings.rs`, `source_py/`, `simulation_py/`, `theranostic_bindings/`, `medium_py/`), all through `breast_fwi_bindings::complex_compat`.
- **Contradicted record.** `gap_audit.md:431-434` (closed 2026-07-12) says `nd_to_leto1` and `leto1_to_nd1` duplicated Leto's conversions and "All consumers now invoke the provider conversions directly". cdf8ec217 (2026-07-21) added `pub fn nd_to_leto1` back. The locked Leto (7bbfe9523, `Cargo.lock`) and Leto `origin/main` hold no ndarray conversion: `git grep -i ndarray -- '*.rs'` finds no definition, and Leto's root `Cargo.toml:36` records its compatibility dependencies removed per LETO-NDARRAY-BOUNDARY-1.
- **Acceptance:** one owner of the numpy-Leto conversion is decided and recorded, and the code and `gap_audit.md` agree: either Leto provides it and the module and its call sites are gone (the `git grep` above returns no hit), or it stays in `kwavers-python` under a boundary-named module outside `breast_fwi_bindings` whose `expect` messages begin `invariant:` and the record is corrected.
- `xtask legacy-migration-audit` reports zero legacy source references. The
  `owned` module name keeps all 25 binding callers distinct from external ndarray
  imports under the existing audit rule.

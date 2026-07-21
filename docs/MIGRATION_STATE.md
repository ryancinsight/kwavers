# kwavers Migration State — ndarray/nalgebra to Leto

## Overview

The migration from `ndarray`/`nalgebra` to `leto` is **~98% complete** in the Rust source code. 
This document tracks the remaining ndarray/nalgebra references and their resolution strategy.

## Current State

### kwavers Core (COMPLETE)
- All `ndarray`/`nalgebra` imports replaced with `leto` equivalents
- Linear algebra uses `leto_ops::solve`, `leto_ops::inv`, `leto_ops::symmetric_eigenvalues_jacobi`
- Array types: `Array1<T>`, `Array2<T>`, `Array3<T>` from `leto`
- Geometry: `leto::geometry::{Point3, Vector3, Isometry3, ...}` via `kwavers_core::scalar`

### kwavers-math (COMPLETE)
- FFT: Uses `apollo` + `leto` directly
- Linear algebra: `leto_ops` for all operations
- SIMD: `hermes-simd` via `simd_safe` module
- No ndarray/nalgebra imports

### kwavers-solver (COMPLETE)
- Forward solvers: FDTD, PSTD, elastic — all use `leto` arrays
- Inverse: FWI, PINN — all use `leto` and `coeus`
- Workspace: `Array1<T>` from `leto`

### kwavers-analysis (COMPLETE)
- Beamforming, signal processing, validation — all use `leto` arrays
- Example: `advanced_ultrasound_imaging.rs` uses `leto::{Array1, Array2, Array3}`

### kwavers-gpu (COMPLETE)
- Hephaestus-backed GPU backend
- Uses `leto` for array operations

## Remaining ndarray References (Boundary I/O)

### 1. NIfTI Example I/O (Expected — Not a Problem)

**Files:**
- `crates/kwavers/examples/liver_theranostic_reconstruction.rs`
- `crates/kwavers/examples/transcranial_ct_mri_reconstruction.rs`
- `crates/kwavers/examples/transcranial_fwi.rs`

**Pattern:**
```rust
use leto::{Array2, Array3};
// ...
let ct_vol: ArrayD<f64> = ct_obj.into_volume().into_ndarray::<f64>().ok()?;
```

**Resolution:** These are boundary I/O examples using the `nifti` crate. The `into_ndarray()` 
call returns ndarray's `ArrayD`, but the data is immediately used with leto operations.
This is acceptable — ndarray-compat provides zero-cost conversions.

### 2. Python Bindings (Expected — Not a Problem)

**Files:**
- `crates/kwavers-python/src/*.rs` (many files)

**Pattern:** Uses `numpy` crate for Python FFI boundary.

**Resolution:** Python interop is a boundary layer. The `numpy` crate wraps ndarray for 
Ffi boundary, and data flows into leto arrays via conversions.

### 3. Build Tooling (Expected — Not a Problem)

**Files:**
- `xtask/src/migration_audit.rs`
- `xtask/src/main.rs`

**Pattern:** Tooling that scans for ndarray/nalgebra usage patterns.

**Resolution:** Tooling references, not runtime dependencies. Can be updated when needed.

## Remaining ndarray/nalgebra in Cargo.lock

The lock file contains entries for:
- `ndarray` (version 0.16.1) — transitive dependency from other crates
- `burn-ndarray` (version 0.19.1) — transitive dependency from other crates

These are pulled in by crates like `coeus` (for neural network operations) and other 
transitive dependencies. They don't affect kwavers directly.

## Gaps to Close

### 1. ndarray-compat Feature (Optional)

**Status:** NOT enabled in kwavers workspace

**Impact:** Zero-cost conversions between leto and ndarray are available but not used.

**Resolution:** Enable `ndarray-compat` feature on leto for boundary I/O convenience.

### 2. Python Boundary (numpy crate)

**Status:** COMPLETED — Python bindings use crates.io `numpy` 0.29 with PyO3 0.29

**Impact:** The thin Python boundary follows one registry dependency closure.

**Resolution:** The obsolete vendored 0.27 source and patch are removed.

### 3. Transitive Dependencies

**Status:** Some crates still depend on ndarray/nalgebra transitively

**Impact:** None for kwavers itself.

**Resolution:** Monitor and update as upstream crates migrate.

## Migration Summary

| Component | ndarray | nalgebra | Status |
|-----------|---------|---------|--------|
| kwavers core | 0 | 0 | COMPLETE |
| kwavers-math | 0 | 0 | COMPLETE |
| kwavers-solver | 0 | 0 | COMPLETE |
| kwavers-analysis | 0 | 0 | COMPLETE |
| kwavers-gpu | 0 | 0 | COMPLETE |
| kwavers-python | numpy (FFI) | 0 | BOUNDARY |
| kwavers examples (NIfTI) | ArrayD (boundary) | 0 | BOUNDARY |
| kwavers-xtask | 0 | 0 | TOOLING |

## Verification

### Build Status
```bash
cargo build --lib  # Passes
cargo build --examples  # Passes (with nifti feature)
```

### Test Status
```bash
cargo test --lib  # Passes
cargo test --examples  # Passes
```

## Action Items

1. **Enable ndarray-compat** on leto for boundary I/O convenience (optional)
2. **Monitor** transitive ndarray/nalgebra dependencies for upstream updates
3. **Document** in book examples that NIfTI I/O is boundary layer

## Historical Notes

- Original ndarray usage: `Array1`, `Array2`, `Array3`, `ArrayD`, `DMatrix`, `DVector`
- Original nalgebra usage: `DMatrix`, `DVector`, `Vector2`, `Vector3`, `Point2`, `Point3`
- All replaced with: `leto::{Array1, Array2, Array3, ArrayD, ArrayView, ArrayViewMut}`
- Geometry replaced with: `leto::geometry::{Point2, Point3, Vector2, Vector3, ...}`
- Linear algebra replaced with: `leto_ops::{solve, inv, eigenvalues, svd, ...}`
- SIMD replaced with: `hermes-simd` via `simd_safe` module

## Related

- [gap_audit.md](gap_audit.md) — Detailed migration audit
- [backlog.md](backlog.md) — Feature/issue backlog
- [CHANGELOG.md](CHANGELOG.md) — Historical migration notes

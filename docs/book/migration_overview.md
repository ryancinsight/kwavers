# Chapter 35 — Migration Overview: ndarray/nalgebra → Leto

## Status: ~98% Complete

This chapter documents the migration of kwavers from `ndarray` and `nalgebra` to the 
Atlas stack crates, primarily `leto` for array operations and geometry, and 
`leto-ops` for linear algebra operations.

---

## What Was Migrated

### Array Types

| ndarray | Leto |
|---------|-------|
| `Array1<T>` | `leto::Array1<T>` |
| `Array2<T>` | `leto::Array2<T>` |
| `Array3<T>` | `leto::Array3<T>` |
| `ArrayD<T>` | `leto::ArrayD<T>` |
| `ArrayView<'a, T, D>` | `leto::ArrayView<'a, T, D>` |
| `ArrayViewMut<'a, T, D>` | `leto::ArrayViewMut<'a, T, D>` |

### Geometry Types

| nalgebra | Leto |
|----------|-------|
| `Vector2<T>` | `leto::geometry::Vector2<T>` |
| `Vector3<T>` | `leto::geometry::Vector3<T>` |
| `Point2<T>` | `leto::geometry::Point2<T>` |
| `Point3<T>` | `leto::geometry::Point3<T>` |
| `Isometry3<T>` | `leto::geometry::Isometry3<T>` |
| `Quaternion<T>` | `leto::geometry::Quaternion<T>` |
| `UnitQuaternion<T>` | `leto::geometry::UnitQuaternion<T>` |

### Linear Algebra Operations

| nalgebra | Leto-Ops |
|----------|----------|
| `matrix.solve(vector)` | `leto_ops::solve(&matrix.view(), &vector.view())` |
| `matrix.inv()` | `leto_ops::inv(&matrix.view())` |
| `matrix.eigenvalues()` | `leto_ops::eigenvalues(&matrix.view())` |
| `matrix.svd()` | `leto_ops::svd_decompose(&matrix.view())` |
| `matrix.rank()` | `leto_ops::rank(&matrix.view())` |
| `matrix.lu_decompose()` | `leto_ops::lu_decompose(&matrix.view())` |
| `matrix.qr_decompose()` | `leto_ops::qr_decompose(&matrix.view())` |
| `matrix.cholesky_decompose()` | `leto_ops::cholesky_decompose(&matrix.view())` |

### Sparse Matrices

| nalgebra-sparse | Leto-Ops |
|-----------------|----------|
| `CsrMatrix<T>` | `leto_ops::CsrMatrix<T>` |
| `CscMatrix<T>` | `leto_ops::CscMatrix<T>` |
| `CooMatrix<T>` | `leto_ops::CooMatrix<T>` |

---

## Migration Pattern

### Before (ndarray)

```rust
use ndarray::{Array1, Array2, Array3};
use nalgebra::DMatrix;

fn solve_system(a: &DMatrix<f64>, b: &Array1<f64>) -> Array1<f64> {
    let a_inv = a.pseudo_inverse(1e-10).unwrap();
    a_inv * b
}
```

### After (Leto)

```rust
use leto::{Array1, Array2, Array3};
use leto_ops::inv;

fn solve_system(a: &Array2<f64>, b: &Array1<f64>) -> KwaversResult<Array1<f64>> {
    let a_inv = inv(a)?;
    Ok(a_inv.view() * b.view())
}
```

### Before (nalgebra geometry)

```rust
use nalgebra::{Point3, Vector3};

let p = Point3::new(1.0, 2.0, 3.0);
let v = Vector3::new(0.0, 0.0, -1.0);
```

### After (Leto geometry)

```rust
use leto::geometry::{Point3, Vector3};

let p = Point3::new(1.0, 2.0, 3.0);
let v = Vector3::new(0.0, 0.0, -1.0);
```

---

## Why Leto?

### Design Principles

1. **Zero-Cost Abstractions**: Leto uses const generics and ZSTs for dimension encoding
2. **Zero-Sized Types**: Phantom dimensions are ZSTs, no runtime overhead
3. **Single Source of Truth**: All array operations flow through leto, not mixed backends
4. **Redundancy-Free**: No duplicated array logic across CPU/GPU
5. **Deep Hierarchy**: Organized by SRP (Single Responsibility Principle)
6. **Cow (Copy-on-Write)**: Optional CoW semantics for safe mutation patterns
7. **Phantom Types**: Dimension and type-level constraints without runtime cost
8. **GATS (Generic Associated Types)**: Flexible trait implementations

### Performance

Leto provides equivalent or better performance than ndarray:

- Compile-time dimension resolution
- Zero-cost storage abstractions
- SIMD-aware operations via hermes-simd integration
- GPU backends via hephaestus

---

## Migration Steps

### Step 1: Replace Array Imports

```rust
// Before
use ndarray::{Array1, Array2, Array3, ArrayD};

// After  
use leto::{Array1, Array2, Array3, ArrayD};
```

### Step 2: Replace Geometry Imports

```rust
// Before
use nalgebra::{Point3, Vector3, Isometry3};

// After
use leto::geometry::{Point3, Vector3, Isometry3};
```

### Step 3: Replace Linear Algebra Calls

```rust
// Before
let x = a.solve(&b, 1e-10)?;

// After
use leto_ops::solve;
let x = solve(a, b)?;
```

### Step 4: Update Array Construction

```rust
// Before
let a = Array2::zeros((10, 10));

// After (same API, different crate)
use leto::Array2;
let a = Array2::zeros([10, 10]);
```

---

## Compatibility

### ndarray-compat Feature

Leto provides an optional `ndarray-compat` feature for zero-cost conversions:

```toml
# In leto's Cargo.toml
[features]
ndarray-compat = ["dep:ndarray", "std"]
```

```rust
#[cfg(feature = "ndarray-compat")]
use ndarray_compat::*;

// Zero-cost conversions
let leto_array: leto::Array2<f64> = ndarray_array.into();
let ndarray_array: ndarray::Array2<f64> = leto_array.into();
```

### Python Boundary

The kwavers Python bindings use `numpy` for the FFI boundary. Data flows:

```text
numpy.ndarray (Python)
    → numpy (Rust, via pyo3)
    → leto::Array (via conversions)
    → kwavers computations
```

---

## Verification

All examples and tests pass with the Leto migration:

```bash
cargo build --lib          # Core library
cargo build --examples     # All examples
cargo test --lib           # Unit tests
cargo test --examples      # Example tests
```

---

## Remaining Boundary Cases

### 1. NIfTI I/O

Examples using NIfTI files (e.g., `liver_theranostic_reconstruction.rs`) use 
`nifti::IntoNdArray` for boundary I/O. This is acceptable:

- Data enters as ndarray via NIfTI crate
- Converted to leto arrays for computation
- Zero-cost conversion via ndarray-compat

### 2. Python FFI

Python bindings use `numpy` crate directly. This is the expected boundary:

- Python provides numpy arrays
- Converted to leto arrays for computation
- Results returned via pyo3

### 3. Transitive Dependencies

Some crates in the Atlas stack (e.g., `coeus`, `gaia`) still have ndarray/nalgebra 
in their lock files as transitive dependencies. This is not a problem for kwavers.

---

## Files Modified

The following files were modified during migration:

### kwavers Core
- `crates/kwavers-core/src/array.rs`
- `crates/kwavers-core/src/geometry.rs`
- `crates/kwavers-core/src/scalar.rs`

### kwavers Math
- `crates/kwavers-math/src/linear_algebra/ext.rs`
- `crates/kwavers-math/src/linear_algebra/mod.rs`
- `crates/kwavers-math/src/simd_safe/mod.rs`

### kwavers Solver
- `crates/kwavers-solver/src/forward/fdtd/solver.rs`
- `crates/kwavers-solver/src/forward/pstd/solver.rs`
- `crates/kwavers-solver/src/inverse/fwi/mod.rs`

### Examples
- `crates/kwavers/examples/advanced_ultrasound_imaging.rs`
- `crates/kwavers/examples/boundary_smoothing.rs`
- `crates/kwavers/examples/brain_theranostic_monitor.rs`

---

## See Also

- [Linear Algebra Migration](migration_linalg.md)
- [Geometry Migration](migration_geometry.md)
- [SIMD Migration](migration_simd.md)
- [Memory Migration](migration_memory.md)
- [Concurrency Migration](migration_concurrency.md)
- [FFT Migration](migration_fft.md)

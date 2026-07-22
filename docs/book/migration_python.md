# Chapter 42 — Python Integration: PyO3 and NumPy Boundary

kwavers exposes its Rust physics as a Python package (`pykwavers`) via PyO3.
The `numpy` crate bridges `PyArray*` types at the binding boundary.

## Architecture

```
Python numpy array
      │  (zero-copy for read, owned for write)
      ▼
PyReadonlyArray3<f64>  ──→  .as_slice()?  ──→  &[f64]
                                                  │
                                          leto::Array3<f64>
                                                  │
                                         kwavers simulation
                                                  │
                                          leto::Array3<f64>
      ▲                                           │
PyArray3<f64>  ◄──  from_vec(py, flat)  ◄──────────
```

## Reading from Python

```rust
use numpy::PyReadonlyArray3;
use pyo3::prelude::*;

#[pyfunction]
fn simulate(py: Python<'_>, data: PyReadonlyArray3<f64>) -> PyResult<…> {
    // Zero-copy borrow of the Python buffer
    let slice = data.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let array = leto::Array3::from_shape_vec(
        [data.shape()[0], data.shape()[1], data.shape()[2]],
        slice.to_vec(),
    ).unwrap();
    // … run simulation …
}
```

## Returning to Python (allocation-efficient)

The preferred pattern eliminates an intermediate `ndarray::Array1` allocation:

```rust
// Preferred (zero ndarray intermediate):
Ok(numpy::PyArray1::from_vec(py, result_vec).unbind())

// Preferred for 3D:
Ok(numpy::ToPyArray::to_pyarray(
    &leto_ops::csr_to_dense(&csr).into_raw_vec(),
    py).unbind())
```

## Complex numbers

`eunomia::Complex64` is the Atlas complex scalar. At the Python boundary it
is represented as numpy `complex128`:

```rust
use eunomia::Complex64;
use numpy::PyArray1;

let data: Vec<Complex64> = field.iter().map(|&z| z).collect();
// Convert to (re, im) interleaved f64 for numpy complex128
let flat: Vec<f64> = data.iter().flat_map(|z| [z.re, z.im]).collect();
```

## Module registration

```rust
#[pymodule]
fn pykwavers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    Ok(())
}
```

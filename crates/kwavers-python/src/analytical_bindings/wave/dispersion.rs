//! PyO3 bindings: numerical dispersion analysis (FDTD/PSTD phase error,
//! k-space correction error, CFL stability limit).

use kwavers_physics::analytical::wave;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute FDTD numerical phase error for a 1-D Yee grid.
///
/// Args:
///     kh: k*h (wave-number times grid spacing) array.
///     cfl: Courant–Friedrichs–Lewy number.
///
/// Returns:
///     Relative phase error array.
#[pyfunction]
#[pyo3(signature = (kh, cfl))]
pub fn fdtd_phase_error_1d(
    py: Python<'_>,
    kh: PyReadonlyArray1<f64>,
    cfl: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let kh_slice = kh
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::fdtd_phase_error_1d(kh_slice, cfl);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute PSTD numerical phase error.
///
/// Args:
///     kh: k*h array.
///
/// Returns:
///     Relative phase error array.
#[pyfunction]
#[pyo3(signature = (kh,))]
pub fn pstd_phase_error(py: Python<'_>, kh: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let kh_slice = kh
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::pstd_phase_error(kh_slice);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute k-space dispersion-correction phase error.
///
/// Args:
///     kh: k*h array.
///     cfl: CFL number.
///
/// Returns:
///     Residual phase error array after k-space correction.
#[pyfunction]
#[pyo3(signature = (kh, cfl))]
pub fn kspace_correction_error(
    py: Python<'_>,
    kh: PyReadonlyArray1<f64>,
    cfl: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let kh_slice = kh
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = wave::kspace_correction_error(kh_slice, cfl);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the FDTD CFL stability limit for an n-dimensional grid.
///
/// CFL_max = 1 / sqrt(ndim)
///
/// Args:
///     ndim: Number of spatial dimensions (1, 2, or 3).
///
/// Returns:
///     Maximum stable CFL number.
#[pyfunction]
#[pyo3(signature = (ndim,))]
pub fn fdtd_cfl_limit(ndim: u32) -> PyResult<f64> {
    Ok(wave::fdtd_cfl_limit(ndim))
}

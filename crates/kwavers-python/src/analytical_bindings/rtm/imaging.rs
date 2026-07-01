//! RTM imaging-condition and fusion bindings.

use super::arrays::{array2_from_flat, flatten_array2};
use kwavers_physics::analytical::rtm as rtm_mod;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Apply the zero-lag cross-correlation imaging condition for RTM.
///
/// image[i,j] = sum_t p_fwd[i,j,t] * p_bwd[i,j,t]
///
/// This version operates on single-frequency snapshots (no time axis):
/// image[i,j] = Re(p_fwd[i,j] * conj(p_bwd[i,j]))
///
/// Args:
///     p_fwd_real: Real part of forward-propagated field (nx × nz).
///     p_fwd_imag: Imaginary part.
///     p_bwd_real: Real part of back-propagated field (nx × nz).
///     p_bwd_imag: Imaginary part.
///     nx: Number of lateral grid points.
///     nz: Number of axial grid points.
///
/// Returns:
///     Image ndarray of shape (nx, nz).
#[pyfunction]
#[pyo3(signature = (p_fwd_real, p_fwd_imag, p_bwd_real, p_bwd_imag, nx, nz))]
pub fn rtm_imaging_condition(
    py: Python<'_>,
    p_fwd_real: PyReadonlyArray2<f64>,
    p_fwd_imag: PyReadonlyArray2<f64>,
    p_bwd_real: PyReadonlyArray2<f64>,
    p_bwd_imag: PyReadonlyArray2<f64>,
    nx: usize,
    nz: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let fr_flat = flatten_array2(p_fwd_real.as_array(), nx, nz);
    let fi_flat = flatten_array2(p_fwd_imag.as_array(), nx, nz);
    let br_flat = flatten_array2(p_bwd_real.as_array(), nx, nz);
    let bi_flat = flatten_array2(p_bwd_imag.as_array(), nx, nz);
    let flat = rtm_mod::rtm_imaging_condition(&fr_flat, &fi_flat, &br_flat, &bi_flat, nx, nz);
    array2_from_flat(py, nx, nz, flat)
}

/// Fuse multiple single-frequency RTM images by coherent averaging.
///
/// Args:
///     images: List of PyReadonlyArray2 images, each of shape (nx, nz).
///
/// Returns:
///     Fused image of shape (nx, nz).
#[pyfunction]
#[pyo3(signature = (images,))]
pub fn rtm_multi_frequency_fusion(
    py: Python<'_>,
    images: Vec<PyReadonlyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    if images.is_empty() {
        return Err(PyRuntimeError::new_err("images list must not be empty"));
    }
    let first = images[0].as_array();
    let (nx, nz) = first.dim();
    let vecs: Vec<Vec<f64>> = images
        .iter()
        .map(|img| flatten_array2(img.as_array(), nx, nz))
        .collect();
    let flat = rtm_mod::rtm_multi_frequency_fusion(&vecs);
    array2_from_flat(py, nx, nz, flat)
}

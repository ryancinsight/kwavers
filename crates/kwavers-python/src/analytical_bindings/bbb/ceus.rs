//! Contrast-enhanced ultrasound analytical bindings.

use kwavers_physics::analytical::bbb as bbb_mod;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// CEUS backscatter signal vs microbubble concentration (single-scatter + attenuation).
///
///     N_V   = c_mb × 10⁹  [m⁻³]
///     I_bs  = σ_bs · N_V · exp(−2 · 2·σ_bs · N_V · thickness)
///
/// Args:
///     c_mb_ul_ml: MB gas concentration [µL gas / mL tissue].
///     sigma_bs_m2: Backscatter cross-section per bubble [m²].
///     thickness_m: Tissue layer thickness [m].
///
/// Returns:
///     Backscatter signal [arbitrary units proportional to σ_bs].
///
/// Reference:
///     de Jong et al. (1991) Ultrasound Med. Biol. 17(2), 157–169.
#[pyfunction]
#[pyo3(signature = (c_mb_ul_ml, sigma_bs_m2, thickness_m))]
pub fn ceus_backscatter_signal(
    py: Python<'_>,
    c_mb_ul_ml: PyReadonlyArray1<f64>,
    sigma_bs_m2: f64,
    thickness_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let c_s = c_mb_ul_ml
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = bbb_mod::ceus_backscatter_signal(c_s, sigma_bs_m2, thickness_m);
    Ok(result.into_pyarray(py).unbind())
}

/// CEUS backscatter signal plus peak-normalised dB display payload.
#[pyfunction]
#[pyo3(signature = (c_mb_ul_ml, sigma_bs_m2, thickness_m, db_floor=-80.0))]
pub fn ceus_backscatter_display<'py>(
    py: Python<'py>,
    c_mb_ul_ml: PyReadonlyArray1<f64>,
    sigma_bs_m2: f64,
    thickness_m: f64,
    db_floor: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let c_s = c_mb_ul_ml
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let display = bbb_mod::ceus_backscatter_display(c_s, sigma_bs_m2, thickness_m, db_floor)
        .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("signal", display.signal.into_pyarray(py))?;
    out.set_item("signal_db", display.signal_db.into_pyarray(py))?;
    out.set_item("peak_concentration_ul_ml", display.peak_concentration_ul_ml)?;
    out.set_item("peak_signal", display.peak_signal)?;
    Ok(out)
}

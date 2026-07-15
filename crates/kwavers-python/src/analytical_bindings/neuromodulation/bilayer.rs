//! Bilayer-sonophore curve bindings.

use kwavers_physics::acoustics::therapy::neuromodulation::{
    bls_capacitance, quasistatic_deflection, rest_gap,
};
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Quasi-static leaflet deflection [m] versus acoustic pressure [Pa].
#[pyfunction]
#[pyo3(signature = (pressure_pa, v_rest_mv = -65.0, cm0_uf_cm2 = 1.0))]
pub fn bls_deflection_curve(
    py: Python<'_>,
    pressure_pa: PyReadonlyArray1<f64>,
    v_rest_mv: f64,
    cm0_uf_cm2: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p = pressure_pa
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let qm0 = (cm0_uf_cm2 * 1.0e-2) * (v_rest_mv * 1.0e-3);
    let delta = rest_gap(qm0);
    let out: Vec<f64> = p
        .iter()
        .map(|&pac| quasistatic_deflection(pac, qm0, delta))
        .collect();
    Ok(out.to_pyarray(py).unbind())
}

/// Curved-dome bilayer membrane capacitance C_m(Z) (Plaksin Eq. 8).
#[pyfunction]
#[pyo3(signature = (z_m, cm0_uf_cm2, radius_a_m, gap_delta_m))]
pub fn bilayer_capacitance_curve(
    py: Python<'_>,
    z_m: PyReadonlyArray1<f64>,
    cm0_uf_cm2: f64,
    radius_a_m: f64,
    gap_delta_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let z = z_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out: Vec<f64> = z
        .iter()
        .map(|&zi| bls_capacitance(zi, cm0_uf_cm2, radius_a_m, gap_delta_m))
        .collect();
    Ok(out.to_pyarray(py).unbind())
}

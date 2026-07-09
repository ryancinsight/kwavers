//! Temperature-coupled neural response model (Yoo et al. 2022).
//!
//! # Theorem
//!
//! Two-state approximation of temperature-dependent neural activation:
//!
//! ```text
//! d[Ca]/dt = −([Ca] − Ca_rest) / τ_ca + α_T · (T(t) − T_ref)
//! dV/dt    = −(V − V_rest) / τ_v   + k_ca · [Ca]
//! P_act    = σ((V − θ) / slope)    where σ(x) = 1/(1+e^{−x})
//! ```
//!
//! The thermal drive `T(t) − T_ref` couples acoustic heating to calcium
//! influx via `α_T`. Calcium drives membrane depolarization via `k_ca`.
//! The logistic sigmoid maps membrane potential to activation probability.
//!
//! Default constants match the thermal-coupling regime for tFUS at 0.5 MHz
//! (Yoo et al. 2022, Supplementary Table 1).
//!
//! # References
//!
//! - Yoo et al. (2022) Nature Neuroscience 25:1557

use kwavers_physics::acoustics::therapy::sonogenetics::{
    yoo_thermal_neural_response, ThermalNeuralParams,
};
use leto::Array1;
use numpy::{ToPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Solve a two-state temperature-coupled neural response model (Yoo et al. 2022).
///
/// Thin wrapper: the model physics (calcium/voltage ODE + logistic activation, RK4,
/// thermal-drive interpolation) lives in
/// `kwavers_physics::acoustics::therapy::sonogenetics::yoo_thermal_neural_response`
/// (single source of truth). Here we only marshal the optional thermal array and
/// delegate; default parameters match the tFUS 0.5 MHz regime (Yoo 2022 Suppl. Tbl 1).
///
/// Parameters
/// ----------
/// t_end_s : simulation end time [s].
/// dt_s : neural time step [s].
/// membrane_potential_mv_0 : initial membrane potential [mV] (default −70).
/// calcium_conc_um_0 : initial cytosolic calcium [µM] (default 0.1).
/// temperature_c_array : temperature time series [°C]; isothermal at 37 °C if None.
/// dt_thermal_s : time step of the thermal array [s].
///
/// Returns
/// -------
/// (time_s, voltage_mv, response_probability) as numpy arrays.
#[pyfunction]
#[pyo3(signature = (
    t_end_s, dt_s,
    membrane_potential_mv_0 = -70.0,
    calcium_conc_um_0 = 0.1,
    temperature_c_array = None,
    dt_thermal_s = 0.01,
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_hodgkin_huxley_like(
    py: Python<'_>,
    t_end_s: f64,
    dt_s: f64,
    membrane_potential_mv_0: f64,
    calcium_conc_um_0: f64,
    temperature_c_array: Option<PyReadonlyArray1<'_, f64>>,
    dt_thermal_s: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    if t_end_s <= 0.0 {
        return Err(PyValueError::new_err("t_end_s must be > 0"));
    }
    if dt_s <= 0.0 {
        return Err(PyValueError::new_err("dt_s must be > 0"));
    }
    if dt_thermal_s <= 0.0 {
        return Err(PyValueError::new_err("dt_thermal_s must be > 0"));
    }

    let thermal: Vec<f64> = match temperature_c_array {
        Some(arr) => arr.as_array().iter().copied().collect(),
        None => Vec::new(), // empty → physics treats as isothermal at the reference temp
    };
    let (time, voltage, response) = yoo_thermal_neural_response(
        t_end_s,
        dt_s,
        membrane_potential_mv_0,
        calcium_conc_um_0,
        &thermal,
        dt_thermal_s,
        &ThermalNeuralParams::default(),
    );
    Ok((
        Array1::from(time).to_pyarray(py).into(),
        Array1::from(voltage).to_pyarray(py).into(),
        Array1::from(response).to_pyarray(py).into(),
    ))
}


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

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const TAU_V_S: f64 = 0.010; // 10 ms membrane time constant
const K_CA: f64 = 0.05; // mV per µM calcium coupling
const TAU_CA_S: f64 = 0.100; // 100 ms calcium time constant
const ALPHA_T: f64 = 0.002; // µM/°C temperature-calcium coupling
const V_REST_MV: f64 = -70.0; // resting potential [mV]
const CA_REST_UM: f64 = 0.1; // resting calcium [µM]
const T_REF_C: f64 = 37.0; // reference temperature [°C]
const V_THRESH_MV: f64 = -59.0; // sigmoid midpoint [mV]
const V_SLOPE_MV: f64 = 2.8; // sigmoid slope [mV]

/// Solve a two-state temperature-coupled neural response model using RK4.
///
/// Models temperature-dependent calcium influx and membrane depolarisation
/// following Yoo et al. (2022) Nat. Neurosci. The thermal drive array is
/// linearly interpolated to the neural time grid.
///
/// Parameters
/// ----------
/// t_end_s : simulation end time [s].
/// dt_s : neural time step [s].
/// membrane_potential_mv_0 : initial membrane potential [mV].
/// calcium_conc_um_0 : initial cytosolic calcium [µM].
/// temperature_c_array : temperature time series [°C]; isothermal at 37 °C if None.
/// dt_thermal_s : time step of the thermal array [s].
///
/// Returns
/// -------
/// (time_s, voltage_mv, response_probability) as numpy arrays.
#[pyfunction]
#[pyo3(signature = (
    t_end_s, dt_s,
    membrane_potential_mv_0 = V_REST_MV,
    calcium_conc_um_0 = CA_REST_UM,
    temperature_c_array = None,
    dt_thermal_s = 0.01,
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_hodgkin_huxley_like<'py>(
    py: Python<'py>,
    t_end_s: f64,
    dt_s: f64,
    membrane_potential_mv_0: f64,
    calcium_conc_um_0: f64,
    temperature_c_array: Option<PyReadonlyArray1<'py, f64>>,
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

    let thermal_vec: Vec<f64> = match temperature_c_array {
        Some(arr) => arr.as_array().iter().copied().collect(),
        None => vec![T_REF_C],
    };

    let temp_at = |t: f64| -> f64 {
        if thermal_vec.len() == 1 {
            return thermal_vec[0];
        }
        let idx_f = t / dt_thermal_s;
        let idx_lo = (idx_f.floor() as usize).min(thermal_vec.len() - 1);
        let idx_hi = (idx_lo + 1).min(thermal_vec.len() - 1);
        if idx_lo == idx_hi {
            thermal_vec[idx_lo]
        } else {
            let frac = idx_f - idx_lo as f64;
            thermal_vec[idx_lo] * (1.0 - frac) + thermal_vec[idx_hi] * frac
        }
    };

    let rhs = |t: f64, ca: f64, v: f64| -> (f64, f64) {
        let temp = temp_at(t);
        let dca = -(ca - CA_REST_UM) / TAU_CA_S + ALPHA_T * (temp - T_REF_C);
        let dv = -(v - V_REST_MV) / TAU_V_S + K_CA * ca;
        (dca, dv)
    };

    let n_steps = (t_end_s / dt_s).ceil() as usize;
    let n_out = n_steps + 1;
    let mut time_arr = Array1::<f64>::zeros(n_out);
    let mut voltage_arr = Array1::<f64>::zeros(n_out);
    let mut response_arr = Array1::<f64>::zeros(n_out);

    let mut ca = calcium_conc_um_0;
    let mut v = membrane_potential_mv_0;

    let sigmoid = |vm: f64| 1.0 / (1.0 + (-(vm - V_THRESH_MV) / V_SLOPE_MV).exp());

    time_arr[0] = 0.0;
    voltage_arr[0] = v;
    response_arr[0] = sigmoid(v);

    for i in 1..n_out {
        let t_cur = (i - 1) as f64 * dt_s;

        let (dca1, dv1) = rhs(t_cur, ca, v);
        let (dca2, dv2) = rhs(
            t_cur + 0.5 * dt_s,
            ca + 0.5 * dt_s * dca1,
            v + 0.5 * dt_s * dv1,
        );
        let (dca3, dv3) = rhs(
            t_cur + 0.5 * dt_s,
            ca + 0.5 * dt_s * dca2,
            v + 0.5 * dt_s * dv2,
        );
        let (dca4, dv4) = rhs(t_cur + dt_s, ca + dt_s * dca3, v + dt_s * dv3);

        ca += dt_s / 6.0 * (dca1 + 2.0 * dca2 + 2.0 * dca3 + dca4);
        v += dt_s / 6.0 * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4);

        time_arr[i] = t_cur + dt_s;
        voltage_arr[i] = v;
        response_arr[i] = sigmoid(v);
    }

    Ok((
        time_arr.into_pyarray(py).into(),
        voltage_arr.into_pyarray(py).into(),
        response_arr.into_pyarray(py).into(),
    ))
}

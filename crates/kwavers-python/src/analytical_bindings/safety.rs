//! PyO3 bindings for `kwavers::physics::analytical::safety`.

use kwavers::physics::analytical::safety;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute the Mechanical Index (MI).
///
/// MI = |p_neg_pa| / (1e6 * sqrt(f_hz / 1e6))
///
/// Args:
///     p_neg_pa: Peak negative pressure [Pa].
///     f_hz: Frequency [Hz].
///
/// Returns:
///     Mechanical Index (dimensionless).
#[pyfunction]
#[pyo3(signature = (p_neg_pa, f_hz))]
pub fn mechanical_index(p_neg_pa: f64, f_hz: f64) -> PyResult<f64> {
    Ok(safety::mechanical_index(p_neg_pa, f_hz))
}

/// Compute the Mechanical Index over a pressure field (array variant).
///
/// Applies MI = |p_field[i]| / (1e6 * sqrt(f_hz / 1e6)) element-wise.
///
/// Args:
///     p_field: Peak rarefactional pressure field [Pa], passed as 1-D array.
///     f_hz: Centre frequency [Hz].
///
/// Returns:
///     MI array, same length as p_field (dimensionless).
///
/// Reference:
///     FDA Marketing Clearance of Diagnostic Ultrasound Systems, Appendix A.
#[pyfunction]
#[pyo3(signature = (p_field, f_hz))]
pub fn mechanical_index_field(
    py: Python<'_>,
    p_field: PyReadonlyArray1<f64>,
    f_hz: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p_s = p_field
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = safety::mechanical_index_field(p_s, f_hz);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the Thermal Index for soft tissue (TIS).
///
/// Args:
///     wstp_mw: W_STP — time-averaged power at the surface [mW].
///     f_mhz: Frequency [MHz].
///
/// Returns:
///     TIS value.
#[pyfunction]
#[pyo3(signature = (wstp_mw, f_mhz))]
pub fn thermal_index_soft_tissue(wstp_mw: f64, f_mhz: f64) -> PyResult<f64> {
    Ok(safety::thermal_index_soft_tissue(wstp_mw, f_mhz))
}

/// Compute the Thermal Index for bone (TIB).
///
/// Args:
///     w_mw: Beam power at bone surface [mW].
///     f_mhz: Frequency [MHz].
///
/// Returns:
///     TIB value.
#[pyfunction]
#[pyo3(signature = (w_mw, f_mhz))]
pub fn thermal_index_bone(w_mw: f64, f_mhz: f64) -> PyResult<f64> {
    Ok(safety::thermal_index_bone(w_mw, f_mhz))
}

/// Compute the cumulative CEM43 thermal dose over a temperature time series.
///
/// Args:
///     t_celsius: Temperature time series [°C].
///     dt_s: Time-step [s].
///
/// Returns:
///     Cumulative CEM43 array [min].
#[pyfunction]
#[pyo3(signature = (t_celsius, dt_s))]
pub fn cem43_cumulative(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
    dt_s: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = safety::cem43_cumulative(t_s, dt_s);
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the Arrhenius thermal-damage integral Ω.
///
/// Ω = A * ∫ exp(-Ea / (R*T(t))) dt
///
/// Args:
///     t_celsius: Temperature time series [°C].
///     dt_s: Time-step [s].
///     a_per_s: Pre-exponential frequency factor [1/s].
///     ea_j_mol: Activation energy [J/mol].
///
/// Returns:
///     Damage integral Ω (dimensionless) — total over entire time series.
#[pyfunction]
#[pyo3(signature = (t_celsius, dt_s, a_per_s, ea_j_mol))]
pub fn arrhenius_damage_integral(
    t_celsius: PyReadonlyArray1<f64>,
    dt_s: f64,
    a_per_s: f64,
    ea_j_mol: f64,
) -> PyResult<f64> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(safety::arrhenius_damage_integral(
        t_s, dt_s, a_per_s, ea_j_mol,
    ))
}

/// Compute the cumulative Arrhenius thermal-damage integral Ω(t).
///
/// Returns the running sum Ω(t_k) at each discrete step:
///   Ω(t_k) = A · Σ_{i=0}^{k} exp(−Ea / (R · T_K[i])) · dt
///
/// Output has the same length as t_celsius; element k is the total damage
/// from t=0 through t=k·dt_s.  Ω ≥ 1 indicates irreversible damage.
///
/// Args:
///     t_celsius: Temperature time series [°C].
///     dt_s: Time-step [s].
///     a_per_s: Pre-exponential frequency factor [1/s].
///     ea_j_mol: Activation energy [J/mol].
///
/// Returns:
///     Running Ω array [dimensionless], same length as t_celsius.
///
/// Reference:
///     Henriques & Moritz (1947) Am. J. Pathol. 23, 531.
#[pyfunction]
#[pyo3(signature = (t_celsius, dt_s, a_per_s, ea_j_mol))]
pub fn arrhenius_cumulative(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
    dt_s: f64,
    a_per_s: f64,
    ea_j_mol: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = safety::arrhenius_cumulative(t_s, dt_s, a_per_s, ea_j_mol);
    Ok(result.into_pyarray(py).unbind())
}

/// Return the FDA ISPTA diagnostic-ultrasound limit (720 mW/cm²).
///
/// Returns:
///     ISPTA limit [mW/cm²].
#[pyfunction]
pub fn fda_ispta_limit_mw_cm2() -> PyResult<f64> {
    Ok(safety::fda_ispta_limit_mw_cm2())
}

/// Return the FDA ISPPA diagnostic-ultrasound limit (190 W/cm²).
///
/// Returns:
///     ISPPA limit [W/cm²].
#[pyfunction]
pub fn fda_isppa_limit_w_cm2() -> PyResult<f64> {
    Ok(safety::fda_isppa_limit_w_cm2())
}

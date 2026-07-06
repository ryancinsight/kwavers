//! Arrhenius damage and combined kill-probability safety bindings.

use kwavers_physics::analytical::safety;
use numpy::{ToPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

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
    Ok(result.to_pyarray(py).unbind())
}

/// Cumulative thermal cell-death probability P_death(t) = 1 − exp(−Ω(t)).
///
/// Maps an Arrhenius thermal-damage history to a per-step kill probability; at
/// Ω = 1 this is the Henriques (1947) 63 % criterion (1 − e⁻¹ ≈ 0.632). Thin
/// wrapper over `kwavers_physics::analytical::safety::arrhenius_kill_probability`.
///
/// Args:
///     t_celsius: Temperature time series [°C].
///     dt_s: Time-step [s].
///     a_per_s: Pre-exponential frequency factor [1/s].
///     ea_j_mol: Activation energy [J/mol].
///
/// Returns:
///     Cumulative kill probability per step, same length as t_celsius, in [0, 1).
#[pyfunction]
#[pyo3(signature = (t_celsius, dt_s, a_per_s, ea_j_mol))]
pub fn arrhenius_kill_probability(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
    dt_s: f64,
    a_per_s: f64,
    ea_j_mol: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = safety::arrhenius_kill_probability(t_s, dt_s, a_per_s, ea_j_mol);
    Ok(result.to_pyarray(py).unbind())
}

/// Per-voxel thermal kill probability for a steady temperature held for a fixed
/// duration: P = 1 − exp(−A·exp(−Ea/(R·T_K))·duration). Each element of
/// `t_celsius` is an independent voxel. Field analogue of
/// `arrhenius_kill_probability`; thin wrapper over
/// `kwavers_physics::analytical::safety::arrhenius_steady_kill_probability`.
///
/// Args:
///     t_celsius: per-voxel steady temperature [°C].
///     duration_s: exposure duration [s].
///     a_per_s: pre-exponential frequency factor [1/s].
///     ea_j_mol: activation energy [J/mol].
///
/// Returns:
///     Per-voxel kill probability, same length as t_celsius, in [0, 1).
#[pyfunction]
#[pyo3(signature = (t_celsius, duration_s, a_per_s, ea_j_mol))]
pub fn arrhenius_steady_kill_probability(
    py: Python<'_>,
    t_celsius: PyReadonlyArray1<f64>,
    duration_s: f64,
    a_per_s: f64,
    ea_j_mol: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_celsius
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = safety::arrhenius_steady_kill_probability(t_s, duration_s, a_per_s, ea_j_mol);
    Ok(result.to_pyarray(py).unbind())
}

/// Combine independent mechanical and thermal kill probabilities into one
/// biologically-effective kill probability:
///   P_kill = 1 − (1 − P_mech)·(1 − P_thermal)
///
/// Element-wise over the shorter input length; each input clamped to [0, 1].
/// Thin wrapper over
/// `kwavers_physics::analytical::safety::combined_kill_probability`.
///
/// Args:
///     p_mech: Per-voxel mechanical (cavitation) kill probability [0, 1].
///     p_thermal: Per-voxel thermal kill probability [0, 1].
///
/// Returns:
///     Combined kill probability, in [0, 1].
#[pyfunction]
#[pyo3(signature = (p_mech, p_thermal))]
pub fn combined_kill_probability(
    py: Python<'_>,
    p_mech: PyReadonlyArray1<f64>,
    p_thermal: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let m = p_mech
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let t = p_thermal
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = safety::combined_kill_probability(m, t);
    Ok(result.to_pyarray(py).unbind())
}


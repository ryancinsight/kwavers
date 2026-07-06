//! Blood-brain-barrier permeability and closure bindings.

use kwavers_physics::analytical::bbb as bbb_mod;
use numpy::{ToPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// BBB permeability enhancement via Hill dose-response model.
///
///     P(D) = D^n / (D₅₀^n + D^n)
///
/// Args:
///     dose: Cumulative acoustic dose [arbitrary units, e.g. MI²·s].
///     d50: Dose at half-maximum permeability.
///     hill_n: Hill coefficient (dimensionless).
///
/// Returns:
///     Normalised permeability array in [0, 1].
///
/// Reference:
///     McDannold et al. (2008) Ultrasound Med. Biol. 34(6), 930–937.
#[pyfunction]
#[pyo3(signature = (dose, d50, hill_n))]
pub fn bbb_permeability_hill(
    py: Python<'_>,
    dose: PyReadonlyArray1<f64>,
    d50: f64,
    hill_n: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let d_s = dose
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = bbb_mod::bbb_permeability_hill(d_s, d50, hill_n);
    Ok(result.to_pyarray(py).unbind())
}

/// Inertial-cavitation damage probability via logistic dose response.
///
///     P_damage(D) = 1 / (1 + exp[-s · (D - D_thr)])
///
/// Args:
///     dose: Cumulative acoustic dose [arbitrary units, e.g. MI²·s].
///     damage_threshold: Dose at 50% damage probability.
///     slope: Logistic slope in reciprocal dose units.
///
/// Returns:
///     Damage probability array in [0, 1].
#[pyfunction]
#[pyo3(signature = (dose, damage_threshold, slope))]
pub fn bbb_inertial_damage_probability(
    py: Python<'_>,
    dose: PyReadonlyArray1<f64>,
    damage_threshold: f64,
    slope: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let d_s = dose
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = bbb_mod::bbb_inertial_damage_probability(d_s, damage_threshold, slope);
    Ok(result.to_pyarray(py).unbind())
}

/// BBB closure kinetics post-sonication: bi-exponential permeability decay.
///
///     P(t) = perm_peak · [0.6·exp(−t/τ_fast) + 0.4·exp(−t/τ_slow)]
///     τ_fast = 0.5·τ_close,  τ_slow = 3.0·τ_close
///
/// Args:
///     t_h: Time post-sonication [hours].
///     tau_close: Characteristic closing time constant [hours].
///     perm_peak: Peak permeability at t=0 (normalised, ≤ 1.0).
///
/// Returns:
///     Normalised permeability decay array.
///
/// Reference:
///     Deffieux & Konofagou (2010) Ultrasound Med. Biol. 36(7), 1117–1126.
#[pyfunction]
#[pyo3(signature = (t_h, tau_close, perm_peak))]
pub fn bbb_closure_kinetics(
    py: Python<'_>,
    t_h: PyReadonlyArray1<f64>,
    tau_close: f64,
    perm_peak: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let t_s = t_h
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = bbb_mod::bbb_closure_kinetics(t_s, tau_close, perm_peak);
    Ok(result.to_pyarray(py).unbind())
}

/// Bi-exponential post-sonication BBB closure (book §23.6) with explicit fast and
/// slow recovery time constants:
///
///     P(t) = P_peak · [0.6·e^(−t/τ_fast) + 0.4·e^(−t/τ_slow)]
///
/// The fast component (τ_fast ≈ 0.5 h) is tight-junction re-assembly; the slow
/// component (τ_slow ≈ 6 h) is vesicular-transport clearance (Deffieux &
/// Konofagou 2010). This is the function the chapter names — it sets τ_fast and
/// τ_slow independently (unlike `bbb_closure_kinetics`, which locks τ_slow =
/// 6·τ_fast).
///
/// Args:
///     t_h: Time post-sonication [h].
///     p_peak: Peak permeability enhancement at t = 0.
///     tau_fast_h: Fast recovery time constant [h].
///     tau_slow_h: Slow recovery time constant [h].
///
/// Returns:
///     Permeability enhancement P(t) over the time array.
#[pyfunction]
#[pyo3(signature = (t_h, p_peak, tau_fast_h, tau_slow_h))]
pub fn bbb_closure_permeability(
    py: Python<'_>,
    t_h: PyReadonlyArray1<f64>,
    p_peak: f64,
    tau_fast_h: f64,
    tau_slow_h: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    use kwavers_physics::acoustics::transcranial::bbb_opening::bbb_closure_permeability as closure;
    let t_s = t_h
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result: Vec<f64> = t_s
        .iter()
        .map(|&t| closure(t, p_peak, tau_fast_h, tau_slow_h))
        .collect();
    Ok(result.to_pyarray(py).unbind())
}


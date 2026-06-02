//! PyO3 bindings for `kwavers::physics::analytical::bbb`.

use kwavers::physics::analytical::bbb as bbb_mod;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
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
    Ok(result.into_pyarray(py).unbind())
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
    Ok(result.into_pyarray(py).unbind())
}

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

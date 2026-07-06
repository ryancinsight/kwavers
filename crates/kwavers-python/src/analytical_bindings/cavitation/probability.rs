//! Cavitation probability and threshold PyO3 wrappers.

use kwavers_physics::analytical::cavitation;
use numpy::{ToPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute the single-pulse intrinsic-threshold cavitation probability (Gaussian erf-CDF).
///
/// P_cav(|p⁻|) = ½ · (1 + erf((|p⁻| − p_T) / (σ · √2)))   [Theorem 21.1]
///
/// Implements the Maxwell 2013 statistical model: at |p⁻| = p_T the probability
/// is 50 %; it saturates at 0/1 exponentially fast on either side of the threshold.
/// The erf is evaluated via Abramowitz & Stegun 7.1.26 (max error 1.5×10⁻⁷).
///
/// Args:
///     p_arr: Array of |peak negative pressure| magnitudes [Pa].
///     p_threshold: Mean intrinsic threshold [Pa] (bovine liver, 1 MHz: 28.2 MPa).
///     sigma_pa: Standard deviation [Pa] (bovine liver, 1 MHz: 0.96 MPa).
///
/// Returns:
///     P_cav array [dimensionless, 0–1], same length as p_arr.
///
/// Reference:
///     Maxwell et al. (2013) Ultrasound Med. Biol. 39, 449, Table II.
#[pyfunction]
#[pyo3(signature = (p_arr, p_threshold, sigma_pa))]
pub fn intrinsic_threshold_cavitation_probability(
    py: Python<'_>,
    p_arr: PyReadonlyArray1<f64>,
    p_threshold: f64,
    sigma_pa: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p_s = p_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = cavitation::intrinsic_threshold_cavitation_probability(p_s, p_threshold, sigma_pa);
    Ok(result.to_pyarray(py).unbind())
}

/// Frequency-dependent intrinsic cavitation threshold (Vlaisavljevich 2015 log-linear fit).
///
/// p_T(f) = p_T(1 MHz) + slope * log10(f / 1 MHz)   [Pa]
///
/// Args:
///     f_hz: Frequency array [Hz].
///     p_t_1mhz_pa: Threshold at 1 MHz [Pa] (bovine liver: 28.2 MPa).
///     slope_pa_per_decade: Slope [Pa per decade] (bovine liver: 1.4 MPa).
///
/// Returns:
///     Threshold pressure array [Pa], same length as f_hz.
///
/// Reference:
///     Vlaisavljevich et al. (2015) Ultrasound Med. Biol. 41, 1251, Table I.
#[pyfunction]
#[pyo3(signature = (f_hz, p_t_1mhz_pa, slope_pa_per_decade))]
pub fn frequency_dependent_intrinsic_threshold_pa(
    py: Python<'_>,
    f_hz: PyReadonlyArray1<f64>,
    p_t_1mhz_pa: f64,
    slope_pa_per_decade: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f_s = f_hz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = cavitation::frequency_dependent_intrinsic_threshold_pa(
        f_s,
        p_t_1mhz_pa,
        slope_pa_per_decade,
    );
    Ok(result.to_pyarray(py).unbind())
}

/// Cumulative cavitation probability over N independent single-pulse trials.
///
/// P_cum(N) = 1 − (1 − P_single)^N
///
/// The binomial law is analytically continued for non-integer N via
/// exp(N * ln(1 − P_single)).  N is clamped to >= 1.
///
/// Args:
///     p_single: Single-pulse cavitation probability [0, 1].
///     n_pulses_arr: Pulse count array N (may be non-integer, >= 0).
///
/// Returns:
///     Cumulative probability array, same length as n_pulses_arr.
///
/// Reference:
///     Maxwell et al. (2013) Ultrasound Med. Biol. 39, 449.
#[pyfunction]
#[pyo3(signature = (p_single, n_pulses_arr))]
pub fn cumulative_cavitation_probability(
    py: Python<'_>,
    p_single: f64,
    n_pulses_arr: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let n_s = n_pulses_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = cavitation::cumulative_cavitation_probability(p_single, n_s);
    Ok(result.to_pyarray(py).unbind())
}

/// PRF efficacy factor — residual-bubble shielding model (Macoskey 2018).
///
/// E(PRF) = exp(-max(0, PRF * tau_d - 1) * g)
///
/// Args:
///     prf_hz: Pulse repetition frequency array [Hz].
///     bubble_dissolution_time_s: Residual-bubble dissolution time [s] (liver: ~5 ms).
///     shielding_coefficient: Exponential gain g (Macoskey 2018: ~1.2 for liver).
///
/// Returns:
///     Per-pulse efficacy factor array [0, 1], same length as prf_hz.
///
/// Reference:
///     Macoskey et al. (2018) Ultrasound Med. Biol. 44, 2971.
#[pyfunction]
#[pyo3(signature = (prf_hz, bubble_dissolution_time_s, shielding_coefficient))]
pub fn prf_efficacy_factor(
    py: Python<'_>,
    prf_hz: PyReadonlyArray1<f64>,
    bubble_dissolution_time_s: f64,
    shielding_coefficient: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let prf_s = prf_hz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result =
        cavitation::prf_efficacy_factor(prf_s, bubble_dissolution_time_s, shielding_coefficient);
    Ok(result.to_pyarray(py).unbind())
}


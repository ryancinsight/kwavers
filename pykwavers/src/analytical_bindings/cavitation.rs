//! PyO3 bindings for `kwavers::physics::analytical::cavitation`.

use kwavers::physics::analytical::cavitation;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
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
    Ok(result.into_pyarray(py).unbind())
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
    Ok(result.into_pyarray(py).unbind())
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
    Ok(result.into_pyarray(py).unbind())
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
    Ok(result.into_pyarray(py).unbind())
}

/// Compute the Minnaert resonance frequency of a free bubble.
///
/// f_r = (1/(2*pi*r0)) * sqrt(3*gamma*p0 / rho)
///
/// Args:
///     r0_m: Equilibrium bubble radius [m].
///     gamma: Polytropic index.
///     p0_pa: Ambient pressure [Pa].
///     rho: Liquid density [kg/m³].
///
/// Returns:
///     Resonance frequency [Hz].
#[pyfunction]
#[pyo3(signature = (r0_m, gamma, p0_pa, rho))]
pub fn minnaert_resonance_hz(r0_m: f64, gamma: f64, p0_pa: f64, rho: f64) -> PyResult<f64> {
    Ok(cavitation::minnaert_resonance_hz(r0_m, gamma, p0_pa, rho))
}

/// Compute the Blake cavitation threshold pressure.
///
/// Args:
///     r0_m: Initial bubble radius [m].
///     p0_pa: Ambient pressure [Pa].
///     sigma_n_m: Surface tension [N/m].
///
/// Returns:
///     Blake threshold negative pressure [Pa].
#[pyfunction]
#[pyo3(signature = (r0_m, p0_pa, sigma_n_m))]
pub fn blake_threshold_pa(r0_m: f64, p0_pa: f64, sigma_n_m: f64) -> PyResult<f64> {
    Ok(cavitation::blake_threshold_pa(r0_m, p0_pa, sigma_n_m))
}

/// Compute the Rayleigh collapse time of an empty spherical cavity.
///
/// t_c = 0.9147 * r_max * sqrt(rho / p_inf)
///
/// Args:
///     rmax_m: Maximum bubble radius [m].
///     p_inf_pa: Ambient pressure [Pa].
///     rho: Liquid density [kg/m³].
///
/// Returns:
///     Collapse time [s].
#[pyfunction]
#[pyo3(signature = (rmax_m, p_inf_pa, rho))]
pub fn rayleigh_collapse_time_s(rmax_m: f64, p_inf_pa: f64, rho: f64) -> PyResult<f64> {
    Ok(cavitation::rayleigh_collapse_time_s(rmax_m, p_inf_pa, rho))
}

/// Integrate the Rayleigh–Plesset equation with RK4.
///
/// Args:
///     r0_m: Initial radius [m].
///     rdot0: Initial wall velocity [m/s].
///     p_ac_pa: Acoustic pressure amplitude [Pa].
///     freq_hz: Driving frequency [Hz].
///     t_arr: Time array [s].
///     p0_pa: Ambient pressure [Pa].
///     rho: Liquid density [kg/m³].
///     sigma: Surface tension [N/m].
///     mu: Dynamic viscosity [Pa·s].
///     kappa: Polytropic index.
///     p_v_pa: Vapour pressure [Pa].
///
/// Returns:
///     (r, rdot) — tuple of radius [m] and wall-velocity [m/s] arrays.
#[pyfunction]
#[pyo3(signature = (r0_m, rdot0, p_ac_pa, freq_hz, t_arr, p0_pa, rho, sigma, mu, kappa, p_v_pa))]
pub fn rayleigh_plesset_rk4(
    py: Python<'_>,
    r0_m: f64,
    rdot0: f64,
    p_ac_pa: f64,
    freq_hz: f64,
    t_arr: PyReadonlyArray1<f64>,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (r, rdot) = cavitation::rayleigh_plesset_rk4(
        r0_m, rdot0, p_ac_pa, freq_hz, t_s, p0_pa, rho, sigma, mu, kappa, p_v_pa,
    );
    Ok((r.into_pyarray(py).unbind(), rdot.into_pyarray(py).unbind()))
}

/// Integrate the Keller–Miksis equation with RK4.
///
/// Extends Rayleigh–Plesset to include liquid compressibility via *c_liquid*.
///
/// Args:
///     r0_m: Initial radius [m].
///     rdot0: Initial wall velocity [m/s].
///     p_ac_pa: Acoustic driving amplitude [Pa].
///     freq_hz: Frequency [Hz].
///     t_arr: Time array [s].
///     p0_pa: Ambient pressure [Pa].
///     rho: Density [kg/m³].
///     sigma: Surface tension [N/m].
///     mu: Viscosity [Pa·s].
///     kappa: Polytropic index.
///     p_v_pa: Vapour pressure [Pa].
///     c_liquid: Sound speed in the liquid [m/s].
///
/// Returns:
///     (r, rdot) tuple.
#[pyfunction]
#[pyo3(signature = (r0_m, rdot0, p_ac_pa, freq_hz, t_arr, p0_pa, rho, sigma, mu, kappa, p_v_pa, c_liquid))]
pub fn keller_miksis_rk4(
    py: Python<'_>,
    r0_m: f64,
    rdot0: f64,
    p_ac_pa: f64,
    freq_hz: f64,
    t_arr: PyReadonlyArray1<f64>,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
    c_liquid: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let t_s = t_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (r, rdot) = cavitation::keller_miksis_rk4(
        r0_m, rdot0, p_ac_pa, freq_hz, t_s, p0_pa, rho, sigma, mu, kappa, p_v_pa, c_liquid,
    );
    Ok((r.into_pyarray(py).unbind(), rdot.into_pyarray(py).unbind()))
}

/// Compute the power spectrum of a bubble radius time series.
///
/// Args:
///     r_arr: Radius time series [m].
///     dt_s: Sample interval [s].
///     n_fft: FFT length.
///
/// Returns:
///     (frequencies [Hz], power spectral density) tuple.
#[pyfunction]
#[pyo3(signature = (r_arr, dt_s, n_fft))]
pub fn bubble_power_spectrum(
    py: Python<'_>,
    r_arr: PyReadonlyArray1<f64>,
    dt_s: f64,
    n_fft: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let r_s = r_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (freqs, psd) = cavitation::bubble_power_spectrum(r_s, dt_s, n_fft);
    Ok((
        freqs.into_pyarray(py).unbind(),
        psd.into_pyarray(py).unbind(),
    ))
}

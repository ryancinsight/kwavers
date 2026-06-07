//! PyO3 bindings for `kwavers_physics::analytical::cavitation`.

use kwavers_physics::analytical::cavitation;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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

// The Rayleigh–Plesset and Keller–Miksis RK4 integrators are exposed through a
// single canonical binding pair — `solve_rayleigh_plesset` and
// `solve_keller_miksis` (see `crate::bubble_bindings`) — which build the uniform
// time grid from `(t_end_s, n_steps)`, return `(time, radius, rdot)`, and (for
// Keller–Miksis) carry the shell-viscosity `xi_s` parameter. Both delegate to
// the same `kwavers_physics::analytical::cavitation` functions, so no separate
// raw-`t_arr` binding is kept here.

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

/// Epstein–Plesset gas-diffusion dissolution time of a free air bubble in water.
///
/// Integrates the complete Epstein–Plesset (1950) model (Laplace-overpressure
/// surface-tension drive + transient diffusion term) to the time R₀ → ~0, giving
/// the first-principles residual-bubble dissolution time τ_d that governs
/// inter-pulse shielding.
///
/// Args:
///     r0_m: equilibrium bubble radius [m].
///     saturation_fraction: dissolved-gas saturation f = C∞/C_s (1.0 saturated).
///
/// Returns:
///     Dissolution time [s], or -1.0 if the bubble does not dissolve in the
///     integration window (e.g. supersaturated/growing).
///
/// Reference: Epstein & Plesset (1950) J. Chem. Phys. 18, 1505.
#[pyfunction]
#[pyo3(signature = (r0_m, saturation_fraction=0.5))]
pub fn epstein_plesset_dissolution_time(r0_m: f64, saturation_fraction: f64) -> PyResult<f64> {
    use kwavers_physics::acoustics::bubble_dynamics::{
        dissolution_time_numeric, EpsteinPlessetDissolution, GasDiffusionParams,
    };
    let model =
        EpsteinPlessetDissolution::new(GasDiffusionParams::air_in_water(saturation_fraction));
    Ok(dissolution_time_numeric(&model, r0_m, 1e-9).unwrap_or(-1.0))
}

/// Shelled-microbubble (Sarkar 2009) dissolution time with a finite shell
/// gas-permeability — the coated-contrast-agent persistence time.
///
/// Args:
///     r0_m: equilibrium radius [m].
///     saturation_fraction: dissolved-gas saturation f.
///     shell_permeability_m_s: shell gas-permeation coefficient k_s [m/s]
///         (lipid ≈ 1e-6; k_s→∞ recovers the free bubble, k_s→0 stabilises it).
///
/// Returns:
///     Dissolution time [s], or -1.0 if it does not dissolve in the window.
///
/// Reference: Sarkar, Katiyar & Jain (2009) Ann. Biomed. Eng. 37, 2196.
#[pyfunction]
#[pyo3(signature = (r0_m, saturation_fraction=0.5, shell_permeability_m_s=1.0e-6))]
pub fn shelled_dissolution_time(
    r0_m: f64,
    saturation_fraction: f64,
    shell_permeability_m_s: f64,
) -> PyResult<f64> {
    use kwavers_physics::acoustics::bubble_dynamics::{
        dissolution_time_numeric, GasDiffusionParams, ShellPermeationDissolution,
    };
    let model = ShellPermeationDissolution::new(
        GasDiffusionParams::air_in_water(saturation_fraction),
        shell_permeability_m_s,
    );
    Ok(dissolution_time_numeric(&model, r0_m, 1e-9).unwrap_or(-1.0))
}

/// Low-frequency sound speed of a bubbly liquid (Wood 1930) at gas void fraction β.
///
/// Args:
///     void_fraction, c_liquid, rho_liquid, c_gas, rho_gas.
///
/// Returns:
///     Mixture sound speed [m/s] (collapses far below c_liquid for β ≳ 1e-4).
#[pyfunction]
#[pyo3(signature = (void_fraction, c_liquid=1481.0, rho_liquid=998.0, c_gas=343.0, rho_gas=1.2))]
pub fn wood_sound_speed(
    void_fraction: f64,
    c_liquid: f64,
    rho_liquid: f64,
    c_gas: f64,
    rho_gas: f64,
) -> PyResult<f64> {
    use kwavers_physics::acoustics::bubble_dynamics::wood_sound_speed as wood;
    Ok(wood(void_fraction, c_liquid, rho_liquid, c_gas, rho_gas))
}

/// Resonant-scattering attenuation of a monodisperse bubble cloud
/// (Commander & Prosperetti 1989), in Np/m.
///
/// Args:
///     freq_hz, void_fraction, r0_m, c_liquid, rho_liquid, mu_liquid, p0_pa, polytropic.
#[pyfunction]
#[pyo3(signature = (freq_hz, void_fraction, r0_m, c_liquid=1481.0, rho_liquid=998.0,
                    mu_liquid=1.0e-3, p0_pa=101_325.0, polytropic=1.4))]
#[allow(clippy::too_many_arguments)]
pub fn bubbly_cloud_attenuation(
    freq_hz: f64,
    void_fraction: f64,
    r0_m: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
) -> PyResult<f64> {
    use kwavers_physics::acoustics::bubble_dynamics::commander_prosperetti_attenuation as cp;
    Ok(cp(
        freq_hz,
        void_fraction,
        r0_m,
        c_liquid,
        rho_liquid,
        mu_liquid,
        p0_pa,
        polytropic,
    ))
}

/// Frequency-dependent phase velocity of a monodisperse bubble cloud
/// (Commander & Prosperetti 1989), in m/s — the dispersive companion of
/// [`bubbly_cloud_attenuation`].
///
/// The real part of the complex mixture wavenumber sets `c_p(ω) = ω/Re(k_m)`.
/// Below the bubble (Minnaert) resonance the cloud slows the wave toward the
/// Wood limit; above resonance the phase velocity exceeds `c_liquid` (anomalous
/// dispersion). Its `ω→0` limit reproduces `wood_sound_speed`.
///
/// Args:
///     freq_hz, void_fraction, r0_m, c_liquid, rho_liquid, mu_liquid, p0_pa, polytropic.
///
/// Returns:
///     Phase velocity c_p(f) [m/s].
#[pyfunction]
#[pyo3(signature = (freq_hz, void_fraction, r0_m, c_liquid=1481.0, rho_liquid=998.0,
                    mu_liquid=1.0e-3, p0_pa=101_325.0, polytropic=1.4))]
#[allow(clippy::too_many_arguments)]
pub fn bubbly_cloud_phase_velocity(
    freq_hz: f64,
    void_fraction: f64,
    r0_m: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
) -> PyResult<f64> {
    use kwavers_physics::acoustics::bubble_dynamics::commander_prosperetti_phase_velocity as cp;
    Ok(cp(
        freq_hz,
        void_fraction,
        r0_m,
        c_liquid,
        rho_liquid,
        mu_liquid,
        p0_pa,
        polytropic,
    ))
}

/// Deterministic pulse timeline for a rastered/interleaved sub-spot grid
/// sonication — the single source of truth for the pulsing-pattern diagram and
/// the monitor time-base.
///
/// Args:
///     n_subspots: number of sub-spots in the grid.
///     n_repetitions: number of passes (repetitions) over the grid.
///     pulse_duration_s: single histotripsy pulse duration [s] (microseconds).
///     prf_hz: rate of *fired* pulses (any spot) [Hz].
///     interleaved: True → round-robin (one pulse per spot per repetition);
///         False → sequential (all repetitions at a spot before the next).
///
/// Returns:
///     (onset_s, subspot_idx, repetition_idx, pulse_duration_s,
///      repetition_time_s, sonication_duration_s, n_repetitions) — the first
///     three are arrays of length n_subspots·n_repetitions in fire order.
#[pyfunction]
#[pyo3(signature = (n_subspots, n_repetitions, pulse_duration_s, prf_hz, interleaved=true))]
pub fn sonication_schedule(
    py: Python<'_>,
    n_subspots: usize,
    n_repetitions: usize,
    pulse_duration_s: f64,
    prf_hz: f64,
    interleaved: bool,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
    f64,
    f64,
    f64,
    usize,
)> {
    use kwavers_physics::analytical::cavitation::{build_sonication_schedule, SonicationOrder};
    let order = if interleaved {
        SonicationOrder::Interleaved
    } else {
        SonicationOrder::Sequential
    };
    let s = build_sonication_schedule(n_subspots, n_repetitions, pulse_duration_s, prf_hz, order);
    let subspot_i64: Vec<i64> = s.subspot.iter().map(|&v| v as i64).collect();
    let repetition_i64: Vec<i64> = s.repetition.iter().map(|&v| v as i64).collect();
    Ok((
        s.onset_s.into_pyarray(py).unbind(),
        subspot_i64.into_pyarray(py).unbind(),
        repetition_i64.into_pyarray(py).unbind(),
        s.pulse_duration_s,
        s.repetition_time_s,
        s.sonication_duration_s,
        s.n_repetitions,
    ))
}

/// One-way delivered-pressure fraction at the focus: electronic-steering
/// efficiency × interface pressure transmission × tissue power-law attenuation ×
/// residual-gas (Commander–Prosperetti) attenuation over the path.
///
/// Args:
///     steering_eff, interface_z_prox, interface_z_focal, alpha_tissue_np_m,
///     path_len_m, void_beta, freq_hz, r0_m, c_liquid, rho_liquid, mu_liquid,
///     p0_pa, polytropic.
#[pyfunction]
#[pyo3(signature = (steering_eff, interface_z_prox, interface_z_focal, alpha_tissue_np_m,
                    path_len_m, void_beta, freq_hz, r0_m, c_liquid=1481.0, rho_liquid=998.0,
                    mu_liquid=1.0e-3, p0_pa=101_325.0, polytropic=1.4))]
#[allow(clippy::too_many_arguments)]
pub fn forward_delivery_fraction(
    steering_eff: f64,
    interface_z_prox: f64,
    interface_z_focal: f64,
    alpha_tissue_np_m: f64,
    path_len_m: f64,
    void_beta: f64,
    freq_hz: f64,
    r0_m: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
) -> PyResult<f64> {
    Ok(
        kwavers_physics::analytical::cavitation::forward_delivery_fraction(
            steering_eff,
            interface_z_prox,
            interface_z_focal,
            alpha_tissue_np_m,
            path_len_m,
            void_beta,
            freq_hz,
            r0_m,
            c_liquid,
            rho_liquid,
            mu_liquid,
            p0_pa,
            polytropic,
        ),
    )
}

/// Two-way (round-trip) amplitude fraction of a passive cavitation emission
/// measured back at the transducer — the genuine reflection/scattering/
/// attenuation loss that derates the *measured* signal vs the cavitation
/// actually produced at the focus.
///
/// Args:
///     interface_z_prox, interface_z_focal, alpha_tissue_np_m, path_len_m,
///     void_beta, freq_hz, r0_m, c_liquid, rho_liquid, mu_liquid, p0_pa, polytropic.
#[pyfunction]
#[pyo3(signature = (interface_z_prox, interface_z_focal, alpha_tissue_np_m, path_len_m,
                    void_beta, freq_hz, r0_m, c_liquid=1481.0, rho_liquid=998.0,
                    mu_liquid=1.0e-3, p0_pa=101_325.0, polytropic=1.4))]
#[allow(clippy::too_many_arguments)]
pub fn received_signal_fraction(
    interface_z_prox: f64,
    interface_z_focal: f64,
    alpha_tissue_np_m: f64,
    path_len_m: f64,
    void_beta: f64,
    freq_hz: f64,
    r0_m: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
) -> PyResult<f64> {
    Ok(
        kwavers_physics::analytical::cavitation::received_signal_fraction(
            interface_z_prox,
            interface_z_focal,
            alpha_tissue_np_m,
            path_len_m,
            void_beta,
            freq_hz,
            r0_m,
            c_liquid,
            rho_liquid,
            mu_liquid,
            p0_pa,
            polytropic,
        ),
    )
}

/// Normal-incidence pressure transmission coefficient `2·z2/(z1+z2)` between two
/// acoustic impedances [Pa·s/m].
#[pyfunction]
#[pyo3(signature = (z1, z2))]
pub fn pressure_transmission_coefficient(z1: f64, z2: f64) -> PyResult<f64> {
    Ok(kwavers_physics::analytical::cavitation::pressure_transmission_coefficient(z1, z2))
}

/// Local peak-pressure enhancement `1 + |（z2−z1)/(z2+z1)|` at an acoustic interface
/// between impedances `z1` and `z2` [Pa·s/m] — the incident+reflected superposition
/// that makes cavitation nucleate preferentially at tissue boundaries (mild for
/// soft-tissue contrasts, approaching 2 against a gas-filled lacuna).
#[pyfunction]
#[pyo3(signature = (z1, z2))]
pub fn interface_pressure_enhancement(z1: f64, z2: f64) -> PyResult<f64> {
    Ok(kwavers_physics::analytical::cavitation::interface_pressure_enhancement(z1, z2))
}

/// Cavitation-susceptibility multiplier of already-fractionated tissue ("lesion
/// memory"): `S = 1 + k_immediate·f + k_lacuna·f·(1 − exp(−t_since/τ_lacuna))`, with
/// `f` the local fractionation, `time_since_lesion_s` the elapsed time, and
/// `tau_lacuna_s` the gas-evolution (lacuna formation) time constant. The delayed
/// lacuna term is negligible during the first procedure (`t ≪ τ`) and saturates on
/// re-treatment (`t ≫ τ`).
///
/// Args:
///     fractionation, time_since_lesion_s, tau_lacuna_s, k_immediate, k_lacuna.
#[pyfunction]
#[pyo3(signature = (fractionation, time_since_lesion_s, tau_lacuna_s, k_immediate=0.5, k_lacuna=4.0))]
pub fn lacuna_cavitation_susceptibility(
    fractionation: f64,
    time_since_lesion_s: f64,
    tau_lacuna_s: f64,
    k_immediate: f64,
    k_lacuna: f64,
) -> PyResult<f64> {
    Ok(
        kwavers_physics::analytical::cavitation::lacuna_cavitation_susceptibility(
            fractionation,
            time_since_lesion_s,
            tau_lacuna_s,
            k_immediate,
            k_lacuna,
        ),
    )
}

/// Histotripsy mechanical cell-kill fraction from cumulative cavitation dose via the
/// Weibull survival dose–response `kill = 1 − exp(−(dose/d0)^k)` (the cumulative
/// cell-survival form underlying radiobiology's biologically-effective dose, but the
/// mechanism here is mechanical fractionation). `d0` = characteristic dose (≈63 % kill),
/// `weibull_k` > 1 the threshold/shoulder exponent.
#[pyfunction]
#[pyo3(signature = (dose, d0, weibull_k=2.5))]
pub fn histotripsy_kill_fraction(dose: f64, d0: f64, weibull_k: f64) -> PyResult<f64> {
    Ok(kwavers_physics::analytical::cavitation::histotripsy_kill_fraction(dose, d0, weibull_k))
}

/// Lethal cumulative cavitation dose LD_x for cell-kill `fraction` (LD50 ⇒ 0.5):
/// `D = d0·(−ln(1−fraction))^(1/k)`. Inverse of `histotripsy_kill_fraction`.
#[pyfunction]
#[pyo3(signature = (fraction, d0, weibull_k=2.5))]
pub fn histotripsy_lethal_dose(fraction: f64, d0: f64, weibull_k: f64) -> PyResult<f64> {
    Ok(kwavers_physics::analytical::cavitation::histotripsy_lethal_dose(fraction, d0, weibull_k))
}

/// Lateral semi-axis that keeps an anisotropic focal ellipsoid within a clearance
/// constraint.
#[pyfunction]
#[pyo3(signature = (natural_lateral_radius_m, clearance_m, axial_to_lateral_ratio))]
pub fn clipped_lateral_radius_for_clearance(
    natural_lateral_radius_m: f64,
    clearance_m: f64,
    axial_to_lateral_ratio: f64,
) -> PyResult<f64> {
    Ok(
        kwavers_physics::analytical::cavitation::clipped_lateral_radius_for_clearance(
            natural_lateral_radius_m,
            clearance_m,
            axial_to_lateral_ratio,
        ),
    )
}

/// Check whether a beam-axis elongated focal ellipsoid remains inside an allowed mask.
#[pyfunction]
#[pyo3(signature = (allowed_mask, nx, ny, nz, center_x, center_y, center_z, lateral_radius_m, axial_radius_m, dx_m))]
#[allow(clippy::too_many_arguments)]
pub fn ellipsoid_respects_allowed_mask(
    allowed_mask: PyReadonlyArray1<bool>,
    nx: usize,
    ny: usize,
    nz: usize,
    center_x: usize,
    center_y: usize,
    center_z: usize,
    lateral_radius_m: f64,
    axial_radius_m: f64,
    dx_m: f64,
) -> PyResult<bool> {
    let mask = allowed_mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(
        kwavers_physics::analytical::cavitation::ellipsoid_respects_allowed_mask(
            mask,
            nx,
            ny,
            nz,
            center_x,
            center_y,
            center_z,
            lateral_radius_m,
            axial_radius_m,
            dx_m,
        ),
    )
}

/// Apply receive-path and tissue-state scaling to a passive cavitation PSD.
#[pyfunction]
#[pyo3(signature = (psd, receive_fraction, susceptibility))]
pub fn scale_measured_emission_spectrum(
    py: Python<'_>,
    psd: PyReadonlyArray1<f64>,
    receive_fraction: f64,
    susceptibility: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p = psd
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = kwavers_physics::analytical::cavitation::scale_measured_emission_spectrum(
        p,
        receive_fraction,
        susceptibility,
    );
    Ok(out.into_pyarray(py).unbind())
}

/// Convert delivered cumulative histotripsy dose samples to kill fractions.
#[pyfunction]
#[pyo3(signature = (dose, d0, weibull_k))]
pub fn delivered_histotripsy_progress(
    py: Python<'_>,
    dose: PyReadonlyArray1<f64>,
    d0: f64,
    weibull_k: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let d = dose
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out =
        kwavers_physics::analytical::cavitation::delivered_histotripsy_progress(d, d0, weibull_k);
    Ok(out.into_pyarray(py).unbind())
}

/// Backscatter coefficient of partially fractionated tissue (lesion B-mode).
///
/// σ_bsc(f) = σ_liquefied + (σ_intact − σ_liquefied)·(1 − f)^γ. As the
/// fractionation fraction `f` rises the lesion loses speckle scatterers and
/// becomes hypoechoic. Thin wrapper over
/// `kwavers_physics::analytical::cavitation::fractionation_backscatter_coefficient`.
///
/// Args:
///     fractionation: per-voxel fractionation/kill fraction [0, 1].
///     sigma_intact: intact-tissue backscatter coefficient (arb. units).
///     sigma_liquefied: liquefied-homogenate floor backscatter coefficient.
///     gamma: scatterer-loss exponent (≥ 1; 2 = quadratic).
///
/// Returns:
///     Backscatter-coefficient array, same length as `fractionation`.
#[pyfunction]
#[pyo3(signature = (fractionation, sigma_intact, sigma_liquefied, gamma))]
pub fn fractionation_backscatter_coefficient(
    py: Python<'_>,
    fractionation: PyReadonlyArray1<f64>,
    sigma_intact: f64,
    sigma_liquefied: f64,
    gamma: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f = fractionation
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = kwavers_physics::analytical::cavitation::fractionation_backscatter_coefficient(
        f,
        sigma_intact,
        sigma_liquefied,
        gamma,
    );
    Ok(out.into_pyarray(py).unbind())
}

/// Acoustic impedance of partially fractionated tissue (lesion-rim echo).
///
/// Z(f) = z_intact·(1 − f) + z_liquefied·f (linear volume mixing). The spatial
/// gradient of this map produces the specular bright rim at the lesion boundary.
/// Thin wrapper over
/// `kwavers_physics::analytical::cavitation::fractionation_acoustic_impedance`.
///
/// Args:
///     fractionation: per-voxel fractionation/kill fraction [0, 1].
///     z_intact: intact-tissue acoustic impedance [Rayl].
///     z_liquefied: liquefied-homogenate acoustic impedance [Rayl].
///
/// Returns:
///     Acoustic-impedance array, same length as `fractionation`.
#[pyfunction]
#[pyo3(signature = (fractionation, z_intact, z_liquefied))]
pub fn fractionation_acoustic_impedance(
    py: Python<'_>,
    fractionation: PyReadonlyArray1<f64>,
    z_intact: f64,
    z_liquefied: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f = fractionation
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = kwavers_physics::analytical::cavitation::fractionation_acoustic_impedance(
        f,
        z_intact,
        z_liquefied,
    );
    Ok(out.into_pyarray(py).unbind())
}

/// Size boiling-histotripsy lesion and pulse count from a resolved pressure
/// profile. Returns `(pulses, lateral_radius_m, axial_radius_m, pulse_ms)`, or
/// `None` when the focus does not boil within the pulse limit.
#[pyfunction]
#[pyo3(signature = (
    radius_m, normalized_pressure, focal_pressure_pa, focal_depth_m, freq_hz,
    c_m_s, rho_kg_m3, beta_nonlinearity, alpha_np_m, heat_capacity_j_kg_k,
    delta_t_k, tau_max_s, axial_to_lateral_ratio, clearance_m, coverage_target
))]
#[allow(clippy::too_many_arguments)]
pub fn boiling_lesion_from_pressure_profile(
    radius_m: PyReadonlyArray1<f64>,
    normalized_pressure: PyReadonlyArray1<f64>,
    focal_pressure_pa: f64,
    focal_depth_m: f64,
    freq_hz: f64,
    c_m_s: f64,
    rho_kg_m3: f64,
    beta_nonlinearity: f64,
    alpha_np_m: f64,
    heat_capacity_j_kg_k: f64,
    delta_t_k: f64,
    tau_max_s: f64,
    axial_to_lateral_ratio: f64,
    clearance_m: f64,
    coverage_target: f64,
) -> PyResult<Option<(usize, f64, f64, f64)>> {
    let r = radius_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let b = normalized_pressure
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(
        kwavers_physics::analytical::cavitation::boiling_lesion_from_pressure_profile(
            r,
            b,
            focal_pressure_pa,
            focal_depth_m,
            freq_hz,
            c_m_s,
            rho_kg_m3,
            beta_nonlinearity,
            alpha_np_m,
            heat_capacity_j_kg_k,
            delta_t_k,
            tau_max_s,
            axial_to_lateral_ratio,
            clearance_m,
            coverage_target,
        )
        .map(|p| (p.pulses, p.lateral_radius_m, p.axial_radius_m, p.pulse_ms)),
    )
}

/// Boiling-onset time samples from normalized pressure samples.
#[pyfunction]
#[pyo3(signature = (
    normalized_pressure, focal_pressure_pa, focal_depth_m, freq_hz, c_m_s,
    rho_kg_m3, beta_nonlinearity, alpha_np_m, heat_capacity_j_kg_k, delta_t_k
))]
#[allow(clippy::too_many_arguments)]
pub fn boiling_time_profile_from_pressure(
    py: Python<'_>,
    normalized_pressure: PyReadonlyArray1<f64>,
    focal_pressure_pa: f64,
    focal_depth_m: f64,
    freq_hz: f64,
    c_m_s: f64,
    rho_kg_m3: f64,
    beta_nonlinearity: f64,
    alpha_np_m: f64,
    heat_capacity_j_kg_k: f64,
    delta_t_k: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let b = normalized_pressure
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = py.detach(|| {
        kwavers_physics::analytical::cavitation::boiling_time_profile_from_pressure(
            b,
            focal_pressure_pa,
            focal_depth_m,
            freq_hz,
            c_m_s,
            rho_kg_m3,
            beta_nonlinearity,
            alpha_np_m,
            heat_capacity_j_kg_k,
            delta_t_k,
        )
    });
    Ok(out.into_pyarray(py).unbind())
}

/// Propagate one cavitation source PSD to passive receiver-channel PSDs.
#[pyfunction]
#[pyo3(signature = (source_psd, source_xyz, receiver_xyz, alpha_np_m))]
pub fn receiver_channel_psd_from_source(
    py: Python<'_>,
    source_psd: PyReadonlyArray1<f64>,
    source_xyz: PyReadonlyArray1<f64>,
    receiver_xyz: PyReadonlyArray2<f64>,
    alpha_np_m: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let psd = source_psd
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let src = source_xyz
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let recv = receiver_xyz.as_array();
    if src.len() != 3 || recv.ncols() != 3 {
        return Err(PyRuntimeError::new_err(
            "source_xyz must have length 3 and receiver_xyz shape (n, 3)",
        ));
    }
    let recv_flat: Vec<f64> = recv.iter().copied().collect();
    let flat = py.detach(|| {
        kwavers_physics::analytical::cavitation::receiver_channel_psd_from_source(
            psd,
            [src[0], src[1], src[2]],
            &recv_flat,
            alpha_np_m,
        )
    });
    let arr = ndarray::Array2::from_shape_vec((recv.nrows(), psd.len()), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).unbind())
}

/// Sum receiver-channel PSDs into the measured array spectrum.
#[pyfunction]
#[pyo3(signature = (channel_psd))]
pub fn integrate_channel_psd(
    py: Python<'_>,
    channel_psd: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let arr = channel_psd.as_array();
    let flat: Vec<f64> = arr.iter().copied().collect();
    let out = py.detach(|| {
        kwavers_physics::analytical::cavitation::integrate_channel_psd(
            &flat,
            arr.nrows(),
            arr.ncols(),
        )
    });
    Ok(out.into_pyarray(py).unbind())
}

/// Lacuna gas void fraction in fractionated tissue from first-order gas-evolution
/// growth: `β = β_max·f·(1 − exp(−t_since/τ_lacuna))`. Feeds the Wood/Commander–
/// Prosperetti medium coupling so the growing lacuna geometry shields and aberrates
/// subsequent pulses (the persistent gas cavity, distinct from the fast residual
/// bubble-cloud dissolution).
///
/// Args:
///     fractionation, time_since_lesion_s, tau_lacuna_s, beta_max.
#[pyfunction]
#[pyo3(signature = (fractionation, time_since_lesion_s, tau_lacuna_s, beta_max))]
pub fn lacuna_void_fraction(
    fractionation: f64,
    time_since_lesion_s: f64,
    tau_lacuna_s: f64,
    beta_max: f64,
) -> PyResult<f64> {
    Ok(
        kwavers_physics::analytical::cavitation::lacuna_void_fraction(
            fractionation,
            time_since_lesion_s,
            tau_lacuna_s,
            beta_max,
        ),
    )
}

/// Pulse count needed to grow a histotripsy lesion to `target_radius_m` via the
/// cavitation energy-balance model `R_L = R₀·(P₀·N·icd_per_pulse/σ_y)^(1/3)`.
///
/// Used to size the per-spot dose for full tumour coverage and to cap it so the
/// expanding lesion keeps a safe margin from a sensitive structure.
///
/// Args:
///     target_radius_m, r0_m, p0_pa, tissue_yield_stress_pa, icd_per_pulse.
#[pyfunction]
#[pyo3(signature = (target_radius_m, r0_m, p0_pa, tissue_yield_stress_pa, icd_per_pulse))]
pub fn histotripsy_pulses_for_lesion_radius(
    target_radius_m: f64,
    r0_m: f64,
    p0_pa: f64,
    tissue_yield_stress_pa: f64,
    icd_per_pulse: f64,
) -> PyResult<f64> {
    Ok(
        kwavers_physics::analytical::cavitation::histotripsy_pulses_for_lesion_radius(
            target_radius_m,
            r0_m,
            p0_pa,
            tissue_yield_stress_pa,
            icd_per_pulse,
        ),
    )
}

/// Histotripsy lesion radius [m] from accumulated inertial cavitation dose via
/// the cavitation energy-balance model `R_L = R₀·(P₀·icd/σ_y)^(1/3)` (forward
/// of [`histotripsy_pulses_for_lesion_radius`]).
///
/// Args:
///     icd (total dimensionless inertial cavitation dose), r0_m, p0_pa,
///     tissue_yield_stress_pa.
#[pyfunction]
#[pyo3(signature = (icd, r0_m, p0_pa, tissue_yield_stress_pa))]
pub fn histotripsy_lesion_radius_m(
    icd: f64,
    r0_m: f64,
    p0_pa: f64,
    tissue_yield_stress_pa: f64,
) -> PyResult<f64> {
    Ok(
        kwavers_physics::analytical::cavitation::histotripsy_lesion_radius_m(
            icd,
            r0_m,
            p0_pa,
            tissue_yield_stress_pa,
        ),
    )
}

/// Inertial cavitation dose (ICD) from a bubble radius/wall-velocity trajectory:
/// the sum of `(R_max/R₀)³` over detected inertial collapse events (Duryea 2015).
/// Dimensionless, O(1–1000); feeds the lesion energy-balance model.
///
/// Args:
///     r_arr (radius [m]), rdot_arr (wall velocity [m/s]), r0_m (equilibrium [m]).
#[pyfunction]
#[pyo3(signature = (r_arr, rdot_arr, r0_m))]
pub fn inertial_cavitation_dose(
    r_arr: PyReadonlyArray1<f64>,
    rdot_arr: PyReadonlyArray1<f64>,
    r0_m: f64,
) -> PyResult<f64> {
    let r = r_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let rd = rdot_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(cavitation::inertial_cavitation_dose(r, rd, r0_m))
}

/// True single-bubble acoustic-emission simulation via the production adaptive
/// Keller–Miksis solver (gas thermodynamics, mass transfer, compressible
/// radiation damping; Richardson-extrapolation adaptive sub-stepping that
/// survives inertial collapse where a fixed-step RK4 diverges).
///
/// Drives the bubble with p_ac(t) = drive_amp·sin(2π f t) and records the
/// far-field emission p_sc(t) = rho·R/r_obs·(2 Rdot² + R Rddot) using the exact
/// wall acceleration. The harmonic/subharmonic/broadband content of the
/// resulting spectrum is emergent, not imposed.
///
/// Args:
///     r0_m: Equilibrium radius [m].
///     drive_amp_pa: Peak acoustic drive pressure [Pa].
///     drive_freq_hz: Drive frequency [Hz].
///     n_cycles: Number of drive cycles to simulate.
///     n_out: Number of uniform output samples.
///     r_obs_m: Far-field observation distance [m].
///     p0_pa, rho, c_liquid, mu, sigma, pv, gamma: liquid/gas properties.
///     thermal_effects: include gas thermodynamics + mass transfer.
///
/// Returns:
///     (time, radius, wall_velocity, emission, max_compression, max_mach,
///      collapse_count, converged) — four arrays then four diagnostics.
#[pyfunction]
#[pyo3(signature = (r0_m, drive_amp_pa, drive_freq_hz, n_cycles, n_out, r_obs_m,
                    p0_pa=101_325.0, rho=998.0, c_liquid=1481.0, mu=1.0e-3,
                    sigma=0.0725, pv=2330.0, gamma=1.4, thermal_effects=false))]
#[allow(clippy::too_many_arguments)]
pub fn simulate_bubble_emission(
    py: Python<'_>,
    r0_m: f64,
    drive_amp_pa: f64,
    drive_freq_hz: f64,
    n_cycles: f64,
    n_out: usize,
    r_obs_m: f64,
    p0_pa: f64,
    rho: f64,
    c_liquid: f64,
    mu: f64,
    sigma: f64,
    pv: f64,
    gamma: f64,
    thermal_effects: bool,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    u32,
    bool,
)> {
    let cfg = cavitation::BubbleDriveConfig {
        r0_m,
        p0_pa,
        rho,
        c_liquid,
        mu,
        sigma,
        pv,
        gamma,
        drive_freq_hz,
        drive_amp_pa,
        n_cycles,
        n_out,
        r_obs_m,
        thermal_effects,
    };
    let tr = cavitation::simulate_bubble_emission(&cfg);
    Ok((
        tr.time.into_pyarray(py).unbind(),
        tr.radius.into_pyarray(py).unbind(),
        tr.wall_velocity.into_pyarray(py).unbind(),
        tr.emission.into_pyarray(py).unbind(),
        tr.max_compression,
        tr.max_mach,
        tr.collapse_count,
        tr.converged,
    ))
}

/// True *coated* (encapsulated) microbubble emission simulation via the
/// Marmottant shell model (lipid/protein shell with buckling and rupture).
///
/// The shell's piecewise surface tension (σ→0 when buckled, σ→σ_water when
/// ruptured) period-doubles the dynamics, so a clinical contrast microbubble
/// emits a SUBHARMONIC at low drive pressures where a free bubble does not —
/// the marker BBB-opening controllers track. Shell-damped Rayleigh–Plesset is
/// integrated with a fixed-step RK4; the emergent emission spectrum is returned.
///
/// Args:
///     r0_m, drive_amp_pa, drive_freq_hz, n_cycles, n_out, r_obs_m: as for
///         simulate_bubble_emission.
///     chi: shell elastic compression modulus χ [N/m] (lipid ≈ 0.25–1.0).
///     shell_viscosity: shell shear viscosity [Pa·s] (lipid ≈ 0.5).
///     shell_thickness: shell thickness [m] (lipid ≈ 3e-9).
///     sigma_initial: unstressed shell surface tension [N/m] (≈ 0.04).
///     steps_per_cycle: RK4 sub-steps per drive cycle.
///     p0_pa, rho, c_liquid, mu, gamma: liquid/gas properties.
///
/// Returns:
///     (time, radius, wall_velocity, emission, max_compression, max_mach,
///      collapse_count, converged).
#[pyfunction]
#[pyo3(signature = (r0_m, drive_amp_pa, drive_freq_hz, n_cycles, n_out, r_obs_m,
                    chi=0.5, shell_viscosity=0.5, shell_thickness=3.0e-9,
                    sigma_initial=0.04, steps_per_cycle=2000, p0_pa=101_325.0,
                    rho=998.0, c_liquid=1481.0, mu=1.0e-3, gamma=1.4))]
#[allow(clippy::too_many_arguments)]
pub fn simulate_coated_bubble_emission(
    py: Python<'_>,
    r0_m: f64,
    drive_amp_pa: f64,
    drive_freq_hz: f64,
    n_cycles: f64,
    n_out: usize,
    r_obs_m: f64,
    chi: f64,
    shell_viscosity: f64,
    shell_thickness: f64,
    sigma_initial: f64,
    steps_per_cycle: usize,
    p0_pa: f64,
    rho: f64,
    c_liquid: f64,
    mu: f64,
    gamma: f64,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    u32,
    bool,
)> {
    let cfg = cavitation::ShellDriveConfig {
        r0_m,
        p0_pa,
        rho,
        c_liquid,
        mu,
        gamma,
        drive_freq_hz,
        drive_amp_pa,
        n_cycles,
        steps_per_cycle,
        n_out,
        r_obs_m,
        chi,
        shell_viscosity,
        shell_thickness,
        sigma_initial,
    };
    let tr = cavitation::simulate_coated_bubble_emission(&cfg);
    Ok((
        tr.time.into_pyarray(py).unbind(),
        tr.radius.into_pyarray(py).unbind(),
        tr.wall_velocity.into_pyarray(py).unbind(),
        tr.emission.into_pyarray(py).unbind(),
        tr.max_compression,
        tr.max_mach,
        tr.collapse_count,
        tr.converged,
    ))
}

/// Hann-windowed single-sided power spectral density of an emission series.
///
/// Suppresses spectral leakage from dominant harmonic lines so the inter-line
/// floor reflects true inharmonic (broadband / inertial) emission — the
/// estimator passive-cavitation-dose decomposition should run on.
///
/// Args:
///     signal: Emission time series (e.g. from bubble_acoustic_emission_pressure).
///     dt_s: Sample interval [s].
///     n_fft: FFT length (>= signal length; zero-padded).
///
/// Returns:
///     (frequencies [Hz], PSD) tuple over non-negative frequencies.
///
/// Reference:
///     Gyongy & Coussios (2010) JASA 128, 2403.
#[pyfunction]
#[pyo3(signature = (signal, dt_s, n_fft))]
pub fn hann_windowed_power_spectrum(
    py: Python<'_>,
    signal: PyReadonlyArray1<f64>,
    dt_s: f64,
    n_fft: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let s = signal
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (freqs, psd) = cavitation::hann_windowed_power_spectrum(s, dt_s, n_fft);
    Ok((
        freqs.into_pyarray(py).unbind(),
        psd.into_pyarray(py).unbind(),
    ))
}

/// Far-field acoustic emission pressure radiated by a pulsating bubble.
///
/// p_sc(r_obs, t) = (rho * R / r_obs) * (2*Rdot^2 + R*Rddot)
///
/// This is the signal a passive cavitation detector records, computed from the
/// radius/wall-velocity history returned by `solve_keller_miksis` /
/// `solve_rayleigh_plesset`. Rddot is obtained by central differences of Rdot.
///
/// Args:
///     r_arr: Bubble radius series R(t) [m].
///     rdot_arr: Wall velocity series Rdot(t) [m/s] (same length as r_arr).
///     dt_s: Uniform time step [s].
///     rho: Liquid density [kg/m³].
///     r_obs_m: Observation distance from the bubble [m].
///
/// Returns:
///     Emitted-pressure series p_sc(t) [Pa] (same length as r_arr).
///
/// Reference:
///     Leighton (1994) The Acoustic Bubble, §3.2.1; Neppiras (1980) Phys. Rep. 61, 159.
#[pyfunction]
#[pyo3(signature = (r_arr, rdot_arr, dt_s, rho, r_obs_m))]
pub fn bubble_acoustic_emission_pressure(
    py: Python<'_>,
    r_arr: PyReadonlyArray1<f64>,
    rdot_arr: PyReadonlyArray1<f64>,
    dt_s: f64,
    rho: f64,
    r_obs_m: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let r_s = r_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let rd_s = rdot_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = cavitation::bubble_acoustic_emission_pressure(r_s, rd_s, dt_s, rho, r_obs_m);
    Ok(result.into_pyarray(py).unbind())
}

/// Coherently superpose a microbubble population's emission series into one
/// ensemble time series.
///
/// y[t] = sum_i gains[i] * emissions[i][t - delays[i]]. Each of the n_bubbles
/// per-bubble series (rows of the (n_bubbles, n_samples) matrix) is placed at an
/// integer sample delay and gain, accumulating into a buffer of length out_len.
/// Genuine broadband emission is this ensemble effect: a single steady-state
/// bubble is a line spectrum, but a polydisperse population of transient
/// emissions at random nucleation delays fills the inter-harmonic floor.
/// Feed the result to `hann_windowed_power_spectrum`.
///
/// Args:
///     emissions: (n_bubbles, n_samples) per-bubble emission series.
///     delays_samples: per-bubble nucleation/arrival delay [samples].
///     gains: per-bubble amplitude weight.
///     out_len: length of the summed output buffer (>= n_samples + max delay).
///
/// Returns:
///     Summed ensemble emission series of length out_len.
///
/// Reference:
///     Gyongy & Coussios (2010) JASA 128, 2403.
#[pyfunction]
#[pyo3(signature = (emissions, delays_samples, gains, out_len))]
pub fn ensemble_emission_superposition(
    py: Python<'_>,
    emissions: PyReadonlyArray2<f64>,
    delays_samples: PyReadonlyArray1<i64>,
    gains: PyReadonlyArray1<f64>,
    out_len: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let arr = emissions.as_array();
    let (n_bubbles, n_samples) = arr.dim();
    let flat: Vec<f64> = arr.iter().copied().collect();
    let delays: Vec<usize> = delays_samples
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .iter()
        .map(|&d| d.max(0) as usize)
        .collect();
    let g = gains
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = cavitation::ensemble_emission_superposition(
        &flat, n_bubbles, n_samples, &delays, g, out_len,
    );
    Ok(result.into_pyarray(py).unbind())
}

/// Decompose a passive-cavitation emission spectrum into cavitation bands.
///
/// Each PSD bin is assigned to the nearest half-harmonic line k*f0/2 within the
/// half-window rel_halfwidth*f0, integrated above the noise floor:
///   k even (>=2) -> harmonic comb (fundamental); k=1 -> subharmonic;
///   k odd (>=3) -> ultraharmonic; otherwise -> broadband.
///
/// Args:
///     freqs: Frequency axis [Hz], uniformly spaced ascending.
///     psd: Power spectral density at each frequency (same length).
///     f0_hz: Fundamental drive frequency [Hz].
///     rel_halfwidth: Line half-window as fraction of f0 (clamped to (0, 0.25)).
///     noise_floor: Baseline PSD subtracted from every bin (>= 0).
///
/// Returns:
///     (fundamental, subharmonic, ultraharmonic, broadband) band energies
///     [PSD-units * Hz]. Stable-cavitation emission = subharmonic + ultraharmonic;
///     inertial-cavitation emission = broadband.
///
/// Reference:
///     Gyongy & Coussios (2010) JASA 128, 2403; Arvanitis et al. (2012) PLoS ONE 7, e45783.
#[pyfunction]
#[pyo3(signature = (freqs, psd, f0_hz, rel_halfwidth, noise_floor))]
pub fn cavitation_emission_bands(
    freqs: PyReadonlyArray1<f64>,
    psd: PyReadonlyArray1<f64>,
    f0_hz: f64,
    rel_halfwidth: f64,
    noise_floor: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    let f_s = freqs
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let p_s = psd
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let b = cavitation::decompose_emission_spectrum(f_s, p_s, f0_hz, rel_halfwidth, noise_floor);
    Ok((b.fundamental, b.subharmonic, b.ultraharmonic, b.broadband))
}

/// Cumulative cavitation dose: trapezoidal time-integral of an emission-power series.
///
/// D[m] = sum_{i=1..m} 0.5*(P[i-1]+P[i])*dt   [emission-power * s]
///
/// Feed the stable emission (sub+ultra) for the stable-cavitation dose, or the
/// broadband emission for the inertial-cavitation dose. Negative samples clamp to 0.
///
/// Args:
///     power_arr: Per-window band emission power.
///     dt_s: Monitoring-window duration [s].
///
/// Returns:
///     Running cumulative dose array (same length; D[0] = 0).
///
/// Reference:
///     O'Reilly & Hynynen (2012) Radiology 263, 96.
#[pyfunction]
#[pyo3(signature = (power_arr, dt_s))]
pub fn cumulative_cavitation_dose(
    py: Python<'_>,
    power_arr: PyReadonlyArray1<f64>,
    dt_s: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p_s = power_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = cavitation::cumulative_cavitation_dose(p_s, dt_s);
    Ok(result.into_pyarray(py).unbind())
}

/// One step of the closed-loop cavitation-dose pressure controller.
///
/// Safety dominates: if broadband (inertial) emission exceeds inertial_limit,
/// pressure is reduced by (1-gain); else if stable (sub+ultra) emission is below
/// stable_target it is raised by (1+gain); otherwise held. Result clamped to
/// [p_min_pa, p_max_pa].
///
/// Args:
///     current_p_pa: Drive pressure on the just-monitored burst [Pa].
///     stable_emission: Measured sub+ultra-harmonic emission this burst.
///     inertial_emission: Measured broadband emission this burst.
///     stable_target: Stable-emission set-point.
///     inertial_limit: Broadband-emission ceiling.
///     gain: Fractional pressure step per burst (>= 0).
///     p_min_pa, p_max_pa: Drive-pressure clamp [Pa].
///
/// Returns:
///     Drive pressure for the next burst [Pa].
///
/// Reference:
///     McDannold et al. (2006) Phys. Med. Biol. 51, 793.
#[pyfunction]
#[pyo3(signature = (current_p_pa, stable_emission, inertial_emission, stable_target, inertial_limit, gain, p_min_pa, p_max_pa))]
#[allow(clippy::too_many_arguments)]
pub fn cavitation_controller_pressure(
    current_p_pa: f64,
    stable_emission: f64,
    inertial_emission: f64,
    stable_target: f64,
    inertial_limit: f64,
    gain: f64,
    p_min_pa: f64,
    p_max_pa: f64,
) -> PyResult<f64> {
    Ok(cavitation::cavitation_controller_pressure(
        current_p_pa,
        stable_emission,
        inertial_emission,
        stable_target,
        inertial_limit,
        gain,
        p_min_pa,
        p_max_pa,
    ))
}

/// Incoherent power sum of per-element PCD spectra (array integration over V_s).
///
/// Given a (n_channels, n_bins) matrix of per-element power spectra, returns the
/// array-integrated spectrum S(f) = sum_ch S_ch(f). Passive emissions from
/// independent collapse events are mutually incoherent, so their powers add.
///
/// Args:
///     channel_psds: (n_channels, n_bins) power spectra.
///
/// Returns:
///     Array-integrated PSD of length n_bins.
///
/// Reference:
///     Gyongy & Coussios (2010) IEEE TBME 57, 48.
#[pyfunction]
#[pyo3(signature = (channel_psds))]
pub fn integrate_receiver_array_psd(
    py: Python<'_>,
    channel_psds: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let arr = channel_psds.as_array();
    let (n_channels, n_bins) = arr.dim();
    let flat: Vec<f64> = arr.iter().copied().collect();
    let result = cavitation::integrate_receiver_array_psd(&flat, n_channels, n_bins);
    Ok(result.into_pyarray(py).unbind())
}

/// Integrate a passive-acoustic-map emission-energy field over a sonication volume.
///
/// E(V_s) = sum_{voxels in mask} max(source[v], 0) * dv_m3
///
/// Args:
///     source_map: Flattened emission-energy field.
///     mask: Flattened V_s mask (non-zero = inside V_s), same length.
///     dv_m3: Voxel volume [m³].
///
/// Returns:
///     Total emission energy collected from V_s.
#[pyfunction]
#[pyo3(signature = (source_map, mask, dv_m3))]
pub fn emission_energy_in_volume(
    source_map: PyReadonlyArray1<f64>,
    mask: PyReadonlyArray1<f64>,
    dv_m3: f64,
) -> PyResult<f64> {
    let s = source_map
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let m = mask
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(cavitation::emission_energy_in_volume(s, m, dv_m3))
}

// ── Frequency-swept (chirp) cavitation control ───────────────────────────────

/// Parse a sweep-profile string into the physics enum.
fn parse_sweep_profile(profile: &str) -> PyResult<cavitation::SweepProfile> {
    match profile.to_ascii_lowercase().as_str() {
        "linear" => Ok(cavitation::SweepProfile::Linear),
        "triangular" | "triangle" => Ok(cavitation::SweepProfile::Triangular),
        other => Err(PyRuntimeError::new_err(format!(
            "unknown sweep profile '{other}' (expected 'linear' or 'triangular')"
        ))),
    }
}

/// Build a soft-tissue cavitation medium from explicit parameters.
#[allow(clippy::too_many_arguments)]
fn tissue_medium(
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
    c_liquid: f64,
) -> cavitation::CavitationMedium {
    cavitation::CavitationMedium {
        p0_pa,
        rho,
        sigma,
        mu,
        kappa,
        p_v_pa,
        c_liquid,
    }
}

/// Swept-frequency versus monochromatic nuclei engagement.
///
/// Integrates the Keller–Miksis response over a log-normal nuclei population to
/// compare the fraction of nuclei driven into inertial collapse (R_max/R₀ ≥
/// `inertial_ratio`) by a frequency-swept drive versus a single tone (the sweep
/// mean frequency), within a pulse of `pulse_duration_s`. The pulse window caps
/// both integrations, so a microsecond pulse realizes no swept advantage while a
/// millisecond pulse realizes the full enhancement.
///
/// Returns:
///     (mono_fraction, swept_fraction, enhancement_factor, covered_lo_hz,
///      covered_hi_hz).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    median_radius_m, geometric_std, f_start_hz, f_end_hz, sweep_period_s, profile,
    amplitude_pa, pulse_duration_s, n_size_samples=41, inertial_ratio=2.0,
    p0_pa=101_325.0, rho=1050.0, sigma=0.060, mu=1.5e-3, kappa=1.4,
    p_v_pa=2339.0, c_liquid=1540.0
))]
pub fn swept_vs_monochromatic_engagement(
    median_radius_m: f64,
    geometric_std: f64,
    f_start_hz: f64,
    f_end_hz: f64,
    sweep_period_s: f64,
    profile: &str,
    amplitude_pa: f64,
    pulse_duration_s: f64,
    n_size_samples: usize,
    inertial_ratio: f64,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
    c_liquid: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let dist = cavitation::NucleiSizeDistribution::new(median_radius_m, geometric_std)
        .ok_or_else(|| PyRuntimeError::new_err("invalid nuclei size distribution parameters"))?;
    let sweep = cavitation::FrequencySweep::new(
        f_start_hz,
        f_end_hz,
        sweep_period_s,
        parse_sweep_profile(profile)?,
    )
    .ok_or_else(|| PyRuntimeError::new_err("invalid frequency-sweep parameters"))?;
    let medium = tissue_medium(p0_pa, rho, sigma, mu, kappa, p_v_pa, c_liquid);
    let cfg = cavitation::EngagementConfig {
        n_size_samples,
        inertial_ratio,
        ..cavitation::EngagementConfig::default()
    };
    let r = cavitation::swept_vs_monochromatic_engagement(
        &dist,
        &medium,
        &sweep,
        amplitude_pa,
        pulse_duration_s,
        &cfg,
    );
    Ok((
        r.mono_fraction,
        r.swept_fraction,
        r.enhancement_factor,
        r.covered_band_hz.0,
        r.covered_band_hz.1,
    ))
}

/// Peak expansion ratio R_max/R₀ of a single nucleus under a chirped drive
/// (inertial-collapse / fragmentation discriminant).
///
/// Returns:
///     R_max/R₀ (dimensionless).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    r0_m, f_start_hz, f_end_hz, sweep_period_s, profile, amplitude_pa,
    pulse_duration_s, steps_per_cycle=64,
    p0_pa=101_325.0, rho=1050.0, sigma=0.060, mu=1.5e-3, kappa=1.4,
    p_v_pa=2339.0, c_liquid=1540.0
))]
pub fn chirped_peak_expansion_ratio(
    r0_m: f64,
    f_start_hz: f64,
    f_end_hz: f64,
    sweep_period_s: f64,
    profile: &str,
    amplitude_pa: f64,
    pulse_duration_s: f64,
    steps_per_cycle: usize,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
    c_liquid: f64,
) -> PyResult<f64> {
    let sweep = cavitation::FrequencySweep::new(
        f_start_hz,
        f_end_hz,
        sweep_period_s,
        parse_sweep_profile(profile)?,
    )
    .ok_or_else(|| PyRuntimeError::new_err("invalid frequency-sweep parameters"))?;
    let f_res = f_start_hz.max(f_end_hz).max(1.0);
    let n = ((pulse_duration_s * f_res).max(1.0) * steps_per_cycle as f64) as usize;
    let n = n.clamp(steps_per_cycle, 400_000);
    let dt = pulse_duration_s / n as f64;
    let t_arr: Vec<f64> = (0..=n).map(|i| i as f64 * dt).collect();
    Ok(cavitation::chirped_peak_expansion_ratio(
        &sweep,
        amplitude_pa,
        r0_m,
        &t_arr,
        p0_pa,
        rho,
        sigma,
        mu,
        kappa,
        p_v_pa,
        0.0,
        c_liquid,
    ))
}

/// Inter-pulse residual-bubble clearance under a fragmenting clearing sweep.
///
/// Compares the residual void fraction left at the next pulse by passive
/// Epstein–Plesset dissolution versus dissolution after a sweep fragments the
/// residual bubble into `fragment_count` gas-volume-conserving daughters (which
/// dissolve faster, τ ∝ R²).
///
/// Returns:
///     (residual_radius_passive_m, residual_radius_swept_m,
///      void_fraction_passive, void_fraction_swept, clearance_gain).
#[pyfunction]
#[pyo3(signature = (
    initial_void_fraction, initial_radius_m, interval_s, fragment_count,
    saturation_fraction=0.7
))]
pub fn inter_pulse_residual_clearance(
    initial_void_fraction: f64,
    initial_radius_m: f64,
    interval_s: f64,
    fragment_count: f64,
    saturation_fraction: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let params = cavitation::tissue_gas_diffusion(saturation_fraction);
    let c = cavitation::inter_pulse_residual_clearance(
        initial_void_fraction,
        initial_radius_m,
        interval_s,
        fragment_count,
        params,
    );
    Ok((
        c.residual_radius_passive_m,
        c.residual_radius_swept_m,
        c.void_fraction_passive,
        c.void_fraction_swept,
        c.clearance_gain,
    ))
}

/// Epstein–Plesset dissolution time R₀ → 0 [s] for a residual tissue bubble.
///
/// Returns:
///     Dissolution time [s], or None if the bubble does not dissolve (f ≥ 1).
#[pyfunction]
#[pyo3(signature = (r0_m, saturation_fraction=0.7))]
pub fn residual_dissolution_time_s(r0_m: f64, saturation_fraction: f64) -> PyResult<Option<f64>> {
    let params = cavitation::tissue_gas_diffusion(saturation_fraction);
    Ok(cavitation::residual_dissolution_time_s(r0_m, params))
}

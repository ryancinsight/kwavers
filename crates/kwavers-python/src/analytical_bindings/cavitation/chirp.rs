//! Frequency-swept cavitation and shielding-control PyO3 wrappers.

use kwavers_physics::analytical::cavitation;
use numpy::{PyArray1, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

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

/// Cavitation-optimal drive frequency in a band (engaged-fraction argmax).
///
/// At sub-saturation amplitudes the engaged fraction is set by the inertial
/// threshold (lower frequency → larger, lower-threshold nuclei) as much as by
/// resonance, so the optimum is found by a direct scan.
///
/// Returns:
///     (f_optimal_hz, engaged_fraction_at_optimum).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    median_radius_m, geometric_std, f_lo_hz, f_hi_hz, amplitude_pa, pulse_duration_s,
    n_scan=15, n_size_samples=21,
    p0_pa=101_325.0, rho=1050.0, sigma=0.060, mu=1.5e-3, kappa=1.4,
    p_v_pa=2339.0, c_liquid=1540.0
))]
pub fn cavitation_optimal_frequency(
    median_radius_m: f64,
    geometric_std: f64,
    f_lo_hz: f64,
    f_hi_hz: f64,
    amplitude_pa: f64,
    pulse_duration_s: f64,
    n_scan: usize,
    n_size_samples: usize,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
    c_liquid: f64,
) -> PyResult<(f64, f64)> {
    let dist = cavitation::NucleiSizeDistribution::new(median_radius_m, geometric_std)
        .ok_or_else(|| PyRuntimeError::new_err("invalid nuclei size distribution parameters"))?;
    let medium = tissue_medium(p0_pa, rho, sigma, mu, kappa, p_v_pa, c_liquid);
    let cfg = cavitation::EngagementConfig {
        n_size_samples,
        ..cavitation::EngagementConfig::default()
    };
    Ok(cavitation::cavitation_optimal_frequency(
        &dist,
        &medium,
        f_lo_hz,
        f_hi_hz,
        amplitude_pa,
        pulse_duration_s,
        n_scan,
        &cfg,
    ))
}

/// Staged-sonication frequency program: one slow up-and-down sweep across the
/// whole per-spot exposure.
///
/// The build half (stage 0→½) rises from the quiet frequency to the
/// cavitation-optimal turn (`f_peak_hz`), driving cavitation activity to a peak;
/// the wind-down half (½→1) returns to the quiet frequency, tapering new
/// cavitation and (with fragmentation enabled) clearing the residual so the next
/// sonication starts clean.
///
/// Returns:
///     (stage[], frequency_hz[], cavitation_activity[], residual_void[],
///      peak_activity_stage, residual_peak, residual_at_end).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    median_radius_m, geometric_std, f_quiet_hz, f_peak_hz, amplitude_pa, pulse_duration_s,
    n_pulses, prf_hz, void_deposit_per_activity=0.02, residual_radius_m=6e-6,
    clearing_fragment_count=8.0, saturation_fraction=0.7, n_size_samples=21,
    p0_pa=101_325.0, rho=1050.0, sigma=0.060, mu=1.5e-3, kappa=1.4,
    p_v_pa=2339.0, c_liquid=1540.0
))]
pub fn staged_sonication_sweep(
    py: Python<'_>,
    median_radius_m: f64,
    geometric_std: f64,
    f_quiet_hz: f64,
    f_peak_hz: f64,
    amplitude_pa: f64,
    pulse_duration_s: f64,
    n_pulses: usize,
    prf_hz: f64,
    void_deposit_per_activity: f64,
    residual_radius_m: f64,
    clearing_fragment_count: f64,
    saturation_fraction: f64,
    n_size_samples: usize,
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
    c_liquid: f64,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
)> {
    let dist = cavitation::NucleiSizeDistribution::new(median_radius_m, geometric_std)
        .ok_or_else(|| PyRuntimeError::new_err("invalid nuclei size distribution parameters"))?;
    let medium = tissue_medium(p0_pa, rho, sigma, mu, kappa, p_v_pa, c_liquid);
    let diffusion = cavitation::tissue_gas_diffusion(saturation_fraction);
    let cfg = cavitation::EngagementConfig {
        n_size_samples,
        ..cavitation::EngagementConfig::default()
    };
    let p = cavitation::staged_sonication_sweep(
        &dist,
        &medium,
        f_quiet_hz,
        f_peak_hz,
        amplitude_pa,
        pulse_duration_s,
        n_pulses,
        prf_hz,
        void_deposit_per_activity,
        residual_radius_m,
        clearing_fragment_count,
        diffusion,
        &cfg,
    );
    Ok((
        p.stage.to_pyarray(py).unbind(),
        p.frequency_hz.to_pyarray(py).unbind(),
        p.cavitation_activity.to_pyarray(py).unbind(),
        p.residual_void.to_pyarray(py).unbind(),
        p.peak_activity_stage,
        p.residual_peak,
        p.residual_at_end,
    ))
}

/// Build a `ShieldingMedium` from explicit focal-region parameters.
#[allow(clippy::too_many_arguments)]
fn shielding_medium(
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
    r0_m: f64,
    alpha_tissue_np_m: f64,
    path_len_m: f64,
    saturation_fraction: f64,
) -> cavitation::ShieldingMedium {
    cavitation::ShieldingMedium {
        c_liquid,
        rho_liquid,
        mu_liquid,
        p0_pa,
        polytropic,
        r0_m,
        alpha_tissue_np_m,
        path_len_m,
        saturation_fraction,
    }
}

/// Build a `CavitationProduction` source from explicit parameters.
fn shielding_production(
    k_prod_per_s: f64,
    beta_max: f64,
    p_threshold_pa: f64,
    p_ref_pa: f64,
    supralinearity: f64,
) -> cavitation::CavitationProduction {
    cavitation::CavitationProduction {
        k_prod_per_s,
        beta_max,
        p_threshold_pa,
        p_ref_pa,
        supralinearity,
    }
}

/// Time-resolved cavitation-shielding trace under a pulsed, optionally swept drive.
///
/// Integrates the focal void-fraction balance with self-shielding feedback (the
/// delivered focal pressure is derated by Commander–Prosperetti attenuation of
/// the accumulating cloud) and Epstein–Plesset inter-pulse clearance. `freq_mode`
/// is `"fixed"` (constant `f_fixed_hz`) or `"swept"` (chirp over
/// `[f_start_hz, f_end_hz]` with `sweep_period_s` and `profile`).
///
/// Returns:
///     (time_s, void_fraction, delivered_fraction, delivered_pressure_pa,
///      peak_void_fraction, mean_void_fraction, mean_delivered_fraction_on,
///      delivered_energy, unshielded_energy, shielding_loss_fraction).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    drive_pressure_pa, freq_mode, f_fixed_hz, f_start_hz, f_end_hz, sweep_period_s,
    profile, pulse_on_s, pulse_off_s, total_time_s, dt_s,
    k_prod_per_s=50.0, beta_max=1.0e-2, p_threshold_pa=1.0e6, p_ref_pa=1.0e6,
    supralinearity=3.0, c_liquid=1540.0, rho_liquid=1050.0, mu_liquid=1.5e-3,
    p0_pa=101_325.0, polytropic=1.4, r0_m=2.0e-6, alpha_tissue_np_m=5.0,
    path_len_m=0.04, saturation_fraction=0.9
))]
pub fn simulate_shielding_trace(
    py: Python<'_>,
    drive_pressure_pa: f64,
    freq_mode: &str,
    f_fixed_hz: f64,
    f_start_hz: f64,
    f_end_hz: f64,
    sweep_period_s: f64,
    profile: &str,
    pulse_on_s: f64,
    pulse_off_s: f64,
    total_time_s: f64,
    dt_s: f64,
    k_prod_per_s: f64,
    beta_max: f64,
    p_threshold_pa: f64,
    p_ref_pa: f64,
    supralinearity: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
    r0_m: f64,
    alpha_tissue_np_m: f64,
    path_len_m: f64,
    saturation_fraction: f64,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
)> {
    let freq = match freq_mode.to_ascii_lowercase().as_str() {
        "fixed" => cavitation::DriveFrequency::Fixed(f_fixed_hz),
        "swept" => {
            let sweep = cavitation::FrequencySweep::new(
                f_start_hz,
                f_end_hz,
                sweep_period_s,
                parse_sweep_profile(profile)?,
            )
            .ok_or_else(|| PyRuntimeError::new_err("invalid frequency-sweep parameters"))?;
            cavitation::DriveFrequency::Swept(sweep)
        }
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "unknown freq_mode '{other}' (expected 'fixed' or 'swept')"
            )))
        }
    };
    let protocol = cavitation::PulseProtocol::pulsed(pulse_on_s, pulse_off_s);
    let prod = shielding_production(
        k_prod_per_s,
        beta_max,
        p_threshold_pa,
        p_ref_pa,
        supralinearity,
    );
    let medium = shielding_medium(
        c_liquid,
        rho_liquid,
        mu_liquid,
        p0_pa,
        polytropic,
        r0_m,
        alpha_tissue_np_m,
        path_len_m,
        saturation_fraction,
    );
    let cfg = cavitation::ShieldingConfig { total_time_s, dt_s };
    let t =
        cavitation::simulate_shielding(drive_pressure_pa, &freq, &protocol, &prod, &medium, &cfg);
    Ok((
        t.time_s.to_pyarray(py).unbind(),
        t.void_fraction.to_pyarray(py).unbind(),
        t.delivered_fraction.to_pyarray(py).unbind(),
        t.delivered_pressure_pa.to_pyarray(py).unbind(),
        t.peak_void_fraction,
        t.mean_void_fraction,
        t.mean_delivered_fraction_on,
        t.delivered_energy,
        t.unshielded_energy,
        t.shielding_loss_fraction,
    ))
}

/// 2×2 cavitation-shielding control comparison: {CW, pulsed} × {fixed, swept}.
///
/// The fixed tone uses the sweep mean frequency, so the rows isolate the control
/// strategy. Each row is summarised by five metrics `(peak_void_fraction,
/// mean_void_fraction, mean_delivered_fraction_on, delivered_energy,
/// shielding_loss_fraction)`. `mean_delivered_fraction_on` is the duty-fair
/// efficacy metric (transmission while driving); `delivered_energy` is absolute
/// and therefore scales with duty cycle.
///
/// Returns a flat 20-element list ordered cw_fixed, cw_swept, pulsed_fixed,
/// pulsed_swept, each contributing its five metrics in the order above.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    drive_pressure_pa, f_start_hz, f_end_hz, sweep_period_s, profile,
    pulse_on_s, pulse_off_s, total_time_s, dt_s,
    k_prod_per_s=50.0, beta_max=1.0e-2, p_threshold_pa=1.0e6, p_ref_pa=1.0e6,
    supralinearity=3.0, c_liquid=1540.0, rho_liquid=1050.0, mu_liquid=1.5e-3,
    p0_pa=101_325.0, polytropic=1.4, r0_m=2.0e-6, alpha_tissue_np_m=5.0,
    path_len_m=0.04, saturation_fraction=0.9
))]
pub fn compare_shielding_control(
    drive_pressure_pa: f64,
    f_start_hz: f64,
    f_end_hz: f64,
    sweep_period_s: f64,
    profile: &str,
    pulse_on_s: f64,
    pulse_off_s: f64,
    total_time_s: f64,
    dt_s: f64,
    k_prod_per_s: f64,
    beta_max: f64,
    p_threshold_pa: f64,
    p_ref_pa: f64,
    supralinearity: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
    r0_m: f64,
    alpha_tissue_np_m: f64,
    path_len_m: f64,
    saturation_fraction: f64,
) -> PyResult<Vec<f64>> {
    let sweep = cavitation::FrequencySweep::new(
        f_start_hz,
        f_end_hz,
        sweep_period_s,
        parse_sweep_profile(profile)?,
    )
    .ok_or_else(|| PyRuntimeError::new_err("invalid frequency-sweep parameters"))?;
    let protocol = cavitation::PulseProtocol::pulsed(pulse_on_s, pulse_off_s);
    let prod = shielding_production(
        k_prod_per_s,
        beta_max,
        p_threshold_pa,
        p_ref_pa,
        supralinearity,
    );
    let medium = shielding_medium(
        c_liquid,
        rho_liquid,
        mu_liquid,
        p0_pa,
        polytropic,
        r0_m,
        alpha_tissue_np_m,
        path_len_m,
        saturation_fraction,
    );
    let cfg = cavitation::ShieldingConfig { total_time_s, dt_s };
    let c = cavitation::compare_shielding_control(
        drive_pressure_pa,
        &sweep,
        &protocol,
        &prod,
        &medium,
        &cfg,
    );
    let row = |s: &cavitation::ShieldingSummary| {
        [
            s.peak_void_fraction,
            s.mean_void_fraction,
            s.mean_delivered_fraction_on,
            s.delivered_energy,
            s.shielding_loss_fraction,
        ]
    };
    let mut out = Vec::with_capacity(20);
    out.extend_from_slice(&row(&c.cw_fixed));
    out.extend_from_slice(&row(&c.cw_swept));
    out.extend_from_slice(&row(&c.pulsed_fixed));
    out.extend_from_slice(&row(&c.pulsed_swept));
    Ok(out)
}

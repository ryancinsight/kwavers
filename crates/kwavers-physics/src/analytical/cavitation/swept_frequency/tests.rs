//! Tests for the swept-frequency cavitation-control module.

use super::*;

const PI2: f64 = std::f64::consts::TAU;

fn small_cfg() -> EngagementConfig {
    EngagementConfig {
        n_size_samples: 17,
        n_sigma: 3.0,
        inertial_ratio: 2.0,
        steps_per_cycle: 32,
        mono_cycles: 40.0,
        max_sweep_periods: 2.0,
        max_steps: 80_000,
    }
}

// ── chirp waveform ──────────────────────────────────────────────────────────

#[test]
fn chirp_instantaneous_frequency_hits_endpoints() {
    let s = FrequencySweep::new(0.8e6, 1.4e6, 40e-6, SweepProfile::Triangular).unwrap();
    // Triangular: f_start at t=0, f_end at the half-period turn, back at the end.
    assert!((s.instantaneous_frequency(0.0) - 0.8e6).abs() < 1.0);
    assert!((s.instantaneous_frequency(20e-6) - 1.4e6).abs() < 1.0);
    assert!((s.instantaneous_frequency(40e-6) - 0.8e6).abs() < 1.0);
    // Linear: monotone ramp reaching f_end at the period end.
    let l = FrequencySweep::new(0.8e6, 1.4e6, 40e-6, SweepProfile::Linear).unwrap();
    assert!((l.instantaneous_frequency(40e-6 * 0.999_999) - 1.4e6).abs() < 100.0);
}

#[test]
fn chirp_phase_derivative_matches_two_pi_f() {
    let s = FrequencySweep::new(0.7e6, 1.3e6, 50e-6, SweepProfile::Linear).unwrap();
    let dt = 1e-10;
    for &t in &[3e-6, 12e-6, 30e-6, 44e-6] {
        let num = (s.phase(t + dt) - s.phase(t - dt)) / (2.0 * dt);
        let analytic = PI2 * s.instantaneous_frequency(t);
        let rel = (num - analytic).abs() / analytic.abs();
        assert!(
            rel < 1e-4,
            "dφ/dt mismatch at t={t}: num={num}, analytic={analytic}"
        );
    }
}

#[test]
fn chirp_pressure_bounded_by_amplitude() {
    let s = FrequencySweep::new(0.9e6, 1.1e6, 30e-6, SweepProfile::Triangular).unwrap();
    for i in 0..200 {
        let t = i as f64 * 1e-6;
        assert!(s.pressure(t, 2.0e6).abs() <= 2.0e6 + 1.0);
    }
}

#[test]
fn chirp_covered_band_respects_pulse_window() {
    let s = FrequencySweep::new(0.7e6, 1.3e6, 40e-6, SweepProfile::Triangular).unwrap();
    // A pulse spanning >= half a triangular period covers the full band.
    let (lo, hi) = s.covered_band_hz(1e-3);
    assert!((lo - 0.7e6).abs() < 1.0 && (hi - 1.3e6).abs() < 1.0);
    // A microsecond pulse (≪ period) covers only a sliver near f_start.
    let (lo_us, hi_us) = s.covered_band_hz(1e-6);
    assert!(hi_us - lo_us < 0.1 * s.bandwidth_hz());
}

#[test]
fn chirp_rejects_nonphysical() {
    assert!(FrequencySweep::new(-1.0, 1e6, 1e-5, SweepProfile::Linear).is_none());
    assert!(FrequencySweep::new(1e6, 1e6, 0.0, SweepProfile::Linear).is_none());
}

// ── nuclei distribution ──────────────────────────────────────────────────────

#[test]
fn nuclei_pdf_integrates_to_unity() {
    let d = NucleiSizeDistribution::new(3.3e-6, 1.7).unwrap();
    // Number fraction over a very wide band must approach 1.
    let frac = d.number_fraction_in_radius_band(1e-9, 1e-3);
    assert!((frac - 1.0).abs() < 1e-3, "wide-band fraction = {frac}");
}

#[test]
fn nuclei_resonance_inverts_minnaert() {
    let (kappa, p0, rho) = (1.4, 101_325.0, 1050.0);
    let r = NucleiSizeDistribution::resonant_radius_for_frequency(1.0e6, kappa, p0, rho);
    // Round-trip: the Minnaert resonance of that radius is 1 MHz.
    let d = NucleiSizeDistribution::new(r, 1.5).unwrap();
    let f = d.median_resonance_hz(kappa, p0, rho);
    assert!((f - 1.0e6).abs() / 1.0e6 < 1e-9, "round-trip freq = {f}");
}

#[test]
fn nuclei_resonant_band_is_positive_fraction() {
    let d = NucleiSizeDistribution::new(3.3e-6, 1.7).unwrap();
    // A band centred near the median resonance engages a sizeable fraction.
    let frac = d.number_fraction_resonant_in_band(0.6e6, 1.6e6, 1.4, 101_325.0, 1050.0);
    assert!(frac > 0.2 && frac <= 1.0, "resonant-band fraction = {frac}");
}

#[test]
fn nuclei_sample_weights_sum_near_one() {
    let d = NucleiSizeDistribution::new(3.3e-6, 1.7).unwrap();
    let (radii, weights) = d.sample_radii(41, 3.0);
    assert_eq!(radii.len(), 41);
    let total: f64 = weights.iter().sum();
    // ±3σ of a log-normal holds ≈ 99.7 % of the mass.
    assert!(total > 0.99 && total <= 1.0, "weight sum = {total}");
}

// ── chirped dynamics ─────────────────────────────────────────────────────────

#[test]
fn chirped_km_runs_finite_and_expands_with_amplitude() {
    let s = FrequencySweep::new(0.8e6, 1.2e6, 30e-6, SweepProfile::Triangular).unwrap();
    let m = CavitationMedium::soft_tissue();
    let t: Vec<f64> = (0..=4000).map(|i| i as f64 * 1e-8).collect();
    let ratio = |amp: f64| {
        chirped_peak_expansion_ratio(
            &s, amp, 3.3e-6, &t, m.p0_pa, m.rho, m.sigma, m.mu, m.kappa, m.p_v_pa, 0.0, m.c_liquid,
        )
    };
    let lo = ratio(0.2e6);
    let hi = ratio(1.2e6);
    assert!(lo.is_finite() && hi.is_finite());
    assert!(hi > lo, "stronger drive must expand more: lo={lo}, hi={hi}");
    assert!(hi >= 1.0);
}

// ── engagement: swept enhancement and the ms-vs-µs asymmetry ──────────────────

#[test]
fn swept_engages_at_least_as_much_as_tone_for_long_pulse() {
    let d = NucleiSizeDistribution::new(3.3e-6, 1.7).unwrap();
    let m = CavitationMedium::soft_tissue();
    let s = FrequencySweep::new(0.6e6, 1.4e6, 40e-6, SweepProfile::Triangular).unwrap();
    let cfg = small_cfg();
    let r = swept_vs_monochromatic_engagement(&d, &m, &s, 1.0e6, 1e-3, &cfg);
    assert!(r.swept_fraction >= r.mono_fraction - 1e-9, "{r:?}");
    assert!(r.swept_fraction > 0.0, "{r:?}");
    // Sweeping a broad band engages a wider size population than one tone.
    assert!(r.enhancement_factor >= 1.0, "{r:?}");
}

#[test]
fn swept_enhancement_larger_for_ms_than_us_pulse() {
    let d = NucleiSizeDistribution::new(3.3e-6, 1.7).unwrap();
    let m = CavitationMedium::soft_tissue();
    let s = FrequencySweep::new(0.6e6, 1.4e6, 40e-6, SweepProfile::Triangular).unwrap();
    let cfg = small_cfg();
    // Resonance-selective amplitude: only resonant, fully-rung-up bubbles reach
    // the inertial criterion (a far-supra-threshold drive collapses every size
    // regardless of resonance or pulse length, masking the effect).
    let amp = 0.15e6;
    let ms = swept_vs_monochromatic_engagement(&d, &m, &s, amp, 1e-3, &cfg);
    let us = swept_vs_monochromatic_engagement(&d, &m, &s, amp, 2e-6, &cfg);
    println!("ms={ms:?}\nus={us:?}");
    // The ms pulse engages a larger fraction of the nuclei population by sweeping
    // (it covers the whole band and the bubbles ring up); the µs (≈ single-cycle)
    // pulse cannot traverse the band nor ring up, so it engages strictly less.
    assert!(
        ms.swept_fraction > us.swept_fraction,
        "ms swept {} !> us swept {}",
        ms.swept_fraction,
        us.swept_fraction
    );
    // And the µs pulse covers far less of the band than the ms pulse.
    let ms_band = ms.covered_band_hz.1 - ms.covered_band_hz.0;
    let us_band = us.covered_band_hz.1 - us.covered_band_hz.0;
    assert!(us_band < ms_band, "ms band {ms_band}, us band {us_band}");
}

// ── inter-pulse clearance ────────────────────────────────────────────────────

#[test]
fn clearing_sweep_lowers_residual_void_fraction() {
    let params = tissue_gas_diffusion(0.7); // undersaturated tissue
    let c = inter_pulse_residual_clearance(0.02, 6e-6, 0.5, 8.0, params);
    // Fragmentation leaves less residual gas than passive dissolution.
    assert!(
        c.void_fraction_swept < c.void_fraction_passive,
        "swept {} !< passive {}",
        c.void_fraction_swept,
        c.void_fraction_passive
    );
    assert!(
        c.clearance_gain > 1.0,
        "clearance gain = {}",
        c.clearance_gain
    );
    // Smaller daughters than the intact residual.
    assert!(c.residual_radius_swept_m < c.residual_radius_passive_m);
}

#[test]
fn no_fragmentation_recovers_passive() {
    let params = tissue_gas_diffusion(0.7);
    let c = inter_pulse_residual_clearance(0.02, 6e-6, 0.5, 1.0, params);
    assert!((c.clearance_gain - 1.0).abs() < 1e-9, "{c:?}");
    assert!((c.void_fraction_swept - c.void_fraction_passive).abs() < 1e-12);
}

#[test]
fn dissolution_time_scales_with_radius_squared() {
    let params = tissue_gas_diffusion(0.5);
    let t1 = residual_dissolution_time_s(2e-6, params).unwrap();
    let t2 = residual_dissolution_time_s(4e-6, params).unwrap();
    // τ ∝ R₀² ⇒ doubling the radius quadruples the dissolution time.
    assert!((t2 / t1 - 4.0).abs() < 0.2, "ratio = {}", t2 / t1);
}

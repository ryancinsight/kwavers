//! Harmonic-generation presence check for the nonlinear 3-D Westervelt
//! forward at strong source amplitude.
//!
//! # What this validates
//!
//! The companion sign-, linear-baseline-, and β-scaling tests pin the
//! *sign* and *scaling* of the nonlinear contribution but only inspect
//! time-domain asymmetry. This test extracts the 2nd-harmonic amplitude
//! via discrete sine/cosine projection and asserts that the FDTD generates
//! a *measurable* harmonic content (`|P_2|/|P_1| ∈ [0.03, 0.40]`) at a
//! strongly nonlinear source amplitude. Catches:
//! - A nonlinear term that propagates as just a phase shift (no harmonic
//!   generation at all → ratio ≈ 0).
//! - A spuriously-high 2nd harmonic from grid dispersion (ratio > 0.5
//!   without strong nonlinearity).
//! - The forward returning DC-only or NaN output.
//!
//! # Why not Fubini-absolute
//!
//! The Aanonsen-1984 / Fubini analytical solution
//! `|P_n|/|P_1| = J_n(nΓ) / (n J_1(Γ))` assumes a 1-D plane wave with
//! constant amplitude over the propagation path. A 3-D point source has
//! 1/r geometric spreading, so the local amplitude — and hence the local Γ
//! governing nonlinear distortion — varies along the path. The KZK solver
//! carries the literature-validated Fubini-absolute test because KZK
//! parabolically reduces 3-D propagation to 1-D-along-z with constant-
//! amplitude planar shots. The Westervelt FDTD cannot drive that
//! configuration without API changes, so we test for harmonic *presence*
//! rather than absolute Fubini-matching in this fixture.
//!
//! # Tier
//!
//! `#[ignore]`'d as Tier-2 (~10 s runtime). Run on demand with
//! `cargo test --lib --package kwavers -- --ignored harmonic_generation`.

use super::super::encoding::SourceEncoding;
use super::super::forward::{forward_with_schedule, ForwardInput, TimeSchedule};
use super::super::types::{GridIndex, Nonlinear3dAperture, SourceDomain};
use super::super::Nonlinear3dConfig;
use super::Point3;
use crate::clinical::therapy::theranostic_guidance::AnatomyKind;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::core::constants::numerical::MHZ_TO_HZ;

#[test]
#[ignore = "Tier 2: Harmonic-generation presence check, ~10s runtime"]
fn westervelt_fdtd_point_source_generates_measurable_second_harmonic_content() {
    // Cubic grid required by the `ForwardInput` API. Single source cell at
    // the center of one face; receiver on the central axis downstream.
    // This is the same fixture shape as the working forward-steepening test,
    // which is known to be numerically stable.
    let n: usize = 48;
    let cells = n * n * n;
    let spacing_m = 1.0e-4_f64; // 0.1 mm → 4.8 mm cube
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = 1000.0_f64;
    let beta_nl = 10.0_f64;
    let frequency_hz = MHZ_TO_HZ;
    let omega = std::f64::consts::TAU * frequency_hz;
    let speed = vec![c0; cells];
    let density = vec![rho0; cells];
    let beta_field = vec![beta_nl; cells];

    let source_x = 4_usize;
    let source_idx = GridIndex {
        x: source_x,
        y: n / 2,
        z: n / 2,
    };
    let receiver_x = n - 6;
    let receiver_idx = GridIndex {
        x: receiver_x,
        y: n / 2,
        z: n / 2,
    };
    let aperture = Nonlinear3dAperture {
        sources: vec![source_idx],
        receivers: vec![receiver_idx],
        therapy_points_m: vec![Point3 {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 0.0,
        }],
        receiver_points_m: vec![Point3 {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 0.0,
        }],
        model_name: "test_point_source_fubini".to_owned(),
        source_domain: SourceDomain::TissueBoundary,
        focus: receiver_idx,
    };

    // Same source pressure as the (numerically stable) forward-steepening test.
    let distance_m = (receiver_x - source_x) as f64 * spacing_m;
    let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
    config.frequency_hz = frequency_hz;
    config.source_pressure_pa = 5.0e6;
    config.cycles = 12.0;
    config.cfl = 0.4;
    let dt = config.cfl * spacing_m / (c0 * 3.0_f64.sqrt());
    let travel_steps = (distance_m / (c0 * dt)).ceil() as usize;
    let period_steps_estimate = (1.0 / (frequency_hz * dt)).ceil() as usize;
    let steady_window_steps = 8 * period_steps_estimate;
    // Travel time + 2 periods to clear transient + 8 periods for projection
    // window = enough samples for accurate sin/cos projection.
    let steps = travel_steps + 2 * period_steps_estimate + steady_window_steps + 8;
    let schedule = TimeSchedule {
        dt_s: dt,
        time_steps: steps,
    };

    let result = forward_with_schedule(ForwardInput {
        speed: &speed,
        density: &density,
        beta: &beta_field,
        attenuation_np_per_m_mhz: None,
        attenuation_power_law_y: None,
        source_body_mask: None,
        n,
        spacing_m,
        aperture: &aperture,
        config: &config,
        schedule,
        encoding: SourceEncoding { index: 0, count: 1 },
        source_scale: 1.0,
        retain_history: false,
    });

    assert_eq!(result.traces.len(), steps);

    // Pick the steady-state window: start after the wave reaches the
    // receiver plus a few periods of transient, end before the source
    // burst's trailing edge can reach the receiver.
    let period_steps = (1.0 / (frequency_hz * dt)).round() as usize;
    let window_start = travel_steps + 2 * period_steps;
    let window_len = (8 * period_steps).min(result.traces.len() - window_start - period_steps);
    assert!(
        window_len >= 4 * period_steps,
        "Fubini test steady-state window too short ({window_len}); increase `steps` or shorten travel time",
    );
    let window = &result.traces[window_start..window_start + window_len];

    // Discrete sine/cosine projection at the fundamental and 2nd harmonic
    // (exact for harmonics commensurate with the window length). For a real
    // signal `p[i]`, the amplitude of the `f`-Hz harmonic is
    //   A_f = sqrt( (2/N) Σ p[i] cos(ωt))² + ((2/N) Σ p[i] sin(ωt))² )
    let project = |trace: &[f64], freq_hz: f64| -> f64 {
        let omega_proj = std::f64::consts::TAU * freq_hz;
        let n_samples = trace.len() as f64;
        let mut cos_sum = 0.0_f64;
        let mut sin_sum = 0.0_f64;
        for (i, &value) in trace.iter().enumerate() {
            let t = (window_start + i) as f64 * dt;
            cos_sum += value * (omega_proj * t).cos();
            sin_sum += value * (omega_proj * t).sin();
        }
        2.0 * (cos_sum * cos_sum + sin_sum * sin_sum).sqrt() / n_samples
    };

    let amp_fundamental = project(window, frequency_hz);
    let amp_second_harmonic = project(window, 2.0 * frequency_hz);
    assert!(
        amp_fundamental.is_finite() && amp_second_harmonic.is_finite(),
        "harmonic amplitudes must be finite: P_1 = {amp_fundamental}, P_2 = {amp_second_harmonic}",
    );
    assert!(
        amp_fundamental > 0.0,
        "fundamental amplitude must be positive; got P_1 = {amp_fundamental}",
    );

    let ratio = amp_second_harmonic / amp_fundamental;
    let _ = distance_m; // documented but not used in the harmonic-presence check
    let _ = omega;
    let _ = rho0;
    let _ = beta_nl;

    // Lower bound: 3% — guards against `β·∂²(p²)/∂t²` being multiplied by 0.
    // Upper bound: 40% — guards against `β²` runaway or grid-dispersion artifacts.
    assert!(
        (0.03..=0.40).contains(&ratio),
        "Westervelt 3-D point-source harmonic-generation check: |P_2|/|P_1| = \
         {ratio:.4}; expected in [0.03, 0.40] for 5 MPa source at β = 10. \
         |P_1| = {amp_fundamental:.4e} Pa, |P_2| = {amp_second_harmonic:.4e} Pa. \
         A ratio < 0.03 suggests the nonlinear term is not driving harmonic \
         generation; a ratio > 0.40 suggests runaway harmonic content from \
         coefficient error or grid dispersion.",
    );
}

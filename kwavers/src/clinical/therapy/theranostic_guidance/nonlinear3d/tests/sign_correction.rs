//! Sign-sensitive regression for the Westervelt forward.
//!
//! # Theorem (forward steepening, Hamilton & Blackstock 1998 §3.6)
//!
//! For a finite-amplitude wave propagating in a medium with `β > 0`, the
//! instantaneous local sound speed is `c(p) ≈ c₀ + β·p/(ρ₀c₀)`. Compressions
//! (`p > 0`) travel faster than rarefactions, producing forward steepening:
//! at a fixed downstream receiver the rising edge of each cycle sharpens
//! while the falling edge stretches. This makes
//! `max_t ∂p/∂t  >  |min_t ∂p/∂t|`.
//!
//! A Westervelt FDTD with the *correct* sign on `∂²(p²)/∂t²` (positive on
//! the explicit update of `p_{n+1}`) reproduces this asymmetry. A sign flip
//! inverts the asymmetry — peaks would *round* while troughs sharpen, which
//! is non-physical.

use super::super::encoding::SourceEncoding;
use super::super::forward::{forward_with_schedule, ForwardInput, TimeSchedule};
use super::super::types::{GridIndex, Nonlinear3dAperture, SourceDomain};
use super::super::Nonlinear3dConfig;
use super::Point3;
use crate::clinical::therapy::theranostic_guidance::AnatomyKind;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

#[test]
fn forward_westervelt_exhibits_physical_forward_steepening_with_corrected_sign() {
    let n = 24;
    let spacing_m = 4.0e-4_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = 1000.0_f64;
    let beta_nl = 10.0_f64;
    let cells = n * n * n;
    let speed = vec![c0; cells];
    let density = vec![rho0; cells];
    let beta_field = vec![beta_nl; cells];

    let source_idx = GridIndex {
        x: 2,
        y: n / 2,
        z: n / 2,
    };
    let receiver_idx = GridIndex {
        x: n - 3,
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
        model_name: "test_homogeneous_axial".to_owned(),
        source_domain: SourceDomain::TissueBoundary,
        focus: receiver_idx,
    };

    let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
    config.frequency_hz = 1.0e6;
    config.source_pressure_pa = 20.0e6;
    config.cycles = 12.0;
    config.cfl = 0.4;
    let dt = config.cfl * spacing_m / (c0 * 3.0_f64.sqrt());
    let steps = 360;
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
    let traces = &result.traces;

    // Ignore the leading linear travel horizon before energy reaches the receiver;
    // measure steepening on the steady-state window only.
    let linear_arrival_steps =
        ((n - 5) as f64 * spacing_m / (c0 * dt)).ceil() as usize + (steps / 8);
    let window = &traces[linear_arrival_steps.min(traces.len() - 2)..];
    assert!(
        window.len() >= 32,
        "steady-state window too short ({} samples); increase `steps` or shrink the travel horizon",
        window.len(),
    );

    let dp_dt: Vec<f64> = window.windows(2).map(|w| (w[1] - w[0]) / dt).collect();
    assert!(dp_dt.iter().all(|v| v.is_finite()));
    let dp_dt_max = dp_dt.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let dp_dt_min = dp_dt.iter().copied().fold(f64::INFINITY, f64::min);

    assert!(
        dp_dt_max > 0.0 && dp_dt_min < 0.0,
        "expected sign-changing time derivative; got max={dp_dt_max:.3e}, min={dp_dt_min:.3e}",
    );

    let ratio = dp_dt_max / dp_dt_min.abs();
    assert!(
        ratio > 1.0,
        "Westervelt sign regression: max(dp/dt)/|min(dp/dt)| = {ratio:.4}; \
         expected > 1.0 for the physically correct `+q·∂²(p²)/∂t²` Westervelt sign. \
         A ratio < 1 indicates a sign-flipped nonlinear term (reverse steepening).",
    );
}

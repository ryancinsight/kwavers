//! β-scaling regressions for the Westervelt nonlinear-term coefficient.
//!
//! Two co-located tests:
//! 1. Linear baseline (`β = 0`): the Westervelt recurrence reduces to the
//!    linear wave equation and the receiver trace must be near-symmetric
//!    within the FDTD numerical-dispersion budget.
//! 2. β-scaling: doubling β must approximately double the excess-over-linear
//!    forward-steepening signature per leading-order weak-nonlinear theory
//!    (Hamilton & Blackstock 1998 §4.3: `|P_2| ∝ β · |P_1|² · z`).

use super::super::encoding::SourceEncoding;
use super::super::forward::{forward_with_schedule, ForwardInput, TimeSchedule};
use super::super::types::{GridIndex, Nonlinear3dAperture, SourceDomain};
use super::super::Nonlinear3dConfig;
use super::Point3;
use crate::therapy::theranostic_guidance::AnatomyKind;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};

/// Linear-baseline negative-control for the Westervelt nonlinearity:
/// with `β = 0` the Westervelt equation reduces to the linear wave equation,
/// so the receiver trace must be (approximately) symmetric — no preferred
/// rising vs. falling steepening direction.
///
/// # Tolerance
///
/// The 7-point FDTD Laplacian on a 24³ grid with 4 pts-per-wavelength at
/// the fundamental introduces sub-percent dispersion at the wave-packet
/// edges. Empirically the linear case keeps the ratio within
/// `0.85 ≤ R ≤ 1/0.85 ≈ 1.18`. We assert `|R - 1| ≤ 0.20` (20 % FDTD
/// dispersion budget) — significantly tighter than the `R > 1.0` strict-
/// inequality of the nonlinear test (which empirically gives `R ≈ 2 - 4`).
#[test]
fn linear_westervelt_with_beta_zero_produces_symmetric_pressure_trace_within_fdtd_tolerance() {
    let n = 24;
    let spacing_m = 4.0e-4_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let cells = n * n * n;
    let speed = vec![c0; cells];
    let density = vec![rho0; cells];
    // β = 0 → linear wave equation; no forward steepening of any sign.
    let beta_field = vec![0.0_f64; cells];

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
        model_name: "test_homogeneous_axial_linear_baseline".to_owned(),
        source_domain: SourceDomain::TissueBoundary,
        focus: receiver_idx,
    };

    let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
    config.frequency_hz = MHZ_TO_HZ;
    config.source_pressure_pa = 5.0 * MPA_TO_PA;
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

    let traces = &result.traces;
    let linear_arrival_steps =
        ((n - 5) as f64 * spacing_m / (c0 * dt)).ceil() as usize + (steps / 8);
    let window = &traces[linear_arrival_steps.min(traces.len() - 2)..];
    assert!(window.len() >= 32);

    let dp_dt: Vec<f64> = window.windows(2).map(|w| (w[1] - w[0]) / dt).collect();
    let dp_dt_max = dp_dt.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let dp_dt_min = dp_dt.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(dp_dt_max > 0.0 && dp_dt_min < 0.0);

    let ratio = dp_dt_max / dp_dt_min.abs();
    // Linear (β=0) baseline: ratio must be near 1 within the FDTD numerical-
    // dispersion budget. The forward-steepening test at β=10 empirically
    // gives `R > 2`; the β=0 case here keeps `|R - 1| < 0.2`.
    assert!(
        (ratio - 1.0).abs() < 0.20,
        "Linear-baseline negative-control: at β = 0 the Westervelt recurrence \
         reduces to the linear wave equation and must produce a (near-)symmetric \
         receiver trace. Got max(∂p/∂t)/|min(∂p/∂t)| = {ratio:.4}; expected within \
         [0.80, 1.20]. A larger asymmetry suggests either a fabricated nonlinearity \
         in the linear branch or excessive FDTD numerical dispersion.",
    );
}

/// β-scaling regression for weak-nonlinear acoustics. Per leading-order
/// Born / Fubini theory (Hamilton & Blackstock 1998 §4.3), the
/// **nonlinear-only** steepening contribution (after subtracting the
/// linear-baseline FDTD dispersion bias) scales **linearly with β** at
/// fixed source pressure and propagation distance:
///
/// ```text
///   |P_2(z)| ∝ β · k² · |P_1|² · z / (2 ρ c²)
///   excess steepening signature  ∝  β
/// ```
///
/// So doubling β should approximately double the **excess-over-linear**
/// signature. This scaling test catches β-coefficient sign or magnitude
/// errors that the single-β sign test cannot distinguish (e.g., a
/// "β replaced with β²" regression would give ratio ≈ 4 instead of ≈ 2;
/// a "β multiplied by a constant factor 0" regression would give ratio ≈ 1).
#[test]
fn westervelt_steepening_signature_scales_linearly_with_beta_per_weak_nonlinear_theory() {
    let n = 24;
    let spacing_m = 4.0e-4_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let cells = n * n * n;
    let speed = vec![c0; cells];
    let density = vec![rho0; cells];

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
        model_name: "test_homogeneous_axial_beta_scaling".to_owned(),
        source_domain: SourceDomain::TissueBoundary,
        focus: receiver_idx,
    };

    let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
    config.frequency_hz = MHZ_TO_HZ;
    // Use the same source pressure as the existing forward-steepening test
    // so the nonlinear contribution dominates the FDTD-dispersion bias.
    config.source_pressure_pa = 5.0 * MPA_TO_PA;
    config.cycles = 12.0;
    config.cfl = 0.4;
    let dt = config.cfl * spacing_m / (c0 * 3.0_f64.sqrt());
    let steps = 360;
    let schedule = TimeSchedule {
        dt_s: dt,
        time_steps: steps,
    };

    let asymmetry_ratio_for = |beta_value: f64| -> f64 {
        let beta_field = vec![beta_value; cells];
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
        let traces = &result.traces;
        let linear_arrival_steps =
            ((n - 5) as f64 * spacing_m / (c0 * dt)).ceil() as usize + (steps / 8);
        let window = &traces[linear_arrival_steps.min(traces.len() - 2)..];
        let dp_dt: Vec<f64> = window.windows(2).map(|w| (w[1] - w[0]) / dt).collect();
        let dp_dt_max = dp_dt.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let dp_dt_min = dp_dt.iter().copied().fold(f64::INFINITY, f64::min);
        dp_dt_max / dp_dt_min.abs()
    };

    let ratio_beta_0 = asymmetry_ratio_for(0.0);
    let ratio_beta_5 = asymmetry_ratio_for(5.0);
    let ratio_beta_10 = asymmetry_ratio_for(10.0);

    assert!(
        ratio_beta_0.is_finite() && ratio_beta_5.is_finite() && ratio_beta_10.is_finite(),
        "asymmetry ratios must be finite: β=0 → {ratio_beta_0}, β=5 → {ratio_beta_5}, β=10 → {ratio_beta_10}",
    );

    // Excess-over-linear: the β-dependent forward-steepening contribution
    // measured relative to the linear (β = 0) FDTD-dispersion baseline.
    let excess_beta_5 = ratio_beta_5 - ratio_beta_0;
    let excess_beta_10 = ratio_beta_10 - ratio_beta_0;

    assert!(
        excess_beta_5 > 0.0 && excess_beta_10 > 0.0,
        "Both β = 5 and β = 10 must produce excess forward steepening above the \
         linear (β = 0) FDTD-dispersion baseline. Got: \
         β=0 ratio = {ratio_beta_0:.4}, β=5 ratio = {ratio_beta_5:.4} (excess = {excess_beta_5:.4}), \
         β=10 ratio = {ratio_beta_10:.4} (excess = {excess_beta_10:.4}). \
         Non-positive excess at finite β suggests the β·∂²(p²)/∂t² term is not \
         entering the recurrence with the correct sign and magnitude.",
    );

    let scaling = excess_beta_10 / excess_beta_5;
    // Weak-nonlinear theory predicts scaling = 2.0; FDTD dispersion + finite-
    // domain diffraction at strong nonlinearity (p₀ = 5 MPa, near-shocked
    // regime) widen the acceptance band.
    assert!(
        (1.3..=3.0).contains(&scaling),
        "β-scaling regression: excess(β=10)/excess(β=5) = {scaling:.4}; expected in \
         [1.3, 3.0] with target 2.0 per leading-order weak-nonlinear theory \
         (Hamilton & Blackstock 1998 §4.3 — `|P_2| ∝ β·|P_1|²·z`). A scaling \
         near 4 suggests `β²` instead of `β` in the nonlinear term; a scaling \
         near 1 suggests β is not entering the recurrence at all. Raw ratios: \
         β=0 → {ratio_beta_0:.4}, β=5 → {ratio_beta_5:.4}, β=10 → {ratio_beta_10:.4}.",
    );
}

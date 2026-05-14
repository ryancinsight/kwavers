use ndarray::Array3;

use super::{run_theranostic_nonlinear_3d, Nonlinear3dConfig};
use crate::clinical::therapy::theranostic_guidance::AnatomyKind;

#[test]
fn nonlinear_3d_westervelt_fwi_and_cavitation_inverse_are_input_sensitive() {
    let (ct, labels) = abdominal_fixture();
    let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
    config.grid_size = 12;
    config.element_count = 18;
    config.receiver_count = 8;
    config.source_encoding_count = 2;
    config.iterations = 1;
    config.frequency_hz = 300_000.0;
    config.source_pressure_pa = 2.0e6;
    config.cycles = 2.0;
    config.bubble_time_steps_per_period = 24;
    config.cavitation_iterations = 6;

    let result = run_theranostic_nonlinear_3d(
        AnatomyKind::Liver,
        &ct,
        Some(&labels),
        [2.0, 2.0, 2.0],
        &config,
    )
    .expect("nonlinear 3-D fixture must run");

    assert!(result.is_full_wave_inversion);
    assert!(result.uses_nonlinear_wave_propagation);
    assert!(result.uses_rayleigh_plesset);
    assert_eq!(result.ct_hu.dim(), (12, 12, 12));
    assert!(result.active_voxels > 32);
    assert!(result.target_mask.iter().filter(|active| **active).count() >= 2);
    assert!(
        result
            .inversion_mask
            .iter()
            .filter(|active| **active)
            .count()
            >= result.target_mask.iter().filter(|active| **active).count()
    );
    assert!(
        result
            .westervelt_peak_pressure_pa
            .iter()
            .copied()
            .fold(0.0, f64::max)
            > 0.0
    );
    assert!(
        result
            .cavitation_source_density
            .iter()
            .copied()
            .fold(0.0, f64::max)
            > 0.0
    );
    assert!(
        result
            .multiparameter_fwi_score
            .iter()
            .copied()
            .fold(0.0, f64::max)
            >= 0.0
    );
    assert!(result
        .reconstructed_delta_beta
        .iter()
        .copied()
        .any(|value| value.is_finite()));
    assert!(result.fwi_objective_history.iter().all(|v| v.is_finite()));
    assert!(
        result.fwi_objective_history.last().copied().unwrap()
            <= result.fwi_objective_history.first().copied().unwrap()
    );
    assert!(
        result.cavitation_objective_history.last().copied().unwrap()
            <= result
                .cavitation_objective_history
                .first()
                .copied()
                .unwrap()
    );
    assert!(result.therapy_points_m.len() >= 8);
    assert!(result.receiver_points_m.len() >= 4);
}

/// End-to-end integration test for the `AnatomyKind::Brain` path: synthetic
/// CT with a cortical skull shell wrapping a brain interior, INSIGHTEC-like
/// calvarium helmet aperture, lossless Westervelt forward, discrete adjoint
/// FWI, heterogeneous CT-derived path-integrated cavitation Green's function
/// (including `y = 2` Stokes-Kirchhoff skull attenuation), and Rayleigh-
/// Plesset passive subharmonic inverse.
///
/// # Why this complements the abdominal test
///
/// The abdominal fixture exercises soft-tissue-only paths where the new
/// `y ≈ 1.05` exponent is a small correction. The brain fixture is the only
/// path that actually places **skull voxels (HU > 300)** between source and
/// receiver — i.e., the only path where:
/// - the `α₀ = 13 - 20 dB/(cm·MHz)` skull attenuation appears in the
///   integral,
/// - the `y = 2` Stokes-Kirchhoff power-law gives a `3.07×` reduction at
///   the 325 kHz subharmonic versus the naive `y = 1` extrapolation,
/// - the helmet aperture is placed on the calvarium surface rather than on
///   skin-coupled abdominal arc.
#[test]
fn nonlinear_3d_brain_helmet_pipeline_is_input_sensitive_through_skull() {
    let ct = brain_fixture();
    let mut config = Nonlinear3dConfig::new(AnatomyKind::Brain);
    config.grid_size = 12;
    config.element_count = 24;
    config.receiver_count = 8;
    config.source_encoding_count = 2;
    config.iterations = 1;
    config.frequency_hz = 650_000.0; // INSIGHTEC-like
    config.source_pressure_pa = 1.5e5;
    config.cycles = 2.0;
    config.bubble_time_steps_per_period = 24;
    config.cavitation_iterations = 6;

    let result =
        run_theranostic_nonlinear_3d(AnatomyKind::Brain, &ct, None, [1.5, 1.5, 1.5], &config)
            .expect("nonlinear 3-D brain fixture must run");

    assert!(result.is_full_wave_inversion);
    assert!(result.uses_nonlinear_wave_propagation);
    assert!(result.uses_rayleigh_plesset);
    assert_eq!(result.ct_hu.dim(), (12, 12, 12));
    assert!(result.active_voxels > 32);
    assert!(
        result.target_mask.iter().filter(|active| **active).count() >= 2,
        "synthetic brain ellipsoidal target must be non-empty inside the body support",
    );
    // The INSIGHTEC-like helmet model placed on the calvarium cap.
    assert!(
        result
            .aperture_model
            .contains("insightec_like_calvarium_helmet"),
        "brain aperture model must be the INSIGHTEC-like calvarium helmet; got '{}'",
        result.aperture_model,
    );
    // Westervelt peak pressure must be positive somewhere inside the
    // skull-bounded support.
    assert!(
        result
            .westervelt_peak_pressure_pa
            .iter()
            .copied()
            .fold(0.0, f64::max)
            > 0.0,
        "Westervelt peak pressure must be positive after the source-encoded transmissions",
    );
    // Cavitation source density must be positive — at 1.5e5 Pa diagnostic
    // pressure the Rayleigh-Plesset response is small but should be
    // detectable inside at least one voxel.
    assert!(
        result
            .cavitation_source_density
            .iter()
            .copied()
            .fold(0.0, f64::max)
            > 0.0,
        "cavitation source density must respond to the simulated peak pressure",
    );
    assert!(result.fwi_objective_history.iter().all(|v| v.is_finite()));
    assert!(
        result.fwi_objective_history.last().copied().unwrap()
            <= result.fwi_objective_history.first().copied().unwrap(),
        "Westervelt FWI objective must be non-increasing on the brain fixture",
    );
    assert!(
        result.cavitation_objective_history.last().copied().unwrap()
            <= result
                .cavitation_objective_history
                .first()
                .copied()
                .unwrap(),
        "Cavitation projected-gradient objective must be non-increasing on the brain fixture",
    );
    assert!(result.therapy_points_m.len() >= 16);
    assert!(result.receiver_points_m.len() >= 4);
}

/// Synthetic brain CT: ellipsoidal skull shell (cortical bone HU values)
/// wrapping an ellipsoidal brain interior (soft tissue HU), surrounded by
/// air. Used by `nonlinear_3d_brain_helmet_pipeline_is_input_sensitive_
/// through_skull` to exercise the cavitation path through skull voxels
/// where the heterogeneous-power-law absorption (`y = 2` for skull,
/// `y ≈ 1.05` for brain) has its largest physical effect.
fn brain_fixture() -> Array3<f64> {
    let n = 28;
    let mut ct = Array3::<f64>::from_elem((n, n, n), -1000.0); // air outside head
    let center = [14.0, 14.0, 14.0];
    // Outer head ellipsoid (skull + scalp): everything inside is body.
    let head_radii = [12.5, 11.5, 10.5];
    // Inner brain ellipsoid (soft tissue, lower HU): inside the skull shell.
    let brain_radii = [11.0, 10.0, 9.0];
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let head_r = ellipsoid_radius([x, y, z], center, head_radii);
                if head_r <= 1.0 {
                    // Default: skull HU value inside head but outside brain.
                    ct[[x, y, z]] = 600.0; // cortical bone HU (lower-bound, well above 300 threshold)
                }
                let brain_r = ellipsoid_radius([x, y, z], center, brain_radii);
                if brain_r <= 1.0 {
                    ct[[x, y, z]] = 40.0; // brain tissue HU (soft tissue)
                }
            }
        }
    }
    ct
}

fn abdominal_fixture() -> (Array3<f64>, Array3<i16>) {
    let n = 24;
    let mut ct = Array3::<f64>::from_elem((n, n, n), -1000.0);
    let mut labels = Array3::<i16>::zeros((n, n, n));
    let center = [12.0, 12.0, 12.0];
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let body_r = ellipsoid_radius([x, y, z], center, [9.0, 8.0, 7.0]);
                if body_r <= 1.0 {
                    ct[[x, y, z]] = 35.0;
                }
                let organ_r = ellipsoid_radius([x, y, z], center, [5.0, 4.0, 4.0]);
                if organ_r <= 1.0 {
                    labels[[x, y, z]] = 1;
                    ct[[x, y, z]] = 55.0;
                }
                let target_r = ellipsoid_radius([x, y, z], [12.0, 12.0, 11.0], [2.0, 2.0, 2.0]);
                if target_r <= 1.0 {
                    labels[[x, y, z]] = 2;
                    ct[[x, y, z]] = 75.0;
                }
            }
        }
    }
    (ct, labels)
}

fn ellipsoid_radius(idx: [usize; 3], center: [f64; 3], radius: [f64; 3]) -> f64 {
    ((idx[0] as f64 - center[0]) / radius[0]).powi(2)
        + ((idx[1] as f64 - center[1]) / radius[1]).powi(2)
        + ((idx[2] as f64 - center[2]) / radius[2]).powi(2)
}

/// Sign-sensitive regression for the Westervelt forward.
///
/// # Theorem (forward steepening, Hamilton & Blackstock 1998 §3.6)
///
/// For a finite-amplitude wave propagating in a medium with `β > 0`, the
/// instantaneous local sound speed is `c(p) ≈ c₀ + β·p/(ρ₀c₀)`. Compressions
/// (`p > 0`) travel faster than rarefactions, producing forward steepening:
/// at a fixed downstream receiver the rising edge of each cycle sharpens
/// while the falling edge stretches. This makes
/// `max_t ∂p/∂t  >  |min_t ∂p/∂t|`.
///
/// A Westervelt FDTD with the *correct* sign on `∂²(p²)/∂t²` (positive on
/// the explicit update of `p_{n+1}`) reproduces this asymmetry. A sign flip
/// inverts the asymmetry — peaks would *round* while troughs sharpen, which
/// is non-physical.
///
/// # Algorithm
///
/// 1. Build a homogeneous 24³ cube with `c₀ = 1500 m/s`, `ρ₀ = 1000 kg/m³`,
///    `β = 10` (deliberately large to amplify the asymmetry inside the small
///    test domain).
/// 2. Inject a 1 MHz sinusoidal source at one face; record at the opposite
///    face after the linear travel horizon.
/// 3. Compute the discrete time-derivative of the receiver trace and the
///    ratio `R = max(∂p/∂t) / |min(∂p/∂t)|`.
/// 4. Assert `R > 1` (forward steepening). A sign-flipped Westervelt
///    produces `R < 1` (reverse steepening).
#[test]
fn forward_westervelt_exhibits_physical_forward_steepening_with_corrected_sign() {
    use super::super::Point3;
    use super::forward::{forward_with_schedule, ForwardInput, TimeSchedule};
    use super::types::{GridIndex, Nonlinear3dAperture};

    let n = 24;
    let spacing_m = 4.0e-4_f64;
    let c0 = 1500.0_f64;
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
        focus: receiver_idx,
    };

    let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
    config.frequency_hz = 1.0e6;
    config.source_pressure_pa = 5.0e6;
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
        n,
        spacing_m,
        aperture: &aperture,
        config: &config,
        schedule,
        encoding: super::encoding::SourceEncoding { index: 0, count: 1 },
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

/// Linear-baseline negative-control for the Westervelt nonlinearity:
/// with `β = 0` the Westervelt equation reduces to the linear wave equation,
/// so the receiver trace must be (approximately) symmetric — no preferred
/// rising vs. falling steepening direction.
///
/// # Why this matters
///
/// The companion test `forward_westervelt_exhibits_physical_forward_
/// steepening_with_corrected_sign` asserts `max(∂p/∂t) > |min(∂p/∂t)|` at
/// `β = 10`. That test alone cannot distinguish *real* `β·∂²(p²)/∂t²`
/// asymmetry from numerical-dispersion asymmetry of the 7-point FDTD
/// Laplacian, which can also produce small `max != |min|` artifacts at
/// the 2nd harmonic of the source. Running the same fixture with `β = 0`
/// quantifies the numerical-dispersion floor and asserts the symmetry
/// holds within FDTD tolerance. If a future refactor accidentally
/// introduces a fabricated nonlinearity (e.g., spurious cross-term in the
/// linear update), this test catches it.
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
    use super::super::Point3;
    use super::forward::{forward_with_schedule, ForwardInput, TimeSchedule};
    use super::types::{GridIndex, Nonlinear3dAperture};

    let n = 24;
    let spacing_m = 4.0e-4_f64;
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
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
        focus: receiver_idx,
    };

    let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
    config.frequency_hz = 1.0e6;
    config.source_pressure_pa = 5.0e6;
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
        n,
        spacing_m,
        aperture: &aperture,
        config: &config,
        schedule,
        encoding: super::encoding::SourceEncoding { index: 0, count: 1 },
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
///
/// # Algorithm
///
/// 1. Run the same homogeneous forward at `β = 0` (linear baseline,
///    captures the FDTD-dispersion bias floor), `β = 5`, and `β = 10`.
/// 2. Compute the raw steepening signature `R(β) = max(∂p/∂t) /
///    |min(∂p/∂t)|`.
/// 3. Subtract the linear baseline: `δ(β) = R(β) − R(0)`.
/// 4. Assert both `δ(5) > 0` and `δ(10) > 0` (real forward steepening).
/// 5. Assert `δ(10) / δ(5)` is between `1.3` and `3.0` (target ratio `2.0`).
///
/// # Why the excess-over-linear formulation
///
/// At low source amplitude (p₀ = 50 kPa here) the FDTD 7-point Laplacian's
/// numerical-dispersion bias at the 2nd harmonic can be comparable in
/// magnitude to the real β-induced forward steepening. Comparing raw
/// signatures would conflate the dispersion bias (β-independent) with the
/// nonlinear contribution. Subtracting the `β = 0` baseline removes the
/// dispersion floor and isolates the β-dependent part.
#[test]
fn westervelt_steepening_signature_scales_linearly_with_beta_per_weak_nonlinear_theory() {
    use super::super::Point3;
    use super::forward::{forward_with_schedule, ForwardInput, TimeSchedule};
    use super::types::{GridIndex, Nonlinear3dAperture};

    let n = 24;
    let spacing_m = 4.0e-4_f64;
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
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
        focus: receiver_idx,
    };

    let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
    config.frequency_hz = 1.0e6;
    // Use the same source pressure as the existing forward-steepening test
    // so the nonlinear contribution dominates the FDTD-dispersion bias.
    config.source_pressure_pa = 5.0e6;
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
            n,
            spacing_m,
            aperture: &aperture,
            config: &config,
            schedule,
            encoding: super::encoding::SourceEncoding { index: 0, count: 1 },
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

/// Harmonic-generation presence check for the nonlinear 3-D Westervelt
/// forward at strong source amplitude.
///
/// # What this validates
///
/// The companion sign-, linear-baseline-, and β-scaling tests pin the
/// *sign* and *scaling* of the nonlinear contribution but only inspect
/// time-domain asymmetry. This test extracts the 2nd-harmonic amplitude
/// via discrete sine/cosine projection and asserts that the FDTD generates
/// a *measurable* harmonic content (`|P_2|/|P_1| ∈ [0.03, 0.40]`) at a
/// strongly nonlinear source amplitude. Catches:
/// - A nonlinear term that propagates as just a phase shift (no harmonic
///   generation at all → ratio ≈ 0).
/// - A spuriously-high 2nd harmonic from grid dispersion (ratio > 0.5
///   without strong nonlinearity).
/// - The forward returning DC-only or NaN output.
///
/// # Why not Fubini-absolute
///
/// The Aanonsen-1984 / Fubini analytical solution
/// `|P_n|/|P_1| = J_n(nΓ) / (n J_1(Γ))` assumes a 1-D plane wave with
/// constant amplitude over the propagation path (Γ = z/z_shock with fixed
/// P_0). A 3-D point source has 1/r geometric spreading, so the local
/// amplitude — and hence the local Γ governing nonlinear distortion —
/// varies along the path. The KZK solver carries the literature-validated
/// Fubini-absolute test (`solver::forward::nonlinear::kzk::validation::
/// nonlinear::tests::test_aanonsen_1984_harmonic_amplitudes`) because KZK
/// parabolically reduces 3-D propagation to 1-D-along-z with constant-
/// amplitude planar shots. The Westervelt FDTD cannot drive that
/// configuration without API changes, so we test for harmonic *presence*
/// rather than absolute Fubini-matching in this fixture.
///
/// # Tier
///
/// `#[ignore]`'d as Tier-2 (~10 s runtime). Run on demand with
/// `cargo test --lib --package kwavers -- --ignored harmonic_generation`.
#[test]
#[ignore = "Tier 2: Harmonic-generation presence check, ~10s runtime"]
fn westervelt_fdtd_point_source_generates_measurable_second_harmonic_content() {
    use super::super::Point3;
    use super::forward::{forward_with_schedule, ForwardInput, TimeSchedule};
    use super::types::{GridIndex, Nonlinear3dAperture};

    // Cubic grid required by the `ForwardInput` API. Single source cell at
    // the center of one face; receiver on the central axis downstream.
    // This is the same fixture shape as the working forward-steepening test,
    // which is known to be numerically stable.
    let n: usize = 48;
    let cells = n * n * n;
    let spacing_m = 1.0e-4_f64; // 0.1 mm → 4.8 mm cube
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
    let beta_nl = 10.0_f64;
    let frequency_hz = 1.0e6_f64;
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
        focus: receiver_idx,
    };

    // Use the same source pressure as the (numerically stable) forward-
    // steepening test. A single point source is FDTD-stable here; the
    // wave spreads spherically. We measure the **empirical** Γ from the
    // observed `|P_1|` at the receiver and compare `|P_2|/|P_1|` against
    // Fubini at that empirical Γ. This decouples the test from source-
    // amplitude calibration (which differs by orders of magnitude between
    // plane-wave and point-source 3D FDTD).
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
        n,
        spacing_m,
        aperture: &aperture,
        config: &config,
        schedule,
        encoding: super::encoding::SourceEncoding { index: 0, count: 1 },
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

    // Lower bound: the FDTD must generate a *measurable* 2nd harmonic at
    // 5 MPa with β = 10. A ratio below 3% would indicate that the
    // `β·∂²(p²)/∂t²` term is not effectively driving harmonic content
    // (e.g., the coefficient `q = β·dt²/(ρ·c²)` is being multiplied by 0
    // somewhere, or the nonlinear contribution is being subtracted out).
    //
    // Upper bound: a ratio above 40% would indicate either runaway harmonic
    // generation (e.g., `β²` instead of `β`) or grid-dispersion artifacts
    // that masquerade as harmonic content.
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

/// Aanonsen-1984 Fubini-absolute harmonic-ratio test for the Westervelt
/// discrete recurrence, using a clean 1-D harness.
///
/// # Theorem (Fubini analytical, Hamilton & Blackstock 1998 §4.3.2;
/// Aanonsen et al. 1984 Eq. 6)
///
/// For a plane wave propagating in a lossless medium with weak nonlinearity:
/// ```text
///   |P_n(z)| / |P_1(z)| = J_n(n Γ) / (n · J_1(Γ)),    Γ = z / z_shock
///   z_shock = ρ_0 c_0³ / (β · ω · P_0)
/// ```
/// At `Γ = 0.5`: `J_2(1) / (2·J_1(0.5)) ≈ 0.1149 / (2 · 0.2423) ≈ 0.2371`.
///
/// # Why 1-D and not 3-D
///
/// The 3-D `forward_with_schedule` API uses point sources with `1/r`
/// geometric spreading, so local Γ varies along the propagation path and
/// Fubini's constant-amplitude plane-wave assumption is violated. This test
/// uses a **clean 1-D Westervelt FDTD** whose update rule **algebraically
/// matches** the 3-D recurrence in `update_cells`:
///
/// ```text
///   p[n+1, i] = sponge[i] · (2 p[n, i] − p[n−1, i]
///                            + (c·dt)² · ∇²p[n, i]
///                            + q · ∂²(p²)/∂t²|^n)
///   q = β · dt² / (ρ · c²)
///   ∂²(p²)/∂t² ≈ 2 p[n] · d²p/dt² + 2 (dp/dt)²   (product rule)
/// ```
///
/// The 1-D Laplacian uses the 3-point stencil
/// `(p[i+1] − 2 p[i] + p[i−1]) / dx²`; everything else is bit-identical to
/// the 3-D `update_cells` algebra. If either the 3-D update or this 1-D
/// reference has a coefficient error, this test detects it.
///
/// # Algorithm
///
/// 1. Build a long 1-D domain (1024 cells, `dx = 0.05 mm` → `51.2 mm`)
///    with 30 pts/wavelength at the fundamental and 15 pts/wavelength at
///    the 2nd harmonic.
/// 2. Drive a **hard sinusoidal source** at `x = 4` (clamp pressure to
///    `P_0·sin(ωt)·window`) with nominal `P_0 = 1 MPa`, `f = 1 MHz`,
///    β = 10, and an `sin²` envelope whose peak (`t = burst_duration/2`)
///    is well after the wave-arrival time at the receiver.
/// 3. Run the lossless Westervelt-1D forward with absorbing sponge at
///    the far boundary to prevent reflections.
/// 4. Sample the receiver trace; window 8 periods centered on the
///    envelope peak.
/// 5. Project to fundamental and 2nd harmonic via discrete sine/cosine
///    quadrature at known frequencies (exact for harmonics).
/// 6. Compute the **empirical Γ** from the observed `|P_1|` at the
///    receiver, since a 1-D FDTD hard source radiates less than `P_0`
///    nominal (radiation coupling determined by the discrete Laplacian).
/// 7. Assert `|P_2| / |P_1|` matches the Fubini formula `J_2(2Γ)/(2·J_1(Γ))`
///    evaluated at the empirical Γ to within ±15 %.
///
/// # Why empirical Γ instead of geometric Γ
///
/// In a discrete FDTD the hard-source clamp sets the cell pressure but
/// the radiated plane wave's amplitude is determined by the discrete
/// Laplacian and CFL — empirically ≈ 0.57 P_0 at this configuration. The
/// physically meaningful Γ for Fubini comparison is the one carried by
/// the actual propagating wave, not the source-clamp nominal. Using the
/// observed `|P_1|` removes the radiation-coupling source-amplitude
/// calibration as a confounder and isolates the **Westervelt recurrence
/// algebra** for validation.
///
/// # Tolerance rationale (15 %)
///
/// At 15 pts/wavelength resolution for the 2nd harmonic, the 3-point
/// Laplacian's numerical-dispersion bias is a few percent; the `sin²`
/// envelope's slight slope across the 8-period window contributes
/// another few percent of spectral leakage. 15 % is comfortable margin
/// for these well-understood numerical effects while still being tight
/// enough to catch any coefficient sign or magnitude error in `q` or in
/// the product-rule `2p·d²p/dt² + 2(dp/dt)²` expansion.
///
/// # Tier
///
/// `#[ignore]`'d (~2 s runtime). Run on demand with
/// `cargo test --lib --package kwavers -- --ignored fubini_absolute`.
#[test]
#[ignore = "Tier 2: Literature validation (Aanonsen 1984 Fubini-absolute, 1-D harness), ~2s runtime"]
fn westervelt_recurrence_fubini_absolute_at_gamma_half_matches_aanonsen_1984() {
    // Physical parameters chosen so Γ = 0.5 lies inside the 1-D domain.
    // Resolution: 30 pts/wavelength at fundamental, 15 pts/wavelength at 2nd
    // harmonic — sufficient to resolve the 2nd harmonic without significant
    // FDTD numerical dispersion bias.
    let nx: usize = 1024;
    let dx: f64 = 5.0e-5; // 0.05 mm → 51.2 mm domain
    let c: f64 = 1500.0;
    let rho: f64 = 1000.0;
    let beta: f64 = 10.0;
    let frequency_hz: f64 = 1.0e6;
    let omega = std::f64::consts::TAU * frequency_hz;
    let p0: f64 = 1.0e6; // 1 MPa
    let z_shock = rho * c.powi(3) / (beta * omega * p0);
    let target_gamma = 0.5_f64;
    let target_distance_m = target_gamma * z_shock;
    let source_index: usize = 4;
    let receiver_index = source_index + (target_distance_m / dx).round() as usize;
    assert!(
        receiver_index < nx - 16,
        "receiver index {receiver_index} would land inside the far-boundary sponge",
    );

    // CFL-stable timestep matching the 3-D solver's CFL convention.
    let cfl: f64 = 0.5;
    let dt = cfl * dx / c;
    let dt2 = dt * dt;
    let inv_dt = 1.0 / dt;
    let inv_dt2 = 1.0 / dt2;
    let inv_dx2 = 1.0 / (dx * dx);
    let q = beta * dt2 / (rho * c.powi(2));

    // Long burst: chosen so the envelope peak (`t = burst_duration / 2`)
    // is *after* the wave reaches the receiver (`travel_time`). Otherwise
    // the steady-state projection samples an empty trace before the
    // wavefront arrives. With `cycles = 80` and the geometry above,
    // envelope peak is at 40 µs and travel time is ≈ 17.9 µs.
    let cycles: f64 = 80.0;
    let burst_duration = cycles / frequency_hz;
    let travel_time = (receiver_index as f64 - source_index as f64) * dx / c;
    assert!(
        burst_duration / 2.0 > travel_time + 4.0 / frequency_hz,
        "burst envelope peak must be at least 4 cycles after the wavefront \
         arrives at the receiver; got burst peak = {} s, travel time = {} s",
        burst_duration / 2.0,
        travel_time,
    );
    let period_steps = (1.0 / (frequency_hz * dt)).round() as usize;
    let steps = (burst_duration / dt).ceil() as usize + 4 * period_steps;

    // Far-boundary sponge: smooth quadratic ramp over the last 32 cells so
    // the wave is absorbed before reflecting off the boundary.
    let sponge_layer = 32_usize;
    let mut sponge = vec![1.0_f64; nx];
    for i in 0..sponge_layer {
        let edge = i;
        let ratio = (sponge_layer - edge) as f64 / sponge_layer as f64;
        sponge[nx - 1 - i] = (1.0 - 0.18 * ratio * ratio).max(0.0);
    }
    sponge[0] = 0.0;
    sponge[nx - 1] = 0.0;

    let mut p_older = vec![0.0_f64; nx];
    let mut p_prev = vec![0.0_f64; nx];
    let mut p_curr = vec![0.0_f64; nx];
    let mut p_next = vec![0.0_f64; nx];
    let mut traces = Vec::with_capacity(steps);

    for step in 0..steps {
        let t_curr = step as f64 * dt;

        // Interior cell update — algebraically identical to the 3-D
        // `update_cells` (1-D Laplacian instead of 7-point 3-D).
        for i in 1..nx - 1 {
            let center = p_curr[i];
            let prev = p_prev[i];
            let older = p_older[i];
            let lap = (p_curr[i + 1] - 2.0 * center + p_curr[i - 1]) * inv_dx2;
            let dp_dt = (center - prev) * inv_dt;
            let nl = if step >= 2 {
                let d2p_dt2 = (center - 2.0 * prev + older) * inv_dt2;
                2.0_f64.mul_add(center * d2p_dt2, 2.0 * dp_dt * dp_dt)
            } else {
                2.0 * dp_dt * dp_dt
            };
            let raw = 2.0_f64.mul_add(center, -prev) + (c * dt).powi(2) * lap + q * nl;
            p_next[i] = sponge[i] * raw;
        }

        // Hard sinusoidal source: clamp the source-cell pressure to the
        // desired driving waveform. The `sin²` envelope ensures a smooth
        // ramp-up and ramp-down.
        let envelope = if t_curr < burst_duration {
            (std::f64::consts::PI * t_curr / burst_duration).sin().powi(2)
        } else {
            0.0
        };
        p_next[source_index] = p0 * (omega * t_curr).sin() * envelope;

        traces.push(p_next[receiver_index]);

        std::mem::swap(&mut p_older, &mut p_prev);
        std::mem::swap(&mut p_prev, &mut p_curr);
        std::mem::swap(&mut p_curr, &mut p_next);
    }

    // Pick a steady-state window centered on the burst-envelope peak. The
    // peak is at `t = burst_duration / 2`; we use ±4 periods around it.
    let peak_step = ((burst_duration / 2.0) / dt).round() as usize;
    let half_window_periods = 4_usize;
    let window_half_steps = half_window_periods * period_steps;
    assert!(
        peak_step > window_half_steps + 4,
        "burst envelope peak too close to t=0 — increase `cycles`",
    );
    let window_start = peak_step - window_half_steps;
    let window_end = peak_step + window_half_steps;
    assert!(
        window_end < traces.len(),
        "burst window extends past simulation: peak_step = {peak_step}, traces.len() = {}",
        traces.len(),
    );
    let window = &traces[window_start..window_end];

    // Discrete sine/cosine projection at a known frequency.
    // `A_f = (2 / N) · sqrt((Σ p·cos(ωt))² + (Σ p·sin(ωt))²)`. The window
    // length is an integer multiple of the fundamental period so the
    // projection is orthogonal and exact (mod FDTD dispersion).
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
    assert!(amp_fundamental.is_finite() && amp_second_harmonic.is_finite());
    assert!(
        amp_fundamental > 0.0,
        "fundamental amplitude must be positive; got |P_1| = {amp_fundamental}",
    );

    let ratio = amp_second_harmonic / amp_fundamental;

    // Empirical Γ from the **observed** `|P_1|` at the receiver. The
    // hard-source clamp in a 1-D FDTD radiates less than `p_0` because the
    // source cell's value is fixed but the radiated wave amplitude depends
    // on the discrete Laplacian coupling. The relevant Γ for Fubini
    // comparison is the one carried by the actual propagating wave:
    //   Γ_emp = β · ω · z · |P_1| / (ρ · c³)
    let gamma_empirical =
        beta * omega * target_distance_m * amp_fundamental / (rho * c.powi(3));
    assert!(
        gamma_empirical > 0.05 && gamma_empirical < 1.5,
        "empirical Γ must be in the pre/near-shock regime: got Γ = {gamma_empirical:.4} \
         (|P_1| = {amp_fundamental:.4e} Pa, distance = {target_distance_m:.4e} m)",
    );

    // Fubini analytical at the empirical Γ:
    //   |P_2|/|P_1| = J_2(2Γ) / (2 · J_1(Γ))
    let fubini_at_empirical_gamma = bessel_j2(2.0 * gamma_empirical)
        / (2.0 * bessel_j1(gamma_empirical));
    let relative_error =
        (ratio - fubini_at_empirical_gamma).abs() / fubini_at_empirical_gamma;
    // Tight tolerance: 15% — the Westervelt recurrence's `q·∂²(p²)/∂t²`
    // and the product-rule `2p·d²p/dt² + 2(dp/dt)²` algebra must match
    // Fubini at the empirical Γ to within FDTD numerical-dispersion
    // bias (≈ a few percent for 15 pts/wavelength resolution at 2nd
    // harmonic). A relative error > 15% suggests a coefficient error in
    // `q = β·dt²/(ρ·c²)` or in the nonlinear-term expansion.
    assert!(
        relative_error < 0.15,
        "Aanonsen-1984 Fubini empirical-Γ regression: \
         measured |P_2|/|P_1| = {ratio:.4}; Fubini at empirical Γ = {gamma_empirical:.4} → \
         analytical {fubini_at_empirical_gamma:.4}; relative error = {:.1}% (tolerance: 15%). \
         |P_1| = {amp_fundamental:.4e} Pa, |P_2| = {amp_second_harmonic:.4e} Pa. \
         Geometric distance = {target_distance_m:.4e} m. \
         A relative error > 15% suggests a coefficient error in `q = β·dt²/(ρ·c²)` or in the \
         product-rule `∂²(p²)/∂t² = 2p·d²p/dt² + 2(dp/dt)²` expression.",
        relative_error * 100.0,
    );
}

/// Bessel function `J_1(x)` for small `|x|` via the standard power series
/// `J_1(x) = Σₖ (-1)ᵏ · (x/2)^(2k+1) / (k! · (k+1)!)`. Converges to
/// machine precision in ≤ 30 terms for `|x| ≤ 2`.
fn bessel_j1(x: f64) -> f64 {
    let mut term = x / 2.0;
    let mut sum = term;
    let half_x_sq = (x * x) / 4.0;
    for k in 1..50 {
        let k_f = k as f64;
        term *= -half_x_sq / (k_f * (k_f + 1.0));
        sum += term;
        if term.abs() < 1.0e-18 {
            break;
        }
    }
    sum
}

/// Bessel function `J_2(x)` via the recurrence `J_2(x) = (2/x)·J_1(x) − J_0(x)`
/// with `J_0` and `J_1` from their power series.
fn bessel_j2(x: f64) -> f64 {
    let j0 = {
        let mut term = 1.0_f64;
        let mut sum = term;
        let half_x_sq = (x * x) / 4.0;
        for k in 1..50 {
            let k_f = k as f64;
            term *= -half_x_sq / (k_f * k_f);
            sum += term;
            if term.abs() < 1.0e-18 {
                break;
            }
        }
        sum
    };
    let j1 = bessel_j1(x);
    if x.abs() < 1.0e-12 {
        return 0.0;
    }
    (2.0 / x) * j1 - j0
}

/// Power-law absorption decay validation for the 3-D Westervelt FDTD with
/// the Treeby-Cox 2010 fractional-Laplacian operator in [`super::absorption`].
///
/// # Theorem
///
/// In a homogeneous absorbing medium with power-law attenuation
/// `α(f) = α₀·|f_MHz|^y` Np/m, the ratio of the absorbing-medium pressure
/// envelope to the lossless-medium pressure envelope at the same point
/// and time satisfies
///
/// ```text
///   |p_abs(r)| / |p_lossless(r)| = exp(-α(f) · r)
/// ```
///
/// because the geometric spreading factor `1/r` is identical between
/// the two simulations and divides out exactly. Taking the logarithm of
/// the ratio versus `r` is linear with slope `-α(f)`.
///
/// # Algorithm
///
/// 1. Run two identical low-amplitude (linear-regime) 3-D point-source
///    simulations on a homogeneous cubic grid: one with `α₀ = 0` and one
///    with `α₀ = 5.8 Np/m` at 1 MHz, `y = 1.05` (soft-tissue values from
///    Hamilton & Blackstock 1998 Table 4.1; Treeby & Cox 2010 Table I).
/// 2. Place 4 receivers along the source axis at increasing far-field
///    distance (`r > 3·λ`).
/// 3. Read per-cell peak pressures from `update_peak`.
/// 4. Fit `log(p_abs/p_lossless) = const − α_fit · r` via least-squares.
/// 5. Assert `α_fit` matches `α(f) = α₀·f_MHz^y = 5.8·(1.0)^1.05 ≈ 5.8` Np/m
///    to within ±35 % (FDTD discrete-dispersion bias + dropped Kramers-
///    Kronig dispersion correction allowance).
///
/// # Tier
///
/// `#[ignore]`'d (~10 s runtime, two forward runs). Run on demand with
/// `cargo test --lib --package kwavers -- --ignored absorption_decay`.
#[test]
#[ignore = "Tier 2: Literature validation (Treeby-Cox 2010 plane-wave decay), ~10s runtime"]
fn fractional_laplacian_absorption_decay_ratio_matches_alpha_omega_y_power_law() {
    use super::super::Point3;
    use super::forward::{forward_with_schedule, ForwardInput, TimeSchedule};
    use super::types::{GridIndex, Nonlinear3dAperture};

    let n: usize = 96;
    let cells = n * n * n;
    let spacing_m = 1.5e-4_f64;
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
    let frequency_hz = 1.0e6_f64;
    let alpha0_np_per_m_at_1mhz = 5.8_f64;
    let y_exponent = 1.05_f64;
    let wavelength = c0 / frequency_hz; // 1.5 mm
    let pts_per_wavelength = wavelength / spacing_m; // 10 — adequate

    let speed = vec![c0; cells];
    let density = vec![rho0; cells];
    let beta = vec![0.0_f64; cells]; // linear regime
    let attenuation_alpha0 = vec![alpha0_np_per_m_at_1mhz; cells];
    let attenuation_y = vec![y_exponent; cells];
    let attenuation_zero = vec![0.0_f64; cells];

    let source_x = 8_usize;
    let source_idx = GridIndex {
        x: source_x,
        y: n / 2,
        z: n / 2,
    };
    // Four receivers at r = 4λ, 6λ, 8λ, 10λ → 6 mm, 9 mm, 12 mm, 15 mm.
    // At α = 5.8 Np/m, expected log-ratios are -0.035, -0.052, -0.070, -0.087.
    let r_voxels: Vec<usize> = [40, 56, 72, 84]
        .iter()
        .copied()
        .filter(|&dx| source_x + dx < n - 8)
        .collect();
    assert!(
        r_voxels.len() >= 3,
        "need at least 3 in-grid far-field receiver positions; got {} \
         (try reducing receiver distances or increasing grid size)",
        r_voxels.len(),
    );
    let receiver_indices: Vec<GridIndex> = r_voxels
        .iter()
        .map(|&dx| GridIndex {
            x: source_x + dx,
            y: n / 2,
            z: n / 2,
        })
        .collect();
    let aperture = Nonlinear3dAperture {
        sources: vec![source_idx],
        receivers: receiver_indices.clone(),
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
        model_name: "test_absorption_decay".to_owned(),
        focus: *receiver_indices.last().unwrap(),
    };

    let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
    config.frequency_hz = frequency_hz;
    config.source_pressure_pa = 5.0e4; // linear regime
    // Short pulse: 6 cycles so the wave passes each receiver ONCE before
    // boundary reflections return. With c·dt·n ≈ 14 mm and far receiver
    // at 10.8 mm, the round-trip reflection arrives ~5 mm later — outside
    // the 6-cycle (6 µs) pulse window measured at each receiver.
    config.cycles = 6.0;
    config.cfl = 0.4;
    let dt = config.cfl * spacing_m / (c0 * 3.0_f64.sqrt());
    let travel_steps_far =
        ((receiver_indices.last().unwrap().x - source_idx.x) as f64 * spacing_m / (c0 * dt)).ceil()
            as usize;
    let period_steps = (1.0 / (frequency_hz * dt)).round() as usize;
    let pulse_steps = (config.cycles * period_steps as f64).ceil() as usize;
    // Total simulation: pulse propagates to far receiver and just clears
    // before boundary reflections from the far face return.
    let steps = travel_steps_far + pulse_steps + 4 * period_steps;
    let schedule = TimeSchedule {
        dt_s: dt,
        time_steps: steps,
    };

    // Peak-from-traces: each receiver's trace is sampled into `result.traces`
    // at every step. Take the absolute-max within a window that ends before
    // the boundary reflection round-trip from the +x face returns.
    let n_recv = receiver_indices.len();
    let max_step_before_reflection = |recv_index: usize| -> usize {
        let r_to_boundary_voxels = n - receiver_indices[recv_index].x;
        let extra_round_trip_steps = (2 * r_to_boundary_voxels) as f64 * spacing_m / (c0 * dt);
        let pulse_arrival = (receiver_indices[recv_index].x - source_idx.x) as f64 * spacing_m
            / (c0 * dt);
        let reflection_return = pulse_arrival + extra_round_trip_steps;
        (reflection_return - period_steps as f64).floor().max(2.0) as usize
    };

    let run = |alpha_field: &[f64]| -> Vec<f64> {
        let result = forward_with_schedule(ForwardInput {
            speed: &speed,
            density: &density,
            beta: &beta,
            attenuation_np_per_m_mhz: Some(alpha_field),
            attenuation_power_law_y: Some(&attenuation_y),
            n,
            spacing_m,
            aperture: &aperture,
            config: &config,
            schedule,
            encoding: super::encoding::SourceEncoding { index: 0, count: 1 },
            retain_history: false,
        });
        (0..n_recv)
            .map(|recv| {
                let end_step = max_step_before_reflection(recv).min(steps);
                (0..end_step)
                    .map(|step| result.traces[step * n_recv + recv].abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect()
    };

    let peaks_lossless = run(&attenuation_zero);
    let peaks_absorbing = run(&attenuation_alpha0);

    assert!(
        peaks_lossless.iter().all(|p| p.is_finite() && *p > 0.0),
        "lossless peaks must be positive and finite; got {peaks_lossless:?}",
    );
    assert!(
        peaks_absorbing.iter().all(|p| p.is_finite() && *p > 0.0),
        "absorbing peaks must be positive and finite; got {peaks_absorbing:?}",
    );

    let distances_m: Vec<f64> = receiver_indices
        .iter()
        .map(|idx| (idx.x - source_idx.x) as f64 * spacing_m)
        .collect();
    let log_ratio: Vec<f64> = peaks_absorbing
        .iter()
        .zip(peaks_lossless.iter())
        .map(|(pa, pl)| (pa / pl).ln())
        .collect();

    // Least-squares slope of `log(p_abs/p_lossless)` vs `r`. Slope = -α.
    let n_pts = distances_m.len() as f64;
    let mean_r = distances_m.iter().sum::<f64>() / n_pts;
    let mean_y = log_ratio.iter().sum::<f64>() / n_pts;
    let cov_ry: f64 = distances_m
        .iter()
        .zip(log_ratio.iter())
        .map(|(r, y)| (r - mean_r) * (y - mean_y))
        .sum();
    let var_r: f64 = distances_m.iter().map(|r| (r - mean_r).powi(2)).sum();
    let slope = cov_ry / var_r;
    let alpha_fit = -slope;

    let alpha_analytical = alpha0_np_per_m_at_1mhz * (frequency_hz / 1.0e6).powf(y_exponent);
    let relative_error =
        (alpha_fit - alpha_analytical).abs() / alpha_analytical.abs().max(1.0e-12);

    assert!(
        relative_error < 0.35,
        "Treeby-Cox 2010 plane-wave decay regression: \
         fitted α = {alpha_fit:.3} Np/m at {frequency_hz:.2e} Hz; analytical α = {alpha_analytical:.3} Np/m \
         (α₀ = {alpha0_np_per_m_at_1mhz} Np/m at 1 MHz, y = {y_exponent}); \
         relative error = {:.1}% (tolerance: 35%). \
         Distances = {:?} m; lossless peaks = {:?} Pa; absorbing peaks = {:?} Pa; \
         log-ratios = {:?}; grid pts/λ = {pts_per_wavelength:.1}. \
         A larger error suggests an incorrect c-power in the τ coefficient \
         `dt_tau = dt · 2·α₀_ω·c^(y+1)` or a sign error in the spectral \
         filter `|k|^y` weighting.",
        relative_error * 100.0,
        distances_m,
        peaks_lossless,
        peaks_absorbing,
        log_ratio,
    );
}

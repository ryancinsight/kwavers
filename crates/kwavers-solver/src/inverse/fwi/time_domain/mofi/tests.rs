//! In-silico MOFI validation (mirrors Bates et al. 2026, Figs 1–2): recover a
//! known rigid (SE(2)) misalignment of a sound-speed template from acoustic data
//! alone, to sub-millimetre / sub-degree accuracy.

use super::nonrigid::{
    align_nonrigid, sample_displacement, warp_template, FfdBasis, FfdConfig, FfdField,
};
use super::transform::{transform_template, PlaneGeometry};
use super::{
    align, align_from, align_homotopy, align_pipeline, align_with_calibration, coarse_pose_search,
    default_homotopy, CoarseSearchConfig, MofiConfig, PipelineConfig, RigidTransform,
};
use crate::inverse::fwi::time_domain::{FwiEngine, FwiGeometry, FwiProcessor};
use crate::inverse::reconstruction::seismic::MisfitType;
use crate::inverse::seismic::parameters::FwiParameters;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_grid::Grid;
use kwavers_source::{GridSource, SourceMode};
use ndarray::{Array2, Array3};

/// Asymmetric two-blob sound-speed template on a water background. Asymmetry is
/// required for the rotation to be identifiable from the data.
fn phantom(nx: usize, ny: usize, c0: f64) -> Array3<f64> {
    let mut t = Array3::from_elem((nx, ny, 1), c0);
    for j in 0..ny {
        for i in 0..nx {
            let r1 = (i as f64 - 11.0).powi(2) + (j as f64 - 13.0).powi(2);
            let r2 = (i as f64 - 21.0).powi(2) + (j as f64 - 19.0).powi(2);
            t[[i, j, 0]] += 500.0 * (-r1 / 8.0).exp() + 300.0 * (-r2 / 8.0).exp();
        }
    }
    t
}

/// MOFI recovers a known (θ, δ₁, δ₂) misalignment from transmission/scatter data.
/// # Panics
/// - Panics on any assertion failure or solve error.
#[test]
fn mofi_recovers_known_rigid_misalignment() {
    let (nx, ny) = (32usize, 32);
    let dx = 1e-3;
    let grid = Grid::new(nx, ny, 1, dx, dx, dx).expect("grid");
    let c0 = SOUND_SPEED_WATER_SIM;
    let template = phantom(nx, ny, c0);

    // One source on the left; receivers on an inset frame (ring-like coverage,
    // as in a UST ring array) so rotation and translation are both observable.
    let nt = 200usize;
    let dt = 2e-7; // 2-D CFL c·dt/dx = 0.3 < 1/√2.
                   // Two simultaneous sources from orthogonal sides (left + top) give the
                   // wavefield two propagation directions, sharpening rotation sensitivity.
    let mut p_mask = Array3::from_elem((nx, ny, 1), 0.0_f64);
    p_mask[[2, 16, 0]] = 1.0;
    p_mask[[16, 2, 0]] = 1.0;
    let mut p_signal = Array2::zeros((1, nt));
    for t in 0..24 {
        let phase = (t as f64) * 0.35;
        p_signal[[0, t]] = (-phase * phase / 9.0).exp() * (2.0 * phase).sin();
    }
    let source = GridSource {
        p0: None,
        u0: None,
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Additive,
        u_mask: None,
        u_signal: None,
        u_mode: SourceMode::default(),
    };
    let mut sensor_mask = Array3::from_elem((nx, ny, 1), false);
    for i in 3..nx - 3 {
        sensor_mask[[i, 3, 0]] = true;
        sensor_mask[[i, ny - 4, 0]] = true;
    }
    for j in 3..ny - 3 {
        sensor_mask[[3, j, 0]] = true;
        sensor_mask[[nx - 4, j, 0]] = true;
    }
    let geometry = FwiGeometry::new(source, sensor_mask);

    let parameters = FwiParameters {
        nt,
        dt,
        frequency: 5e5,
        ..FwiParameters::default()
    };
    let processor = FwiProcessor::new(parameters).with_engine(FwiEngine::SecondOrderSelfAdjoint);

    // Ground-truth misalignment and the data it produces.
    let phi_true = RigidTransform {
        theta_rad: 6.0_f64.to_radians(),
        delta_x_m: 2.0e-3,
        delta_y_m: -1.5e-3,
    };
    let geom = PlaneGeometry::centered(nx, ny, dx, dx);
    let true_model = transform_template(&template, &phi_true, &geom, c0);
    let observed = processor
        .forward_model_sensor_only(&true_model, &geometry, &grid)
        .expect("observed data");

    let config = MofiConfig {
        max_iterations: 100,
        initial_step_m: 4e-3,
        background_c: c0,
        tolerance: 1e-8,
        ..MofiConfig::default()
    };
    let result =
        align(&processor, &template, &observed, &geometry, &grid, &config).expect("MOFI alignment");

    // (1) Misfit must collapse (the recovered template explains the data).
    assert!(
        result.final_misfit < 0.1 * result.initial_misfit,
        "MOFI must substantially reduce the misfit; initial = {:e}, final = {:e}",
        result.initial_misfit,
        result.final_misfit
    );

    // (2) Recovered transform within the paper's tolerances: <1° and <1 mm.
    let d_theta_deg = (result.transform.theta_rad - phi_true.theta_rad)
        .to_degrees()
        .abs();
    let d_dx_mm = (result.transform.delta_x_m - phi_true.delta_x_m).abs() * 1e3;
    let d_dy_mm = (result.transform.delta_y_m - phi_true.delta_y_m).abs() * 1e3;
    assert!(
        d_theta_deg < 1.0,
        "rotation must be recovered to <1°; |Δθ| = {d_theta_deg:.4}° \
         (rec {:.4}°, true {:.4}°)",
        result.transform.theta_rad.to_degrees(),
        phi_true.theta_rad.to_degrees()
    );
    assert!(
        d_dx_mm < 1.0,
        "δ₁ must be recovered to <1 mm; |Δδ₁| = {d_dx_mm:.4} mm \
         (rec {:.4}, true {:.4})",
        result.transform.delta_x_m * 1e3,
        phi_true.delta_x_m * 1e3
    );
    assert!(
        d_dy_mm < 1.0,
        "δ₂ must be recovered to <1 mm; |Δδ₂| = {d_dy_mm:.4} mm \
         (rec {:.4}, true {:.4})",
        result.transform.delta_y_m * 1e3,
        phi_true.delta_y_m * 1e3
    );
}

/// Build the standard ring-array phantom problem (two orthogonal simultaneous
/// sources, inset receiver frame) and a self-adjoint MOFI processor.
fn ring_problem() -> (Grid, FwiGeometry, FwiProcessor, Array3<f64>, f64) {
    let (nx, ny) = (32usize, 32);
    let dx = 1e-3;
    let grid = Grid::new(nx, ny, 1, dx, dx, dx).expect("grid");
    let c0 = SOUND_SPEED_WATER_SIM;
    let template = phantom(nx, ny, c0);
    let nt = 200usize;
    let mut p_mask = Array3::from_elem((nx, ny, 1), 0.0_f64);
    p_mask[[2, 16, 0]] = 1.0;
    p_mask[[16, 2, 0]] = 1.0;
    let mut p_signal = Array2::zeros((1, nt));
    for t in 0..24 {
        let phase = (t as f64) * 0.35;
        p_signal[[0, t]] = (-phase * phase / 9.0).exp() * (2.0 * phase).sin();
    }
    let source = GridSource {
        p0: None,
        u0: None,
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Additive,
        u_mask: None,
        u_signal: None,
        u_mode: SourceMode::default(),
    };
    let mut sensor_mask = Array3::from_elem((nx, ny, 1), false);
    for i in 3..nx - 3 {
        sensor_mask[[i, 3, 0]] = true;
        sensor_mask[[i, ny - 4, 0]] = true;
    }
    for j in 3..ny - 3 {
        sensor_mask[[3, j, 0]] = true;
        sensor_mask[[nx - 4, j, 0]] = true;
    }
    let geometry = FwiGeometry::new(source, sensor_mask);
    let parameters = FwiParameters {
        nt,
        dt: 2e-7,
        frequency: 5e5,
        ..FwiParameters::default()
    };
    let processor = FwiProcessor::new(parameters).with_engine(FwiEngine::SecondOrderSelfAdjoint);
    (grid, geometry, processor, template, c0)
}

/// A misfit homotopy (Wasserstein → envelope → L2) recovers a LARGE rotation
/// that single-misfit L2 MOFI cannot, and is never worse than plain L2.
/// # Panics
/// - Panics on any assertion failure or solve error.
#[test]
fn homotopy_recovers_large_rotation() {
    let (grid, geometry, processor, template, c0) = ring_problem();
    let (nx, ny, _) = grid.dimensions();
    let geom = PlaneGeometry::centered(nx, ny, grid.dx, grid.dy);

    // A large pose error, outside the single-misfit L2 capture basin.
    let phi_true = RigidTransform {
        theta_rad: 28.0_f64.to_radians(),
        delta_x_m: 2.5e-3,
        delta_y_m: -1.5e-3,
    };
    let true_model = transform_template(&template, &phi_true, &geom, c0);
    let observed = processor
        .forward_model_sensor_only(&true_model, &geometry, &grid)
        .expect("observed");

    let config = MofiConfig {
        max_iterations: 60,
        initial_step_m: 5e-3,
        background_c: c0,
        tolerance: 1e-8,
        ..MofiConfig::default()
    };

    let plain =
        align(&processor, &template, &observed, &geometry, &grid, &config).expect("plain L2 MOFI");
    let homotopy = align_homotopy(
        &processor,
        &template,
        &observed,
        &geometry,
        &grid,
        &default_homotopy(config),
    )
    .expect("MOFI homotopy");

    let err = |r: &RigidTransform| -> (f64, f64) {
        let dtheta = (r.theta_rad - phi_true.theta_rad).to_degrees().abs();
        let dtrans = (((r.delta_x_m - phi_true.delta_x_m).powi(2)
            + (r.delta_y_m - phi_true.delta_y_m).powi(2))
        .sqrt())
            * 1e3;
        (dtheta, dtrans)
    };
    let (h_theta, h_trans) = err(&homotopy.transform);
    let (p_theta, p_trans) = err(&plain.transform);

    // (1) Homotopy recovers the large misalignment to near the paper tolerance.
    assert!(
        h_theta < 1.5 && h_trans < 1.0,
        "homotopy must recover the large pose; |Δθ| = {h_theta:.3}°, |Δδ| = {h_trans:.3} mm \
         (plain L2: {p_theta:.3}°, {p_trans:.3} mm)"
    );
    // (2) Homotopy is at least as accurate as plain L2 (no regression; it
    //     typically rescues a case where plain L2 is cycle-skipped).
    assert!(
        h_theta <= p_theta + 1e-6,
        "homotopy rotation error must not exceed plain L2; homotopy {h_theta:.3}°, plain {p_theta:.3}°"
    );
}

/// Joint pose + sound-speed calibration recovers BOTH a rigid misalignment and a
/// systematic template-speed error (`α ≠ 1`) that rigid alignment alone cannot.
/// # Panics
/// - Panics on any assertion failure or solve error.
#[test]
fn calibration_recovers_pose_and_speed_error() {
    let (grid, geometry, processor, template, c0) = ring_problem();
    let (nx, ny, _) = grid.dimensions();
    let geom = PlaneGeometry::centered(nx, ny, grid.dx, grid.dy);

    // Truth: a modest pose error AND the real skull is 25% faster in contrast
    // than the template predicts (α_true = 1.25).
    let phi_true = RigidTransform {
        theta_rad: 5.0_f64.to_radians(),
        delta_x_m: 1.5e-3,
        delta_y_m: -1.0e-3,
    };
    let alpha_true = 1.25_f64;
    let posed = transform_template(&template, &phi_true, &geom, c0);
    let true_model = posed.mapv(|c| c0 + alpha_true * (c - c0));
    let observed = processor
        .forward_model_sensor_only(&true_model, &geometry, &grid)
        .expect("observed");

    let config = MofiConfig {
        max_iterations: 25,
        initial_step_m: 4e-3,
        background_c: c0,
        tolerance: 1e-8,
        ..MofiConfig::default()
    };

    // Rigid-only MOFI (α fixed at 1) cannot explain the speed error.
    let rigid_only =
        align(&processor, &template, &observed, &geometry, &grid, &config).expect("rigid-only");
    let rigid_misfit = rigid_only.final_misfit;

    let result = align_with_calibration(
        &processor,
        &template,
        &observed,
        &geometry,
        &grid,
        &config,
        4,
        RigidTransform::identity(),
    )
    .expect("calibrated MOFI");

    // (1) Speed scale recovered.
    assert!(
        (result.speed_scale - alpha_true).abs() < 0.05,
        "α must be recovered to <0.05; got {:.4} (true {alpha_true})",
        result.speed_scale
    );
    // (2) Pose still recovered to paper tolerance.
    let d_theta = (result.transform.theta_rad - phi_true.theta_rad)
        .to_degrees()
        .abs();
    let d_trans = (((result.transform.delta_x_m - phi_true.delta_x_m).powi(2)
        + (result.transform.delta_y_m - phi_true.delta_y_m).powi(2))
    .sqrt())
        * 1e3;
    assert!(
        d_theta < 1.0 && d_trans < 1.0,
        "calibrated pose must stay <1°/<1mm; |Δθ|={d_theta:.3}°, |Δδ|={d_trans:.3}mm"
    );
    // (3) Calibration explains the data far better than rigid-only.
    assert!(
        result.final_misfit < 0.5 * rigid_misfit,
        "calibration must beat rigid-only misfit; calibrated {:e}, rigid-only {rigid_misfit:e}",
        result.final_misfit
    );
}

/// The coarse global pose search rescues a far misalignment that local MOFI from
/// the identity cannot reach, then a warm-started refine nails it.
/// # Panics
/// - Panics on any assertion failure or solve error.
#[test]
fn coarse_initializer_rescues_far_misalignment() {
    let (grid, geometry, processor, template, c0) = ring_problem();
    let (nx, ny, _) = grid.dimensions();
    let geom = PlaneGeometry::centered(nx, ny, grid.dx, grid.dy);

    // A far pose error well outside any local basin.
    let phi_true = RigidTransform {
        theta_rad: 45.0_f64.to_radians(),
        delta_x_m: 4.0e-3,
        delta_y_m: -3.0e-3,
    };
    let true_model = transform_template(&template, &phi_true, &geom, c0);
    let observed = processor
        .forward_model_sensor_only(&true_model, &geometry, &grid)
        .expect("observed");

    let config = MofiConfig {
        max_iterations: 50,
        initial_step_m: 4e-3,
        background_c: c0,
        tolerance: 1e-8,
        ..MofiConfig::default()
    };

    let err = |r: &RigidTransform| -> (f64, f64) {
        let dtheta = (r.theta_rad - phi_true.theta_rad).to_degrees().abs();
        let dtrans = (((r.delta_x_m - phi_true.delta_x_m).powi(2)
            + (r.delta_y_m - phi_true.delta_y_m).powi(2))
        .sqrt())
            * 1e3;
        (dtheta, dtrans)
    };

    // Local MOFI from identity: expected to miss the far optimum.
    let plain = align(&processor, &template, &observed, &geometry, &grid, &config).expect("plain");
    let (plain_theta, _) = err(&plain.transform);

    // Coarse search → warm-started refine.
    let search = CoarseSearchConfig {
        theta_max_rad: 60.0_f64.to_radians(),
        theta_steps: 9,
        delta_max_m: 6e-3,
        delta_steps: 7,
        background_c: c0,
    };
    // The global search uses a cycle-skip-robust BUT arrival-time-sensitive misfit
    // (Wasserstein/optimal transport) so its minimum is the true pose, not an L2
    // alias. (Envelope is robust but phase-blind, hence rotation-insensitive — a
    // poor pose search.) The local refine then uses L2 precision.
    let search_processor = processor.clone().with_misfit(MisfitType::Wasserstein);
    let (seed, _) = coarse_pose_search(
        &search_processor,
        &template,
        &observed,
        &geometry,
        &grid,
        &search,
    )
    .expect("coarse search");
    let refined = align_from(
        &processor, &template, &observed, &geometry, &grid, &config, seed,
    )
    .expect("refine");
    let (r_theta, r_trans) = err(&refined.transform);

    assert!(
        r_theta < 1.5 && r_trans < 1.5,
        "coarse-initialised MOFI must recover the far pose; |Δθ|={r_theta:.3}°, |Δδ|={r_trans:.3}mm"
    );
    assert!(
        r_theta < plain_theta,
        "coarse initialisation must beat local-from-identity; refined {r_theta:.3}°, plain {plain_theta:.3}°"
    );
}

/// Non-rigid FFD recovers a known smooth deformation that no rigid transform can
/// represent: the misfit collapses and the recovered dense displacement field
/// approximates the truth in the illuminated region.
/// # Panics
/// - Panics on any assertion failure or solve error.
#[test]
fn nonrigid_ffd_recovers_smooth_deformation() {
    let (grid, geometry, processor, template, c0) = ring_problem();
    let (nx, ny, _) = grid.dimensions();
    let geom = PlaneGeometry::centered(nx, ny, grid.dx, grid.dy);

    // No source rescaling: the FFD smoothness weight is now RELATIVE to the data
    // misfit (auto-scaled internally), so the regulariser is balanced regardless
    // of the engine's absolute amplitude scale.

    // Ground-truth non-rigid warp over the feature region (not expressible as a
    // rigid SE(2) transform): displacements at the interior lattice points.
    let mut true_field = FfdField::zeros(4, 4, FfdBasis::Bilinear);
    let set = |f: &mut FfdField, p: usize, q: usize, ux: f64, uy: f64| {
        let idx = q * f.n_ctrl_x + p;
        f.ux[idx] = ux;
        f.uy[idx] = uy;
    };
    set(&mut true_field, 1, 1, 2.5e-3, -2.0e-3);
    set(&mut true_field, 2, 1, 2.0e-3, 2.0e-3);
    set(&mut true_field, 1, 2, -2.0e-3, 2.5e-3);
    set(&mut true_field, 2, 2, -2.5e-3, -2.0e-3);
    let true_model = warp_template(&template, &true_field, &geom, c0);
    let observed = processor
        .forward_model_sensor_only(&true_model, &geometry, &grid)
        .expect("observed");

    let config = FfdConfig {
        n_ctrl_x: 4,
        n_ctrl_y: 4,
        max_iterations: 80,
        initial_step_m: 2e-3,
        smoothness_weight: 1e-4, // relative to J₀ (auto-scaled internally)
        background_c: c0,
        tolerance: 1e-9,
        ..FfdConfig::default()
    };
    let result = align_nonrigid(&processor, &template, &observed, &geometry, &grid, &config)
        .expect("FFD alignment");

    // (1) The deformation explains the data.
    assert!(
        result.final_misfit < 0.2 * result.initial_misfit,
        "FFD must substantially reduce the misfit; initial {:e}, final {:e}",
        result.initial_misfit,
        result.final_misfit
    );

    // (2) The recovered dense displacement matches the truth in the illuminated
    //     feature region (FFD is non-unique globally; compare where the data
    //     constrains it). RMS displacement error well below the 1.5 mm peak.
    let mut sq = 0.0;
    let mut n = 0usize;
    for j in 8..24 {
        for i in 8..24 {
            let (tux, tuy) = sample_displacement(&true_field, i, j, nx, ny);
            let (rux, ruy) = sample_displacement(&result.field, i, j, nx, ny);
            sq += (rux - tux).powi(2) + (ruy - tuy).powi(2);
            n += 1;
        }
    }
    let rms_mm = (sq / n as f64).sqrt() * 1e3;
    assert!(
        rms_mm < 1.0,
        "recovered displacement field must approximate the truth in the illuminated \
         region; RMS error = {rms_mm:.3} mm (true peak ~2.5 mm)"
    );
}

/// The cubic-B-spline FFD basis (C² deformations) recovers a smooth warp, like
/// the bilinear basis but with a smoother lattice expansion.
/// # Panics
/// - Panics on any assertion failure or solve error.
#[test]
fn nonrigid_ffd_cubic_bspline_recovers_smooth_deformation() {
    let (grid, geometry, processor, template, c0) = ring_problem();
    let (nx, ny, _) = grid.dimensions();
    let geom = PlaneGeometry::centered(nx, ny, grid.dx, grid.dy);

    let mut true_field = FfdField::zeros(4, 4, FfdBasis::CubicBSpline);
    let set = |f: &mut FfdField, p: usize, q: usize, ux: f64, uy: f64| {
        let idx = q * f.n_ctrl_x + p;
        f.ux[idx] = ux;
        f.uy[idx] = uy;
    };
    set(&mut true_field, 1, 1, 2.0e-3, -1.5e-3);
    set(&mut true_field, 2, 2, -2.0e-3, 1.5e-3);
    let true_model = warp_template(&template, &true_field, &geom, c0);
    let observed = processor
        .forward_model_sensor_only(&true_model, &geometry, &grid)
        .expect("observed");

    let config = FfdConfig {
        n_ctrl_x: 4,
        n_ctrl_y: 4,
        basis: FfdBasis::CubicBSpline,
        max_iterations: 80,
        initial_step_m: 2e-3,
        smoothness_weight: 1e-4,
        background_c: c0,
        tolerance: 1e-9,
        ..FfdConfig::default()
    };
    let result = align_nonrigid(&processor, &template, &observed, &geometry, &grid, &config)
        .expect("cubic FFD");

    assert!(
        result.final_misfit < 0.2 * result.initial_misfit,
        "cubic FFD must reduce misfit; initial {:e}, final {:e}",
        result.initial_misfit,
        result.final_misfit
    );
    let mut sq = 0.0;
    let mut n = 0usize;
    for j in 8..24 {
        for i in 8..24 {
            let (tux, tuy) = sample_displacement(&true_field, i, j, nx, ny);
            let (rux, ruy) = sample_displacement(&result.field, i, j, nx, ny);
            sq += (rux - tux).powi(2) + (ruy - tuy).powi(2);
            n += 1;
        }
    }
    let rms_mm = (sq / n as f64).sqrt() * 1e3;
    assert!(
        rms_mm < 1.0,
        "cubic FFD displacement RMS error must be small; {rms_mm:.3} mm (peak ~2 mm)"
    );
}

/// Full pipeline (coarse search → rigid + calibration → non-rigid) recovers a
/// compound misalignment — large rotation + translation + speed error + a
/// non-rigid warp — that no single pathway handles alone.
/// # Panics
/// - Panics on any assertion failure or solve error.
#[test]
fn full_pipeline_recovers_compound_misalignment() {
    let (grid, geometry, processor, template, c0) = ring_problem();
    let (nx, ny, _) = grid.dimensions();
    let geom = PlaneGeometry::centered(nx, ny, grid.dx, grid.dy);

    // Compound ground truth: pose, contrast scale, and a non-rigid warp, applied
    // in the pipeline's compositional order: warp ∘ pose ∘ contrast-scale.
    let pose_true = RigidTransform {
        theta_rad: 20.0_f64.to_radians(),
        delta_x_m: 2.5e-3,
        delta_y_m: -1.5e-3,
    };
    let alpha_true = 1.2_f64;
    let mut ffd_true = FfdField::zeros(4, 4, FfdBasis::Bilinear);
    let set = |f: &mut FfdField, p: usize, q: usize, ux: f64, uy: f64| {
        let idx = q * f.n_ctrl_x + p;
        f.ux[idx] = ux;
        f.uy[idx] = uy;
    };
    set(&mut ffd_true, 1, 1, 1.5e-3, -1.2e-3);
    set(&mut ffd_true, 2, 2, -1.5e-3, 1.2e-3);

    let scaled = template.mapv(|c| c0 + alpha_true * (c - c0));
    let posed = transform_template(&scaled, &pose_true, &geom, c0);
    let true_model = warp_template(&posed, &ffd_true, &geom, c0);
    let observed = processor
        .forward_model_sensor_only(&true_model, &geometry, &grid)
        .expect("observed");

    let config = PipelineConfig {
        coarse: CoarseSearchConfig {
            theta_max_rad: 45.0_f64.to_radians(),
            theta_steps: 7,
            delta_max_m: 4e-3,
            delta_steps: 5,
            background_c: c0,
        },
        search_misfit: MisfitType::Wasserstein,
        rigid: MofiConfig {
            max_iterations: 30,
            initial_step_m: 4e-3,
            background_c: c0,
            tolerance: 1e-8,
            ..MofiConfig::default()
        },
        calibration_outer: 3,
        ffd: FfdConfig {
            n_ctrl_x: 4,
            n_ctrl_y: 4,
            max_iterations: 50,
            initial_step_m: 1.5e-3,
            smoothness_weight: 1e-4, // relative to J₀ (auto-scaled internally)
            background_c: c0,
            tolerance: 1e-9,
            ..FfdConfig::default()
        },
    };

    let result = align_pipeline(&processor, &template, &observed, &geometry, &grid, &config)
        .expect("pipeline");

    // (1) The compound misalignment is explained: large misfit reduction.
    assert!(
        result.final_misfit < 0.1 * result.initial_misfit,
        "pipeline must collapse the misfit; initial {:e}, final {:e}",
        result.initial_misfit,
        result.final_misfit
    );
    // (2) The COMBINED aligned model matches the truth in the illuminated region.
    //     (The rigid pose alone is biased by the warp — the pose/FFD split is not
    //     unique — so model-space agreement, not pose-vs-truth, is the meaningful
    //     metric for a compound misalignment.)
    let final_scaled = template.mapv(|c| c0 + result.speed_scale * (c - c0));
    let final_posed = transform_template(&final_scaled, &result.transform, &geom, c0);
    let final_model = warp_template(&final_posed, &result.ffd, &geom, c0);
    let mut sq = 0.0;
    let mut n = 0usize;
    for j in 8..24 {
        for i in 8..24 {
            sq += (final_model[[i, j, 0]] - true_model[[i, j, 0]]).powi(2);
            n += 1;
        }
    }
    let rms_c = (sq / n as f64).sqrt();
    assert!(
        rms_c < 40.0,
        "aligned model must match truth in the illuminated region; RMS Δc = {rms_c:.2} m/s \
         (contrast ~500 m/s)"
    );
    // (3) Sound-speed calibration recovered.
    assert!(
        (result.speed_scale - alpha_true).abs() < 0.15,
        "pipeline α must be close; got {:.4} (true {alpha_true})",
        result.speed_scale
    );
}

/// MOFI is stationary at the truth: starting already aligned, it neither moves
/// nor invents a spurious offset.
/// # Panics
/// - Panics on any assertion failure or solve error.
#[test]
fn mofi_is_stationary_when_already_aligned() {
    let (nx, ny) = (32usize, 32);
    let dx = 1e-3;
    let grid = Grid::new(nx, ny, 1, dx, dx, dx).expect("grid");
    let c0 = SOUND_SPEED_WATER_SIM;
    let template = phantom(nx, ny, c0);

    let nt = 200usize;
    let mut p_mask = Array3::from_elem((nx, ny, 1), 0.0_f64);
    p_mask[[2, 16, 0]] = 1.0;
    let mut p_signal = Array2::zeros((1, nt));
    for t in 0..24 {
        let phase = (t as f64) * 0.35;
        p_signal[[0, t]] = (-phase * phase / 9.0).exp() * (2.0 * phase).sin();
    }
    let source = GridSource {
        p0: None,
        u0: None,
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Additive,
        u_mask: None,
        u_signal: None,
        u_mode: SourceMode::default(),
    };
    let mut sensor_mask = Array3::from_elem((nx, ny, 1), false);
    for j in 3..ny - 3 {
        sensor_mask[[nx - 4, j, 0]] = true;
    }
    let geometry = FwiGeometry::new(source, sensor_mask);
    let parameters = FwiParameters {
        nt,
        dt: 2e-7,
        frequency: 5e5,
        ..FwiParameters::default()
    };
    let processor = FwiProcessor::new(parameters).with_engine(FwiEngine::SecondOrderSelfAdjoint);

    // Observed generated from the untransformed template ⇒ φ = 0 is optimal.
    let observed = processor
        .forward_model_sensor_only(&template, &geometry, &grid)
        .expect("observed");

    let result = align(
        &processor,
        &template,
        &observed,
        &geometry,
        &grid,
        &MofiConfig {
            background_c: c0,
            ..MofiConfig::default()
        },
    )
    .expect("MOFI");

    assert!(
        result.transform.theta_rad.to_degrees().abs() < 0.2,
        "rotation must stay ~0 at the aligned truth; got {:.4}°",
        result.transform.theta_rad.to_degrees()
    );
    assert!(
        result.transform.delta_x_m.abs() * 1e3 < 0.2
            && result.transform.delta_y_m.abs() * 1e3 < 0.2,
        "translation must stay ~0 at the aligned truth; got ({:.4}, {:.4}) mm",
        result.transform.delta_x_m * 1e3,
        result.transform.delta_y_m * 1e3
    );
}

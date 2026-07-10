//! End-to-end value-semantic tests for the L-BFGS FWI driver
//! ([`FwiProcessor::invert_lbfgs`]).
//!
//! The test builds a true velocity model with a localized fast anomaly,
//! synthesizes "observed" data by forward-modelling it, then inverts from a
//! homogeneous starting model. L-BFGS must (a) drive the data misfit well below
//! the starting misfit and (b) move the model toward the truth at the anomaly.

use super::super::{FwiEngine, FwiGeometry, FwiProcessor};
use crate::inverse::seismic::parameters::FwiParameters;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_grid::Grid;
use kwavers_source::{GridSource, SourceMode};
use leto::{
    Array2,
    Array3,
};

/// Build the shared small-grid FWI problem: grid, geometry, true model, and a
/// homogeneous initial model. The anomaly is a `+anomaly` m/s Gaussian bump at
/// the grid centre.
fn build_problem(anomaly: f64) -> (Grid, FwiGeometry, FwiParameters, Array3<f64>, Array3<f64>) {
    let (nx, ny, nz) = (8usize, 8, 8);
    let dx = 1e-3;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).expect("grid");
    let dims = (nx, ny, nz);

    let c0 = SOUND_SPEED_WATER_SIM;
    let initial = Array3::from_elem(dims, c0);
    let mut truth = initial.clone();
    for ([ix, iy, iz], value) in truth.indexed_iter_mut().expect("invariant: owned array yields indexed iterator") {
        let r2 = (ix as f64 - 3.5).powi(2) + (iy as f64 - 3.5).powi(2) + (iz as f64 - 3.5).powi(2);
        *value += anomaly * (-r2 / 3.0).exp();
    }

    let mut sensor_mask = Array3::from_elem(dims, false);
    for iy in 2..6 {
        for iz in 2..6 {
            sensor_mask[[6, iy, iz]] = true;
        }
    }
    let nt = 96usize;
    let dt = 1e-7;

    let mut p_mask = Array3::from_elem(dims, 0.0_f64);
    p_mask[[1, 4, 4]] = 1.0;
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
    let geometry = FwiGeometry::new(source, sensor_mask);

    // Pure data-misfit inversion: disable regularization so the test exercises
    // the optimizer rather than the prior. (With the default Tikhonov weight,
    // the `λ·c` term — O(1) — swamps the small physical gradient of this
    // few-voxel problem by >10 orders of magnitude.) Mute the near-source
    // artefact per the `source_mute_radius` recommendation.
    let regularization = crate::inverse::seismic::parameters::RegularizationParameters {
        tikhonov_weight: 0.0,
        tv_weight: 0.0,
        directional_tv_weight: 0.0,
        directional_tv_adaptive: false,
        smoothness_weight: 0.0,
    };
    let parameters = FwiParameters {
        nt,
        dt,
        frequency: 5e5,
        max_iterations: 8,
        step_size: 50.0,
        tolerance: 1e-6,
        source_mute_radius: 3,
        regularization,
        ..FwiParameters::default()
    };

    (grid, geometry, parameters, truth, initial)
}

/// L-BFGS reduces the data misfit and moves the model toward the truth.
/// # Panics
/// - Panics on any assertion failure or solver error.
#[test]
fn lbfgs_reduces_misfit_and_recovers_anomaly() {
    let anomaly = 60.0_f64;
    let (grid, geometry, parameters, truth, initial) = build_problem(anomaly);

    let processor = FwiProcessor::new(parameters);

    // Synthesize "observed" data from the true model.
    let (observed, _history) = processor
        .forward_model(&truth, &geometry, &grid)
        .expect("observed forward");

    let initial_misfit = processor
        .compute_objective(&initial, &observed, &geometry, &grid)
        .expect("initial misfit");
    assert!(
        initial_misfit > 0.0,
        "initial misfit must be non-zero (initial != truth); got {initial_misfit:e}"
    );

    let recovered = processor
        .invert_lbfgs(&observed, &initial, &geometry, &grid, 6)
        .expect("L-BFGS inversion");

    let final_misfit = processor
        .compute_objective(&recovered, &observed, &geometry, &grid)
        .expect("final misfit");

    // (a) the misfit must drop substantially.
    assert!(
        final_misfit < 0.5 * initial_misfit,
        "L-BFGS must at least halve the misfit; initial = {initial_misfit:e}, \
         final = {final_misfit:e}"
    );

    // (b) the model must move toward the truth at the anomaly centre: the
    //     recovered velocity there must exceed the homogeneous start (the true
    //     anomaly is positive) and not overshoot the truth grossly.
    let c_init = initial[[3, 3, 3]];
    let c_true = truth[[3, 3, 3]];
    let c_rec = recovered[[3, 3, 3]];
    assert!(
        c_rec > c_init,
        "recovered anomaly must rise above the homogeneous start; \
         init = {c_init}, recovered = {c_rec}"
    );
    let init_err = (c_init - c_true).abs();
    let rec_err = (c_rec - c_true).abs();
    assert!(
        rec_err < init_err,
        "recovered model must be closer to truth at the anomaly; \
         |init−true| = {init_err}, |rec−true| = {rec_err}"
    );

    // (c) the model error in the *illuminated* region (the ray-covered core
    //     between the source at ix=1 and the receiver line at ix=6, where the
    //     anomaly sits) must decrease. The global RMS is NOT asserted: a
    //     single-shot, limited-aperture FWI is globally ill-posed — it reduces
    //     the data misfit and improves the illuminated zone but can place
    //     compensating artefacts in unilluminated voxels, so global model RMS is
    //     not guaranteed to fall (Virieux & Operto 2009 §4: model nullspace).
    let region_rms = |a: &Array3<f64>| -> f64 {
        let mut sum = 0.0;
        let mut n = 0usize;
        for ix in 2..6 {
            for iy in 2..6 {
                for iz in 2..6 {
                    sum += (a[[ix, iy, iz]] - truth[[ix, iy, iz]]).powi(2);
                    n += 1;
                }
            }
        }
        (sum / n as f64).sqrt()
    };
    assert!(
        region_rms(&recovered) < region_rms(&initial),
        "illuminated-region RMS model error must decrease: init = {}, recovered = {}",
        region_rms(&initial),
        region_rms(&recovered)
    );
}

/// The **exact-gradient** self-adjoint engine (ADR 016) drives the same
/// orchestration (forward → exact adjoint → L-BFGS) end to end: misfit drops and
/// the anomaly is recovered. This exercises the `FwiEngine::SecondOrderSelfAdjoint`
/// dispatch through `forward_model`, `misfit_and_gradient`, and the line search.
/// # Panics
/// - Panics on any assertion failure or solver error.
#[test]
fn self_adjoint_engine_lbfgs_reduces_misfit_and_recovers_anomaly() {
    let anomaly = 60.0_f64;
    let (grid, geometry, parameters, truth, initial) = build_problem(anomaly);

    // No source-amplitude rescaling: the self-adjoint engine's dt²·ρc²·s injection
    // gives J≈5e-17 and ‖g‖∞~1e-18, but invert_lbfgs now judges convergence
    // RELATIVE to the initial gradient norm, so small-amplitude data is handled
    // directly (no absolute f64::EPSILON guard to trip).
    let processor = FwiProcessor::new(parameters).with_engine(FwiEngine::SecondOrderSelfAdjoint);

    // "Observed" data must come from the same (self-adjoint) forward map.
    let (observed, _history) = processor
        .forward_model(&truth, &geometry, &grid)
        .expect("observed forward (self-adjoint)");

    let initial_misfit = processor
        .compute_objective(&initial, &observed, &geometry, &grid)
        .expect("initial misfit");
    assert!(
        initial_misfit > 0.0,
        "initial misfit must be non-zero; got {initial_misfit:e}"
    );

    let recovered = processor
        .invert_lbfgs(&observed, &initial, &geometry, &grid, 6)
        .expect("self-adjoint L-BFGS inversion");

    let final_misfit = processor
        .compute_objective(&recovered, &observed, &geometry, &grid)
        .expect("final misfit");
    assert!(
        final_misfit < 0.5 * initial_misfit,
        "self-adjoint FWI must at least halve the misfit; initial = {initial_misfit:e}, \
         final = {final_misfit:e}"
    );

    let c_init = initial[[3, 3, 3]];
    let c_true = truth[[3, 3, 3]];
    let c_rec = recovered[[3, 3, 3]];
    assert!(
        c_rec > c_init,
        "recovered anomaly must rise above the homogeneous start; init = {c_init}, rec = {c_rec}"
    );
    assert!(
        (c_rec - c_true).abs() < (c_init - c_true).abs(),
        "recovered model must be closer to truth at the anomaly; \
         |init−true| = {}, |rec−true| = {}",
        (c_init - c_true).abs(),
        (c_rec - c_true).abs()
    );
}

/// At the true model the gradient is ~0, so L-BFGS must return immediately with
/// the model essentially unchanged (no spurious update).
/// # Panics
/// - Panics on any assertion failure or solver error.
#[test]
fn lbfgs_is_stationary_at_the_truth() {
    let (grid, geometry, parameters, truth, _initial) = build_problem(60.0);
    let processor = FwiProcessor::new(parameters);

    let (observed, _h) = processor
        .forward_model(&truth, &geometry, &grid)
        .expect("observed forward");

    // Start at the truth: misfit is zero, so any accepted step would have to
    // strictly decrease an already-zero objective — impossible → no movement.
    let recovered = processor
        .invert_lbfgs(&observed, &truth, &geometry, &grid, 6)
        .expect("L-BFGS at truth");

    let max_move = recovered
        .iter()
        .zip(truth.iter())
        .fold(0.0_f64, |m, (&r, &t)| m.max((r - t).abs()));
    assert!(
        max_move < 1e-6,
        "L-BFGS must not move away from the true (zero-misfit) model; \
         max |Δc| = {max_move:e}"
    );
}

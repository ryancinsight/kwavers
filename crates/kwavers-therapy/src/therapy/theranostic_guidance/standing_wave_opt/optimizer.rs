//! Gradient-descent phase optimizer for standing-wave suppression.
//!
//! # Objective
//!
//! ```text
//! f(φ) = w_swi × SWI(φ)  −  w_focal × p_focal(φ) / p_focal_ref
//! ```
//!
//! Gradient `∂f/∂φ_i` is computed by central differences:
//!
//! ```text
//! ∂f/∂φ_i ≈ [f(φ + δ eᵢ) − f(φ − δ eᵢ)] / (2δ)
//! ```
//!
//! Each evaluation is an `O(n_elements × nx × ny)` superposition plus an
//! `O(N)` DFT, which is essentially instant after the Green's functions are
//! precomputed.  The step size `α` is selected by Armijo backtracking.

use leto::Array2;

use super::{
    config::StandingWaveOptConfig,
    result::StandingWaveOptResult,
    swi::{compute_focal_pressure, compute_swi, superpose},
};
use kwavers_core::constants::numerical::TWO_PI;

// ---------------------------------------------------------------------------
// Objective
// ---------------------------------------------------------------------------

fn objective(
    phases: &[f64],
    g_re: &[Array2<f64>],
    g_im: &[Array2<f64>],
    p_focal_ref: f64,
    config: &StandingWaveOptConfig,
) -> (f64, f64, f64, Array2<f64>, Array2<f64>) {
    let (p_re, p_im) = superpose(g_re, g_im, phases);
    let swi = compute_swi(&p_re, &p_im, config);
    let p_focal = compute_focal_pressure(&p_re, &p_im, config);
    let p_norm = p_focal / p_focal_ref.max(1e-30);
    let obj = config.swi_weight * swi - config.focal_weight * p_norm;
    (obj, swi, p_focal, p_re, p_im)
}

// ---------------------------------------------------------------------------
// Gradient via central differences
// ---------------------------------------------------------------------------

fn gradient(
    phases: &[f64],
    g_re: &[Array2<f64>],
    g_im: &[Array2<f64>],
    p_focal_ref: f64,
    config: &StandingWaveOptConfig,
) -> Vec<f64> {
    let n = phases.len();
    let delta = config.grad_delta_rad;
    let mut grad = vec![0.0f64; n];
    for i in 0..n {
        let mut phi_fwd = phases.to_vec();
        let mut phi_bwd = phases.to_vec();
        phi_fwd[i] += delta;
        phi_bwd[i] -= delta;
        let (f_fwd, _, _, _, _) = objective(&phi_fwd, g_re, g_im, p_focal_ref, config);
        let (f_bwd, _, _, _, _) = objective(&phi_bwd, g_re, g_im, p_focal_ref, config);
        grad[i] = (f_fwd - f_bwd) / (2.0 * delta);
    }
    grad
}

// ---------------------------------------------------------------------------
// Armijo backtracking line search
// ---------------------------------------------------------------------------

fn backtrack(
    phases: &[f64],
    grad: &[f64],
    f0: f64,
    g_re: &[Array2<f64>],
    g_im: &[Array2<f64>],
    p_focal_ref: f64,
    config: &StandingWaveOptConfig,
) -> f64 {
    let slope: f64 = grad.iter().map(|g| g * g).sum(); // ‖∇f‖²
    let mut alpha = config.line_search_alpha0;
    for _ in 0..config.line_search_max {
        let phi_trial: Vec<f64> = phases
            .iter()
            .zip(grad)
            .map(|(&p, &g)| p - alpha * g)
            .collect();
        let (f_trial, _, _, _, _) = objective(&phi_trial, g_re, g_im, p_focal_ref, config);
        if f_trial <= f0 - config.armijo_c1 * alpha * slope {
            return alpha;
        }
        alpha *= config.line_search_beta;
    }
    alpha
}

// ---------------------------------------------------------------------------
// Main optimization loop
// ---------------------------------------------------------------------------

fn wrap_phases(phases: Vec<f64>) -> Vec<f64> {
    phases
        .into_iter()
        .map(|p| (p + std::f64::consts::PI) % (TWO_PI) - std::f64::consts::PI)
        .collect()
}

fn snap_indices(n_iter: usize, n_snapshots: usize) -> Vec<usize> {
    if n_snapshots <= 1 {
        return vec![0];
    }
    (0..n_snapshots)
        .map(|k| (k * n_iter / (n_snapshots - 1)).min(n_iter))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(super) fn run_optimization(
    g_re: Vec<Array2<f64>>,
    g_im: Vec<Array2<f64>>,
    config: &StandingWaveOptConfig,
) -> StandingWaveOptResult {
    use leto::Array3;

    let element_ys = config.element_ys();
    let mut phases = config.das_phases(&element_ys);
    let nx = config.nx;
    let ny = config.ny;

    // Evaluate initial DAS state
    let (_, swi0, p_focal0, init_re, init_im) = objective(&phases, &g_re, &g_im, 1.0, config);
    let p_focal_ref = p_focal0.max(1e-30);

    let (obj0, _, _, _, _) = objective(&phases, &g_re, &g_im, p_focal_ref, config);
    let initial_phases = phases.clone();

    let mut swi_history = vec![swi0];
    let mut focal_history = vec![p_focal0];
    let mut obj_history = vec![obj0];

    // Snapshots
    let snap_iters = snap_indices(config.n_opt_iter, config.n_snapshots);
    let n_snap = snap_iters.len();
    let mut snap_re: Array3<f32> = Array3::zeros((n_snap, nx, ny));
    let mut snap_im: Array3<f32> = Array3::zeros((n_snap, nx, ny));
    let mut snap_iter_out: Vec<usize> = Vec::with_capacity(n_snap);
    let mut snap_cursor = 0usize;

    // Capture iter-0 snapshot
    if snap_iters.contains(&0) {
        snap_re
            .index_axis_mut::<2>(0, snap_cursor)
            .expect("invariant: snapshot index in bounds")
            .assign(&init_re.mapv(|v| v as f32).view());
        snap_im
            .index_axis_mut::<2>(0, snap_cursor)
            .expect("invariant: snapshot index in bounds")
            .assign(&init_im.mapv(|v| v as f32).view());
        snap_iter_out.push(0);
        snap_cursor += 1;
    }

    // Gradient-descent loop
    for k in 1..=config.n_opt_iter {
        let (f0, _, _, _, _) = objective(&phases, &g_re, &g_im, p_focal_ref, config);
        let grad = gradient(&phases, &g_re, &g_im, p_focal_ref, config);
        let alpha = backtrack(&phases, &grad, f0, &g_re, &g_im, p_focal_ref, config);
        phases = wrap_phases(
            phases
                .iter()
                .zip(&grad)
                .map(|(&p, &g)| p - alpha * g)
                .collect(),
        );

        let (obj_k, swi_k, pf_k, p_re_k, p_im_k) =
            objective(&phases, &g_re, &g_im, p_focal_ref, config);
        swi_history.push(swi_k);
        focal_history.push(pf_k);
        obj_history.push(obj_k);

        if snap_iters.contains(&k) && snap_cursor < n_snap {
            snap_re
                .index_axis_mut::<2>(0, snap_cursor)
                .expect("invariant: snapshot index in bounds")
                .assign(&p_re_k.mapv(|v| v as f32).view());
            snap_im
                .index_axis_mut::<2>(0, snap_cursor)
                .expect("invariant: snapshot index in bounds")
                .assign(&p_im_k.mapv(|v| v as f32).view());
            snap_iter_out.push(k);
            snap_cursor += 1;
        }

        if grad.iter().map(|g| g * g).sum::<f64>().sqrt() < 1e-8 {
            break;
        }
    }

    // Final field
    let (_, _, _, final_re, final_im) = objective(&phases, &g_re, &g_im, p_focal_ref, config);

    // Sound-speed map for visualisation
    use super::medium::build_sound_speed;
    let c_map = build_sound_speed(config);

    StandingWaveOptResult {
        nx,
        ny,
        dx_m: config.dx_m,
        frequency_hz: config.frequency_hz,
        n_elements: config.n_elements,
        element_ys,
        source_x: config.source_x,
        focus_x: config.focus_x,
        focus_y: config.focus_y,
        reflector_x_start: config.layer_x_start,
        reflector_x_end: config.layer_x_end,
        pml_cells: config.pml_cells,
        sound_speed_map: c_map.mapv(|v| v as f32),
        swi_history,
        focal_pressure_history: focal_history,
        objective_history: obj_history,
        initial_phases,
        final_phases: phases,
        snapshot_iterations: snap_iter_out,
        snapshot_fields_re: snap_re,
        snapshot_fields_im: snap_im,
        initial_field_re: init_re.mapv(|v| v as f32),
        initial_field_im: init_im.mapv(|v| v as f32),
        final_field_re: final_re.mapv(|v| v as f32),
        final_field_im: final_im.mapv(|v| v as f32),
        swi_weight: config.swi_weight,
        focal_weight: config.focal_weight,
        focal_pressure_ref_pa: p_focal_ref,
    }
}

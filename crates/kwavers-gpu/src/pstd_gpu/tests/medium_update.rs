//! Medium variable update and source-correction tests.

use super::helpers::{make_small_test_solver, read_buffer};

/// Regression guard: variable-only medium updates must preserve static
/// absorption/nonlinearity buffers, and source-kappa disablement must write
/// a resident unity buffer rather than a fresh allocation.
/// # Panics
/// - Panics if `non-empty c0`.
/// - Panics if `non-empty rho`.
///
#[test]
fn test_medium_variable_update_preserves_static_buffers_and_disables_source_correction() {
    let Some(mut solver) = make_small_test_solver() else {
        return;
    };

    let total = solver.nx * solver.ny * solver.nz;
    let new_c0: Vec<f32> = (0..total).map(|i| 1450.0f32 + i as f32 * 0.5).collect();
    let new_rho: Vec<f32> = (0..total).map(|i| 980.0f32 + i as f32 * 0.25).collect();
    solver.update_medium_variable(&new_c0, &new_rho);
    solver.disable_source_correction();

    let c0_sq: Vec<f32> = read_buffer(&solver.device, &solver.queue, &solver.buf_c0_sq, total);
    let rho0: Vec<f32> = read_buffer(&solver.device, &solver.queue, &solver.buf_rho0, total);
    let rho0_inv: Vec<f32> =
        read_buffer(&solver.device, &solver.queue, &solver.buf_rho0_inv, total);
    let bon_a: Vec<f32> = read_buffer(&solver.device, &solver.queue, &solver.buf_bon_a, total);
    let source_kappa: Vec<f32> = read_buffer(
        &solver.device,
        &solver.queue,
        &solver.buf_source_kappa,
        total,
    );

    let first_idx = 0usize;
    let last_idx = total - 1;
    let first_c0 = new_c0[0] as f64;
    let last_c0 = *new_c0.last().expect("non-empty c0") as f64;
    let first_rho = new_rho[0] as f64;
    let last_rho = *new_rho.last().expect("non-empty rho") as f64;

    assert!((c0_sq[first_idx] as f64 - first_c0 * first_c0).abs() < 1e-3);
    assert!((c0_sq[last_idx] as f64 - last_c0 * last_c0).abs() < 1e-3);
    assert!((rho0[first_idx] as f64 - first_rho).abs() < 1e-6);
    assert!((rho0[last_idx] as f64 - last_rho).abs() < 1e-6);
    assert!((rho0_inv[first_idx] as f64 - (1.0 / first_rho)).abs() < 1e-6);
    assert!((rho0_inv[last_idx] as f64 - (1.0 / last_rho)).abs() < 1e-6);
    let bon_a_expected = 0.375_f32;
    let unity_expected = 1.0_f32;
    let tol = 1e-6_f32;
    assert!(bon_a.iter().all(|&v| (v - bon_a_expected).abs() < tol));
    assert!(source_kappa
        .iter()
        .all(|&v| (v - unity_expected).abs() < tol));
}

//! Value-semantic tests for the optimal-transport (Wasserstein) and
//! normalized-cross-correlation misfits.
//!
//! These misfits mitigate cycle-skipping in FWI: the Wasserstein metric is
//! convex with respect to a time shift of the data, where the L2 misfit is
//! oscillatory. The tests pin both the zero-misfit identity and the
//! shift-monotonicity that gives the metrics that property.

use super::types::{MisfitFunction, MisfitType};
use ndarray::Array2;

/// A single trace holding a unit impulse at sample `pos`.
fn impulse_trace(n: usize, pos: usize) -> Array2<f64> {
    let mut a = Array2::zeros((1, n));
    a[[0, pos]] = 1.0;
    a
}

// ── Wasserstein (1-Wasserstein via CDF L1) ────────────────────────────────────

#[test]
fn wasserstein_is_zero_for_identical_data() {
    let m = MisfitFunction::new(MisfitType::Wasserstein);
    let d = impulse_trace(64, 20);
    let j = m.compute(&d, &d).unwrap();
    assert!(j.abs() < 1e-12, "W₁(d,d) = {j} must be 0");
}

#[test]
fn wasserstein_equals_normalized_transport_distance() {
    // For two unit impulses at samples a and b, the 1-Wasserstein distance via
    // the CDF-L1 form (with the implementation's 1/n normalization) is |a−b|/n.
    let n = 100;
    let m = MisfitFunction::new(MisfitType::Wasserstein);
    let obs = impulse_trace(n, 30);
    let syn = impulse_trace(n, 50);
    let j = m.compute(&obs, &syn).unwrap();
    let expected = 20.0 / n as f64;
    assert!(
        (j - expected).abs() < 1e-12,
        "W₁ = {j}, expected {expected}"
    );
}

#[test]
fn wasserstein_is_monotone_in_shift() {
    // Convexity surrogate: larger time shift ⇒ strictly larger Wasserstein misfit
    // (the property that breaks cycle-skipping).
    let n = 128;
    let m = MisfitFunction::new(MisfitType::Wasserstein);
    let obs = impulse_trace(n, 40);
    let near = m.compute(&obs, &impulse_trace(n, 45)).unwrap();
    let far = m.compute(&obs, &impulse_trace(n, 70)).unwrap();
    assert!(
        far > near,
        "W₁ must grow with shift: far {far} vs near {near}"
    );
}

// ── Normalized cross-correlation ──────────────────────────────────────────────

#[test]
fn correlation_misfit_is_zero_for_identical_data() {
    let m = MisfitFunction::new(MisfitType::Correlation);
    let d = Array2::from_shape_fn((1, 32), |(_, j)| (j as f64 * 0.2).sin());
    let j = m.compute(&d, &d).unwrap();
    assert!(j.abs() < 1e-12, "1 − C(d,d) = {j} must be 0");
}

#[test]
fn correlation_misfit_is_one_for_orthogonal_data() {
    // sin and cos over an integer number of periods are orthogonal ⇒ C = 0.
    let n = 200;
    let m = MisfitFunction::new(MisfitType::Correlation);
    let obs = Array2::from_shape_fn((1, n), |(_, j)| {
        (2.0 * std::f64::consts::PI * 4.0 * j as f64 / n as f64).sin()
    });
    let syn = Array2::from_shape_fn((1, n), |(_, j)| {
        (2.0 * std::f64::consts::PI * 4.0 * j as f64 / n as f64).cos()
    });
    let j = m.compute(&obs, &syn).unwrap();
    assert!((j - 1.0).abs() < 1e-2, "orthogonal ⇒ 1 − C ≈ 1, got {j}");
}

#[test]
fn correlation_misfit_is_two_for_anticorrelated_data() {
    let m = MisfitFunction::new(MisfitType::Correlation);
    let obs = Array2::from_shape_fn((1, 50), |(_, j)| (j as f64 * 0.3).sin());
    let syn = obs.mapv(|x| -x);
    let j = m.compute(&obs, &syn).unwrap();
    assert!(
        (j - 2.0).abs() < 1e-10,
        "anti-correlated ⇒ 1 − (−1) = 2, got {j}"
    );
}

#[test]
fn correlation_adjoint_is_orthogonal_to_synthetic() {
    // ∂J/∂d_syn is orthogonal to d_syn (the radial direction carries no
    // correlation change), a defining property of the normalized correlation
    // gradient.
    let m = MisfitFunction::new(MisfitType::Correlation);
    let obs = Array2::from_shape_fn((1, 40), |(_, j)| (j as f64 * 0.25).sin());
    let syn = Array2::from_shape_fn((1, 40), |(_, j)| (j as f64 * 0.25 + 0.4).sin());
    let adj = m.compute_adjoint_source(&obs, &syn).unwrap();
    let dot: f64 = adj.row(0).dot(&syn.row(0));
    assert!(dot.abs() < 1e-9, "adjoint·syn = {dot} must vanish");
}

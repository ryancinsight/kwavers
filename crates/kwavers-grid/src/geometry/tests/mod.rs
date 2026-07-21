//! Geometry contract tests.

mod contract;
mod sampling;

fn assert_within_absolute_error(actual: f64, expected: f64, bound: f64) {
    let error = (actual - expected).abs();
    assert!(
        error <= bound,
        "absolute error {error:e} exceeds derived bound {bound:e}: actual={actual:e}, expected={expected:e}"
    );
}

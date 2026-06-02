//! Small vector kernels used by speed-shift solvers.

pub(super) fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(av, bv)| av * bv).sum()
}

pub(super) fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yv, xv) in y.iter_mut().zip(x.iter()) {
        *yv += alpha * *xv;
    }
}

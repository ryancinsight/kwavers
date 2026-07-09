use eunomia::Complex64;
use kwavers_core::utils::iterators::apply_inplace;
use leto::Array2;

pub(super) fn shrinkage_to_identity_real(covariance: &Array2<f64>, alpha: f64) -> Array2<f64> {
    let m = covariance.nrows().max(1);
    let mut out = covariance.clone();

    let mut trace = 0.0;
    for i in 0..m {
        trace += covariance[(i, i)];
    }
    let mu = trace / (m as f64);

    apply_inplace(&mut out, |v| (1.0 - alpha) * v);
    for i in 0..m {
        out[(i, i)] += alpha * mu;
    }
    out
}

pub(super) fn shrinkage_to_identity_complex(
    covariance: &Array2<Complex64>,
    alpha: f64,
) -> Array2<Complex64> {
    let m = covariance.nrows().max(1);
    let mut out = covariance.clone();

    let mut trace_re = 0.0;
    for i in 0..m {
        trace_re += covariance[(i, i)].re;
    }
    let mu = trace_re / (m as f64);

    apply_inplace(&mut out, |v| v * (1.0 - alpha));
    for i in 0..m {
        out[(i, i)] += Complex64::new(alpha * mu, 0.0);
    }
    out
}

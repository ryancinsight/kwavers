/// Solve a tridiagonal linear system using the Thomas algorithm.
///
/// The coefficient slices define:
/// `lower[i] * x[i - 1] + diag[i] * x[i] + upper[i] * x[i + 1] = rhs[i]`.
/// Returns `None` when a pivot is numerically singular.
pub(super) fn thomas_solve(
    lower: &[f64],
    diag: &[f64],
    upper: &[f64],
    rhs: &[f64],
) -> Option<Vec<f64>> {
    let n = diag.len();
    let mut c_prime = vec![0.0_f64; n];
    let mut d_prime = vec![0.0_f64; n];
    let mut x = vec![0.0_f64; n];

    if diag[0].abs() < 1e-300 {
        return None;
    }
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let pivot = diag[i] - lower[i] * c_prime[i - 1];
        if pivot.abs() < 1e-300 {
            return None;
        }
        c_prime[i] = upper[i] / pivot;
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / pivot;
    }

    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Some(x)
}

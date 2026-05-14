//! Shared Westervelt finite-difference stencil kernels.

pub(super) fn index(x: usize, y: usize, z: usize, n: usize) -> usize {
    (x * n + y) * n + z
}

pub(super) fn laplacian(field: &[f64], x: usize, y: usize, z: usize, n: usize, dx: f64) -> f64 {
    let i = index(x, y, z, n);
    let n2 = n * n;
    (field[i - n2] + field[i + n2] + field[i - n] + field[i + n] + field[i - 1] + field[i + 1]
        - 6.0 * field[i])
        / (dx * dx)
}

pub(super) fn nonlinear_term(
    curr: &[f64],
    prev: &[f64],
    older: &[f64],
    i: usize,
    dt: f64,
    step: usize,
) -> f64 {
    let dp = (curr[i] - prev[i]) / dt;
    if step >= 2 {
        let d2 = (curr[i] - 2.0 * prev[i] + older[i]) / (dt * dt);
        2.0 * curr[i] * d2 + 2.0 * dp * dp
    } else {
        2.0 * dp * dp
    }
}

pub(super) fn sponge(n: usize) -> Vec<f64> {
    let layer = (n / 8).max(2);
    let mut values = vec![1.0; n * n * n];
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let edge = x.min(y).min(z).min(n - 1 - x).min(n - 1 - y).min(n - 1 - z);
                if edge < layer {
                    let ratio = (layer - edge) as f64 / layer as f64;
                    values[index(x, y, z, n)] = (1.0 - 0.18 * ratio * ratio).max(0.0);
                }
            }
        }
    }
    values
}

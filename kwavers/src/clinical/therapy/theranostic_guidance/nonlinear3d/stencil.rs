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

#[derive(Clone, Copy, Debug)]
pub(super) struct WesterveltCellTerms {
    pub pressure_to_bulk_modulus: f64,
    pub denominator: f64,
    pub numerator: f64,
    pub pressure_increment: f64,
}

/// Finite-amplitude Westervelt cell terms.
///
/// Starting from
/// `(1 - 2 βp/(ρc²)) p_tt = c²∇²p + 2 β/(ρc²) p_t²`, the leapfrog update is
/// `p[n+1] = 2p[n] - p[n-1] + numerator / denominator`.
/// Solving the pressure-dependent inertia term in the denominator prevents the
/// explicit `p * p_tt` feedback loop from creating nonphysical runaway peaks at
/// histotripsy drive while reducing exactly to the linear acoustic update when
/// `β = 0`.
pub(super) fn westervelt_cell_terms(
    center: f64,
    previous: f64,
    laplacian: f64,
    speed: f64,
    density: f64,
    beta: f64,
    dt: f64,
) -> WesterveltCellTerms {
    let c2 = speed * speed;
    let inv_bulk = 1.0 / (density * c2).max(1.0e-18);
    let pressure_to_bulk_modulus = beta * inv_bulk;
    let pressure_increment = center - previous;
    let denominator = 1.0 - 2.0 * pressure_to_bulk_modulus * center;
    let numerator =
        c2 * dt * dt * laplacian + 2.0 * pressure_to_bulk_modulus * pressure_increment.powi(2);
    WesterveltCellTerms {
        pressure_to_bulk_modulus,
        denominator,
        numerator,
        pressure_increment,
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

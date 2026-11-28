use leto::Array2;
use std::f64::consts::PI;

/// Row `i` of a row-major history grid as a slice.
pub(super) fn history_row(grid: &Array2<f64>, i: usize) -> &[f64] {
    let [_, width] = grid.shape();
    let data = grid
        .as_slice()
        .expect("invariant: history grids are built by `Array2::zeros` in standard layout");
    &data[i * width..(i + 1) * width]
}

/// Pair geometry: coupling weights `W[[i, j]] = ρ/(4π d_ij)`
/// (diagonal 0) and retardation delays `d_ij / c` [s].
#[must_use]
pub fn pair_geometry(
    positions: &[[f64; 3]],
    density_kg_m3: f64,
    propagation_sound_speed_m_s: f64,
) -> (Array2<f64>, Array2<f64>) {
    let n = positions.len();
    let mut weights = Array2::zeros([n, n]);
    let mut delays = Array2::zeros([n, n]);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = positions[i][0] - positions[j][0];
            let dy = positions[i][1] - positions[j][1];
            let dz = positions[i][2] - positions[j][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d > 0.0 {
                weights[[i, j]] = density_kg_m3 / (4.0 * PI * d);
                delays[[i, j]] = d / propagation_sound_speed_m_s;
            }
        }
    }
    (weights, delays)
}

/// Linear interpolation of one history row at time `tq` (zero
/// outside the history), the Python reference's `interp_rows`.
fn interp_row(row: &[f64], dt: f64, tq: f64) -> f64 {
    if row.len() < 2 || tq < 0.0 {
        return 0.0;
    }
    let x = tq / dt;
    let i0 = x.floor();
    if i0 < 0.0 || i0 as usize >= row.len() - 1 {
        return 0.0;
    }
    let i0 = i0 as usize;
    let frac = x - i0 as f64;
    row[i0] * (1.0 - frac) + row[i0 + 1] * frac
}

/// Coupling field on the grid: `coup[[i, k]] = Σ_j W[[i, j]] ·
/// V̈_j(t_k − delay_ij)`.
#[must_use]
pub fn coupling_field(
    vddot_prev: &Array2<f64>,
    weights: &Array2<f64>,
    delays: &Array2<f64>,
    dt: f64,
) -> Array2<f64> {
    let [n, _] = weights.shape();
    let nsteps = if n == 0 { 0 } else { vddot_prev.shape()[1] };
    let mut coup = Array2::zeros([n, nsteps]);
    for i in 0..n {
        for k in 0..nsteps {
            let t = k as f64 * dt;
            let mut sum = 0.0;
            for j in 0..n {
                if weights[[i, j]] != 0.0 {
                    sum += weights[[i, j]]
                        * interp_row(history_row(vddot_prev, j), dt, t - delays[[i, j]]);
                }
            }
            coup[[i, k]] = sum;
        }
    }
    coup
}

/// Centred-difference derivative of each row on the grid
/// (one-sided at the ends) — `np.gradient` semantics, as in the
/// Python reference.
pub(super) fn grid_gradient(grid: &Array2<f64>, dt: f64) -> Array2<f64> {
    let [rows, n] = grid.shape();
    let mut out = Array2::zeros([rows, n]);
    if n < 2 {
        return out;
    }
    for r in 0..rows {
        out[[r, 0]] = (grid[[r, 1]] - grid[[r, 0]]) / dt;
        out[[r, n - 1]] = (grid[[r, n - 1]] - grid[[r, n - 2]]) / dt;
        for k in 1..n - 1 {
            out[[r, k]] = (grid[[r, k + 1]] - grid[[r, k - 1]]) / (2.0 * dt);
        }
    }
    out
}

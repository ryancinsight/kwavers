//! Direct inversion — Gauss-Seidel optimization of `J(k²) = ‖∇²u + k²u‖² + λ‖∇k²‖²`.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_imaging::ultrasound::elastography::ElasticityMap;
use kwavers_physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;
use leto::Array3 as LetoArray3;
use ndarray::Array3;

use super::super::algorithms::{fill_boundaries, spatial_smoothing};
use super::super::types::elasticity_map_from_speed;
use kwavers_core::constants::numerical::TWO_PI;

/// Direct inversion (most accurate method)
///
/// Solves inverse problem directly from wave equation using iterative optimization
/// to minimize the residual error.
///
/// # Theory
///
/// Minimizes the functional `J(k²) = ‖∇²u + k²u‖² + λ‖∇(k²)‖²`
/// where `k = ω/cs` is the wavenumber.
///
/// The optimization is solved using a Gauss-Seidel iterative scheme:
/// `θ_i = (λ Σ θ_j − u_i ∇²u_i) / (u_i² + 6λ)`
/// where `θ = k²`.
///
/// # References
///
/// - McLaughlin & Renzi (2006): "Direct inversion methods"
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub(super) fn direct_inversion(
    displacement: &DisplacementField,
    grid: &Grid,
    density: f64,
    frequency: f64,
) -> KwaversResult<ElasticityMap> {
    let (nx, ny, nz) = displacement.uz.dim();
    let mut shear_wave_speed = LetoArray3::zeros([nx, ny, nz]);

    // 1. Compute Laplacian of displacement field
    let laplacian = compute_laplacian(&displacement.uz, grid);

    // 2. Initialize wavenumber squared (theta = k^2)
    let omega = TWO_PI * frequency;
    let initial_k = omega / 3.0;
    let initial_theta = initial_k * initial_k;

    let mut theta = Array3::from_elem((nx, ny, nz), initial_theta);

    // 3. Optimization parameters
    let max_iterations = 50;
    // Regularization parameter lambda — scaled by mean squared displacement
    // so the data term `(u·θ)²` and the smoothing term `λ·(∇θ)²` balance.
    let mean_sq_disp = displacement.uz.iter().map(|x| x * x).sum::<f64>() / (nx * ny * nz) as f64;
    let lambda = mean_sq_disp.max(1e-18) * 1.0;

    // 4. Iterative Optimization (Gauss-Seidel)
    for _ in 0..max_iterations {
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let u_val = displacement.uz[[i, j, k]];
                    let lap_val = laplacian[[i, j, k]];

                    let sum_theta = theta[[i + 1, j, k]]
                        + theta[[i - 1, j, k]]
                        + theta[[i, j + 1, k]]
                        + theta[[i, j - 1, k]]
                        + theta[[i, j, k + 1]]
                        + theta[[i, j, k - 1]];

                    let numerator = lambda * sum_theta - u_val * lap_val;
                    let denominator = u_val.mul_add(u_val, 6.0 * lambda);

                    theta[[i, j, k]] = numerator / denominator;
                }
            }
        }
    }

    // 5. Convert back to speed
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let k_squared = theta[[i, j, k]];

                // Clamp k^2 to correspond to speed range [0.5, 20.0] m/s
                let w2 = omega * omega;
                let k2_max = w2 / (0.5 * 0.5);
                let k2_min = w2 / (20.0 * 20.0);

                let valid_k2 = k_squared.clamp(k2_min, k2_max);

                let cs = omega / valid_k2.sqrt();
                shear_wave_speed[[i, j, k]] = cs;
            }
        }
    }

    spatial_smoothing(&mut shear_wave_speed);
    fill_boundaries(&mut shear_wave_speed);

    Ok(elasticity_map_from_speed(shear_wave_speed, density))
}

/// Compute Laplacian of a scalar field using 7-point stencil
fn compute_laplacian(field: &Array3<f64>, grid: &Grid) -> Array3<f64> {
    let (nx, ny, nz) = field.dim();
    let mut laplacian = Array3::zeros((nx, ny, nz));

    let idx2 = 1.0 / (grid.dx * grid.dx);
    let idy2 = 1.0 / (grid.dy * grid.dy);
    let idz2 = 1.0 / (grid.dz * grid.dz);

    // 2-D plane-strain input (nz < 3) has no z-curvature, so ∂²u/∂z² = 0 and the
    // in-plane Laplacian carries the inversion. Without this guard the 3-D
    // interior loop is empty for nz = 1, leaving a zero Laplacian and a degenerate
    // μ = -ρω²u/∇²u (silent garbage on a 2-D imaging plane).
    let has_z_interior = nz >= 3;
    let (k_lo, k_hi) = if has_z_interior { (1, nz - 1) } else { (0, nz) };
    for i in 1..nx.saturating_sub(1) {
        for j in 1..ny.saturating_sub(1) {
            for k in k_lo..k_hi {
                let center = field[[i, j, k]];

                let d2x =
                    (2.0f64.mul_add(-center, field[[i + 1, j, k]]) + field[[i - 1, j, k]]) * idx2;
                let d2y =
                    (2.0f64.mul_add(-center, field[[i, j + 1, k]]) + field[[i, j - 1, k]]) * idy2;
                let d2z = if has_z_interior {
                    (2.0f64.mul_add(-center, field[[i, j, k + 1]]) + field[[i, j, k - 1]]) * idz2
                } else {
                    0.0
                };

                laplacian[[i, j, k]] = d2x + d2y + d2z;
            }
        }
    }

    laplacian
}

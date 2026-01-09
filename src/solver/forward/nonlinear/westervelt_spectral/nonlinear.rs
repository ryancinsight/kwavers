//! Nonlinear term computation for Westervelt equation

use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::{Array3, ArrayView3, Zip};

/// Compute the nonlinear term for the Westervelt equation
///
/// The Westervelt nonlinear term is: (β/ρc⁴) * ∂²(p²)/∂t²
/// Where ∂²(p²)/∂t² = 2p * ∂²p/∂t² + 2(∂p/∂t)²
pub fn compute_nonlinear_term(
    pressure: &Array3<f64>,
    prev_pressure: &Array3<f64>,
    pressure_history: Option<&Array3<f64>>,
    medium: &dyn Medium,
    grid: &Grid,
    dt: f64,
) -> Array3<f64> {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let mut nonlinear_term = Array3::<f64>::zeros((nx, ny, nz));

    // Get spatially varying medium properties
    let rho_arr = medium.density_array();
    let c_arr = medium.sound_speed_array();

    Zip::indexed(&mut nonlinear_term)
        .and(pressure)
        .and(prev_pressure)
        .for_each(|(i, j, k), nl_val, &p_curr, &p_prev| {
            if i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1 {
                let rho = rho_arr[[i, j, k]].max(1e-9);
                let c = c_arr[[i, j, k]].max(1e-9);

                // Get spatially-varying nonlinearity coefficient
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let beta = crate::domain::medium::AcousticProperties::nonlinearity_coefficient(
                    medium, x, y, z, grid,
                );

                // Calculate nonlinear coefficient (negative for Westervelt equation)
                // Westervelt: ∇²p - (1/c²)∂²p/∂t² = - (β/ρc⁴)∂²(p²)/∂t² - δ∇²(∂p/∂t) - Q
                let nonlinear_coeff = -beta / (rho * c.powi(4));

                let term = if let Some(p_history) = pressure_history {
                    // Full second-order accuracy with pressure history
                    let p_prev_prev = p_history[[i, j, k]];
                    let d2p_dt2 = (p_curr - 2.0 * p_prev + p_prev_prev) / (dt * dt);
                    let dp_dt = (p_curr - p_prev) / dt;

                    // Westervelt nonlinear term
                    let p_squared_second_deriv = 2.0 * p_curr * d2p_dt2 + 2.0 * dp_dt.powi(2);
                    nonlinear_coeff * p_squared_second_deriv
                } else {
                    // Bootstrap for first iteration
                    let dp_dt = (p_curr - p_prev) / dt.max(1e-12);
                    let d2p_dt2_bootstrap = dp_dt / dt.max(1e-12);

                    let p_squared_second_deriv =
                        2.0 * p_curr * d2p_dt2_bootstrap + 2.0 * dp_dt.powi(2);
                    nonlinear_coeff * p_squared_second_deriv
                };

                *nl_val = term;
            } else {
                *nl_val = 0.0;
            }
        });

    nonlinear_term
}

/// Compute viscoelastic damping term
///
/// Implements the viscoelastic damping: (4μ/3 + `μ_B`) * ∇²(∂p/∂t)
pub fn compute_viscoelastic_term(
    pressure: &Array3<f64>,
    prev_pressure: &Array3<f64>,
    eta_s_arr: &Array3<f64>, // Shear viscosity
    eta_b_arr: &Array3<f64>, // Bulk viscosity
    rho_arr: &ArrayView3<f64>,
    grid: &Grid,
    dt: f64,
) -> Array3<f64> {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let mut damping_term = Array3::<f64>::zeros((nx, ny, nz));

    let dx2_inv = 1.0 / (grid.dx * grid.dx);
    let dy2_inv = 1.0 / (grid.dy * grid.dy);
    let dz2_inv = 1.0 / (grid.dz * grid.dz);

    // Compute ∂p/∂t
    let dp_dt = (pressure - prev_pressure) / dt;

    // Apply Laplacian to ∂p/∂t with viscosity coefficients
    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let laplacian_dp_dt = (dp_dt[[i + 1, j, k]] - 2.0 * dp_dt[[i, j, k]]
                    + dp_dt[[i - 1, j, k]])
                    * dx2_inv
                    + (dp_dt[[i, j + 1, k]] - 2.0 * dp_dt[[i, j, k]] + dp_dt[[i, j - 1, k]])
                        * dy2_inv
                    + (dp_dt[[i, j, k + 1]] - 2.0 * dp_dt[[i, j, k]] + dp_dt[[i, j, k - 1]])
                        * dz2_inv;

                let eta_s = eta_s_arr[[i, j, k]];
                let eta_b = eta_b_arr[[i, j, k]];
                let rho = rho_arr[[i, j, k]].max(1e-9);

                // Viscoelastic damping coefficient
                let visc_coeff = (4.0 * eta_s / 3.0 + eta_b) / rho;

                damping_term[[i, j, k]] = visc_coeff * laplacian_dp_dt;
            }
        }
    }

    damping_term
}

//! Kuznetsov equation solver for nonlinear acoustics with diffusion

use crate::{error::KwaversResult, grid::Grid, medium::Medium};
use ndarray::{Array3, Zip};

use super::config::AcousticSolverConfig;
use super::solver::AcousticSolver;

/// Kuznetsov equation solver
pub struct KuznetsovSolver {
    config: AcousticSolverConfig,
    grid: Grid,
    velocity_potential: Array3<f64>,
}

impl KuznetsovSolver {
    /// Create a new Kuznetsov solver
    pub fn new(config: AcousticSolverConfig, grid: Grid) -> KwaversResult<Self> {
        let velocity_potential = Array3::zeros((grid.nx, grid.ny, grid.nz));

        Ok(Self {
            config,
            grid,
            velocity_potential,
        })
    }
}

impl AcousticSolver for KuznetsovSolver {
    fn update(
        &mut self,
        pressure: &mut Array3<f64>,
        medium: &dyn Medium,
        source_term: &Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        // Kuznetsov equation:
        // ∂²φ/∂t² - c²∇²φ = -β/(2ρc⁴) * ∂²(∂φ/∂t)²/∂t² + δ∇²(∂φ/∂t)
        // where φ is velocity potential, p = ρ₀ ∂φ/∂t

        // Add source to velocity potential
        self.velocity_potential = &self.velocity_potential + source_term;

        // Compute Laplacian of velocity potential
        let laplacian = compute_laplacian(&self.velocity_potential, grid);

        // Linear wave propagation term
        let mut linear_term = Array3::zeros(pressure.dim());
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    let c = medium.sound_speed(x, y, z, grid);
                    linear_term[[i, j, k]] = c * c * laplacian[[i, j, k]];
                }
            }
        }

        // Nonlinear term (simplified for now)
        let beta = self.config.nonlinearity_scaling;
        let nonlinear_term =
            compute_nonlinear_term(pressure, &self.velocity_potential, medium, grid, beta);

        // Diffusion term (thermoviscous losses)
        let diffusion_term = compute_diffusion_term(pressure, medium, grid);

        // Update velocity potential
        self.velocity_potential =
            &self.velocity_potential + &(dt * (&linear_term + &nonlinear_term + &diffusion_term));

        // Convert velocity potential to pressure: p = ρ₀ ∂φ/∂t
        // Using finite difference approximation
        Zip::from(pressure)
            .and(&self.velocity_potential)
            .for_each(|p, &phi| {
                *p = phi / dt; // Simplified - should track previous potential
            });

        Ok(())
    }

    fn name(&self) -> &str {
        "Kuznetsov Nonlinear Acoustic Solver with Diffusion"
    }

    fn check_stability(&self, dt: f64, grid: &Grid, max_sound_speed: f64) -> KwaversResult<()> {
        // Kuznetsov equation with diffusion has different stability requirements
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let cfl = max_sound_speed * dt / dx_min;
        let max_cfl = self.config.cfl_safety_factor * 0.7; // More restrictive due to diffusion

        if cfl > max_cfl {
            return Err(crate::error::ValidationError::OutOfRange {
                field: "CFL (Kuznetsov)".to_string(),
                value: cfl.to_string(),
                min: "0".to_string(),
                max: max_cfl.to_string(),
            }
            .into());
        }

        Ok(())
    }
}

/// Compute Laplacian using central differences
fn compute_laplacian(field: &Array3<f64>, grid: &Grid) -> Array3<f64> {
    let (nx, ny, nz) = field.dim();
    let mut laplacian = Array3::zeros((nx, ny, nz));

    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                laplacian[[i, j, k]] = (field[[i + 1, j, k]] - 2.0 * field[[i, j, k]]
                    + field[[i - 1, j, k]])
                    / (grid.dx * grid.dx)
                    + (field[[i, j + 1, k]] - 2.0 * field[[i, j, k]] + field[[i, j - 1, k]])
                        / (grid.dy * grid.dy)
                    + (field[[i, j, k + 1]] - 2.0 * field[[i, j, k]] + field[[i, j, k - 1]])
                        / (grid.dz * grid.dz);
            }
        }
    }

    laplacian
}

/// Compute nonlinear term for Kuznetsov equation
fn compute_nonlinear_term(
    pressure: &Array3<f64>,
    velocity_potential: &Array3<f64>,
    medium: &dyn Medium,
    grid: &Grid,
    beta: f64,
) -> Array3<f64> {
    let mut nonlinear = Array3::zeros(pressure.dim());

    // Simplified nonlinear term computation
    // Full implementation would compute ∂²(∂φ/∂t)²/∂t²
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = medium.density(x, y, z, grid);
                let c = medium.sound_speed(x, y, z, grid);

                // Simplified: use pressure squared as proxy
                nonlinear[[i, j, k]] =
                    -beta / (2.0 * rho * c.powi(4)) * pressure[[i, j, k]].powi(2);
            }
        }
    }

    nonlinear
}

/// Compute diffusion term for thermoviscous losses
fn compute_diffusion_term(pressure: &Array3<f64>, medium: &dyn Medium, grid: &Grid) -> Array3<f64> {
    // Diffusivity of sound: δ = (4μ/3 + μ_B)/(ρ₀c₀²)
    // where μ is shear viscosity, μ_B is bulk viscosity

    let laplacian_p = compute_laplacian(pressure, grid);
    let mut diffusion = Array3::zeros(pressure.dim());

    // Apply diffusion coefficient
    // Note: This requires viscosity properties from medium
    // Using simplified constant diffusivity for now
    const DIFFUSIVITY: f64 = 1e-6; // m²/s (typical for water)

    diffusion = laplacian_p * DIFFUSIVITY;

    diffusion
}

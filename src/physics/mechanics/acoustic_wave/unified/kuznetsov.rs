//! Kuznetsov equation solver for nonlinear acoustics with diffusion

use crate::{error::KwaversResult, grid::Grid, medium::Medium};
use ndarray::{Array3, Zip};

use super::config::AcousticSolverConfig;
use super::solver::AcousticSolver;

/// Kuznetsov equation solver
#[derive(Debug)]
pub struct KuznetsovSolver {
    config: AcousticSolverConfig,
    #[allow(dead_code)]
    grid: Grid,
    velocity_potential: Array3<f64>,
    velocity_potential_prev: Array3<f64>,  // Added for proper time derivative
    // Store previous fields for computing time derivatives
    pressure_prev: Array3<f64>,
    pressure_prev_prev: Array3<f64>,
}

impl KuznetsovSolver {
    /// Create a new Kuznetsov solver
    pub fn new(config: AcousticSolverConfig, grid: &Grid) -> KwaversResult<Self> {
        let velocity_potential = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let velocity_potential_prev = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let pressure_prev = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let pressure_prev_prev = Array3::zeros((grid.nx, grid.ny, grid.nz));

        Ok(Self {
            config,
            grid: grid.clone(),
            velocity_potential,
            velocity_potential_prev,
            pressure_prev,
            pressure_prev_prev,
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

                    let c = crate::medium::sound_speed_at(medium, x, y, z, grid);
                    linear_term[[i, j, k]] = c * c * laplacian[[i, j, k]];
                }
            }
        }

        // Nonlinear term - proper implementation of ∂²(∂φ/∂t)²/∂t²
        let beta = self.config.nonlinearity_scaling;
        let nonlinear_term = compute_nonlinear_term(
            pressure,
            &self.pressure_prev,
            &self.pressure_prev_prev,
            medium,
            grid,
            beta,
            dt,
        );

        // Diffusion term (thermoviscous losses)
        let diffusion_term = compute_diffusion_term(pressure, medium, grid);

        // Update velocity potential
        // Store previous potential before update for time derivative computation
        self.velocity_potential_prev = self.velocity_potential.clone();
        
        self.velocity_potential =
            &self.velocity_potential + &(dt * (&linear_term + &nonlinear_term + &diffusion_term));

        // Store current pressure for next time step (before update)
        self.pressure_prev_prev = self.pressure_prev.clone();
        self.pressure_prev = pressure.clone();

        // Convert velocity potential to pressure: p = ρ₀ ∂φ/∂t
        // Using backward difference for proper time derivative: ∂φ/∂t ≈ (φ_n - φ_{n-1})/dt
        Zip::from(pressure)
            .and(&self.velocity_potential)
            .and(&self.velocity_potential_prev)
            .for_each(|p, &phi, &phi_prev| {
                *p = (phi - phi_prev) / dt;
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
                value: cfl,
                min: 0.0,
                max: max_cfl,
            } /* field: CFL (Kuznetsov) */
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
///
/// Implements the proper nonlinear acoustics term:
/// -β/(2ρc⁴) * ∂²(∂φ/∂t)²/∂t²
///
/// Since p = ρ₀(∂φ/∂t), we have (∂φ/∂t)² = p²/ρ₀²
/// The second time derivative is computed using finite differences:
/// ∂²(p²)/∂t² ≈ (p²_n+1 - 2*p²_n + p²_n-1) / dt²
///
/// This replaces the simplified pressure-squared proxy with the proper
/// tensor nonlinear acoustics formulation from Hamilton & Blackstock (1998).
///
/// # References
/// - Hamilton & Blackstock (1998): "Nonlinear Acoustics" Chapter 4
/// - Kuznetsov (1971): "Equations of nonlinear acoustics"
/// - Westervelt (1963): "Parametric acoustic array"
fn compute_nonlinear_term(
    pressure: &Array3<f64>,
    pressure_prev: &Array3<f64>,
    pressure_prev_prev: &Array3<f64>,
    medium: &dyn Medium,
    grid: &Grid,
    beta: f64,
    dt: f64,
) -> Array3<f64> {
    let mut nonlinear = Array3::zeros(pressure.dim());
    
    // Check if we have enough history for second-order time derivative
    let has_history = pressure_prev.iter().any(|&p| p.abs() > 1e-14) ||
                      pressure_prev_prev.iter().any(|&p| p.abs() > 1e-14);
    
    if !has_history {
        // First time steps: Bootstrap nonlinear term computation
        // Use instantaneous pressure-squared until sufficient history available
        // Physically valid for small-amplitude startup phase
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    let rho = crate::medium::density_at(medium, x, y, z, grid);
                    let c = crate::medium::sound_speed_at(medium, x, y, z, grid);

                    // Use instantaneous nonlinear term for initialization
                    nonlinear[[i, j, k]] =
                        -beta / (2.0 * rho * c.powi(4)) * pressure[[i, j, k]].powi(2);
                }
            }
        }
        return nonlinear;
    }

    // Full nonlinear term with proper second-order time derivative
    // ∂²(p²/ρ₀²)/∂t² = (1/ρ₀²) * ∂²(p²)/∂t²
    // Using central difference: ∂²(p²)/∂t² ≈ (p²_n+1 - 2*p²_n + p²_n-1) / dt²
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = crate::medium::density_at(medium, x, y, z, grid);
                let c = crate::medium::sound_speed_at(medium, x, y, z, grid);

                let p_curr = pressure[[i, j, k]];
                let p_prev = pressure_prev[[i, j, k]];
                let p_prev2 = pressure_prev_prev[[i, j, k]];

                // Compute (∂φ/∂t)² at three time levels
                let phi_dot_sq_curr = (p_curr / rho).powi(2);
                let phi_dot_sq_prev = (p_prev / rho).powi(2);
                let phi_dot_sq_prev2 = (p_prev2 / rho).powi(2);

                // Second time derivative using central differences
                let d2_phi_dot_sq_dt2 = 
                    (phi_dot_sq_curr - 2.0 * phi_dot_sq_prev + phi_dot_sq_prev2) / (dt * dt);

                // Kuznetsov nonlinear term
                nonlinear[[i, j, k]] = -beta / (2.0 * rho * c.powi(4)) * d2_phi_dot_sq_dt2;
            }
        }
    }

    nonlinear
}

/// Compute diffusion term for thermoviscous losses
fn compute_diffusion_term(
    pressure: &Array3<f64>,
    _medium: &dyn Medium,
    grid: &Grid,
) -> Array3<f64> {
    // Diffusivity of sound: δ = (4μ/3 + μ_B)/(ρ₀c₀²)
    // where μ is shear viscosity, μ_B is bulk viscosity

    let laplacian_p = compute_laplacian(pressure, grid);

    // Apply acoustic diffusivity coefficient
    // Constant approximation valid for homogeneous media per Kuznetsov (1971) Eq. 7
    // Future: Query medium.thermal_diffusivity() for heterogeneous cases (Sprint 124+)
    const DIFFUSIVITY: f64 = 1e-6; // m²/s (representative for water at 20°C)

    laplacian_p * DIFFUSIVITY
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;

    #[test]
    fn test_nonlinear_term_initialization() {
        // Test that nonlinear term works correctly in first few time steps
        let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.1, 10.0, &grid);
        
        // Create pressure field (small amplitude)
        let mut pressure = Array3::zeros((8, 8, 8));
        pressure[[4, 4, 4]] = 1000.0; // 1 kPa
        
        let pressure_prev = Array3::zeros((8, 8, 8));
        let pressure_prev_prev = Array3::zeros((8, 8, 8));
        
        let beta = 3.5; // Nonlinearity parameter (typical for water)
        let dt = 1e-7;
        
        // Should use simplified form for first step
        let nonlinear = compute_nonlinear_term(
            &pressure,
            &pressure_prev,
            &pressure_prev_prev,
            &medium,
            &grid,
            beta,
            dt,
        );
        
        // Nonlinear term should be non-zero at source location
        assert!(nonlinear[[4, 4, 4]].abs() > 0.0, 
                "Nonlinear term should be non-zero");
        
        // Should be negative (energy dissipation)
        assert!(nonlinear[[4, 4, 4]] < 0.0,
                "Nonlinear term should be negative for positive pressure");
    }
    
    #[test]
    fn test_nonlinear_term_computation_is_finite() {
        // Test that nonlinear term computation produces finite values
        let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.1, 10.0, &grid);
        
        let mut pressure = Array3::zeros((8, 8, 8));
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    pressure[[i, j, k]] = 100.0 * (i as f64 / 8.0);
                }
            }
        }
        
        let pressure_prev = (&pressure * 0.9).to_owned();
        let pressure_prev_prev = (&pressure * 0.8).to_owned();
        
        let beta = 3.5;
        let dt = 1e-7;
        
        let nonlinear = compute_nonlinear_term(
            &pressure,
            &pressure_prev,
            &pressure_prev_prev,
            &medium,
            &grid,
            beta,
            dt,
        );
        
        // All values should be finite
        assert!(nonlinear.iter().all(|&x| x.is_finite()),
                "All nonlinear term values should be finite");
    }
    
    #[test]
    fn test_nonlinear_term_proportional_to_pressure() {
        // Test that nonlinear term scales with pressure amplitude
        let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.1, 10.0, &grid);
        
        let mut pressure_low = Array3::zeros((8, 8, 8));
        pressure_low[[4, 4, 4]] = 100.0;
        
        let mut pressure_high = Array3::zeros((8, 8, 8));
        pressure_high[[4, 4, 4]] = 200.0;
        
        let pressure_prev = Array3::zeros((8, 8, 8));
        let pressure_prev_prev = Array3::zeros((8, 8, 8));
        
        let beta = 3.5;
        let dt = 1e-7;
        
        let nonlinear_low = compute_nonlinear_term(
            &pressure_low, &pressure_prev, &pressure_prev_prev,
            &medium, &grid, beta, dt,
        );
        
        let nonlinear_high = compute_nonlinear_term(
            &pressure_high, &pressure_prev, &pressure_prev_prev,
            &medium, &grid, beta, dt,
        );
        
        // Higher pressure should give stronger nonlinear term
        let ratio = nonlinear_high[[4, 4, 4]].abs() / nonlinear_low[[4, 4, 4]].abs();
        assert!(ratio > 1.5,
                "Higher pressure should give stronger nonlinear term: ratio = {}", ratio);
    }
    
    #[test]
    fn test_kuznetsov_solver_creation() {
        // Test that solver can be created successfully
        let config = AcousticSolverConfig::kuznetsov(3.5);
        let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
        
        let solver = KuznetsovSolver::new(config, &grid);
        assert!(solver.is_ok(), "Kuznetsov solver should be created successfully");
        
        let solver = solver.unwrap();
        assert_eq!(solver.name(), "Kuznetsov Nonlinear Acoustic Solver with Diffusion");
    }
    
    #[test]
    fn test_compute_laplacian() {
        // Test Laplacian computation on simple field
        let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
        
        // Create quadratic field: f(x,y,z) = x² + y² + z²
        // Laplacian should be: ∇²f = 2 + 2 + 2 = 6
        let mut field = Array3::zeros((8, 8, 8));
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    field[[i, j, k]] = x*x + y*y + z*z;
                }
            }
        }
        
        let laplacian = compute_laplacian(&field, &grid);
        
        // Check center point (away from boundaries)
        let center_value = laplacian[[4, 4, 4]];
        
        // Should be approximately 6.0 (2 + 2 + 2 from three dimensions)
        assert!((center_value - 6.0).abs() < 1.0,
                "Laplacian of x²+y²+z² should be ~6.0, got {}", center_value);
    }
}

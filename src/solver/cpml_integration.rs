//! C-PML Integration for Acoustic Solvers
//!
//! This module provides integration of Convolutional PML with acoustic wave solvers,
//! ensuring proper handling of memory variables and field updates.

use crate::boundary::cpml::{CPMLBoundary, CPMLConfig};
use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use log::trace;
use ndarray::{Array3, Array4, Axis};

/// C-PML solver integration for acoustic wave propagation
#[derive(Debug)]
pub struct CPMLSolver {
    /// C-PML boundary instance
    cpml: CPMLBoundary,

    /// Temporary arrays for gradient computation
    grad_x: Array3<f64>,
    grad_y: Array3<f64>,
    grad_z: Array3<f64>,

    /// Grid reference for computations
    nx: usize,
    ny: usize,
    nz: usize,
}

impl CPMLSolver {
    /// Create new C-PML solver integration
    ///
    /// # Arguments
    /// * `config` - CPML configuration
    /// * `grid` - Computational grid
    /// * `dt` - Time step from the main solver
    /// * `sound_speed` - Reference sound speed (typically max in medium)
    pub fn new(config: CPMLConfig, grid: &Grid, dt: f64, sound_speed: f64) -> KwaversResult<Self> {
        let cpml = CPMLBoundary::with_cfl(config, grid, dt, sound_speed)?;

        Ok(Self {
            cpml,
            grad_x: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            grad_y: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            grad_z: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            nx: grid.nx,
            ny: grid.ny,
            nz: grid.nz,
        })
    }

    /// Update acoustic field with C-PML
    ///
    /// This method should be called during the acoustic field update step
    /// to properly apply C-PML absorption with memory variables.
    pub fn update_acoustic_field(
        &mut self,
        pressure: &mut Array3<f64>,
        velocity: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        step: usize,
    ) -> KwaversResult<()> {
        trace!("Updating acoustic field with C-PML");

        // Step 1: Compute pressure gradients
        self.compute_pressure_gradients(pressure, grid);

        // Step 2: Update C-PML memory variables
        self.cpml.update_acoustic_memory(&self.grad_x, 0);
        self.cpml.update_acoustic_memory(&self.grad_y, 1);
        self.cpml.update_acoustic_memory(&self.grad_z, 2);

        // Step 3: Apply C-PML to gradients
        self.cpml.apply_cpml_gradient(&mut self.grad_x, 0);
        self.cpml.apply_cpml_gradient(&mut self.grad_y, 1);
        self.cpml.apply_cpml_gradient(&mut self.grad_z, 2);

        // Step 4: Update velocity field with modified gradients
        self.update_velocity_with_cpml(velocity, grid, medium, dt);

        // Step 5: Compute velocity divergence for pressure update
        let div_v = self.compute_velocity_divergence(velocity, grid);

        // Step 6: Update pressure field
        self.update_pressure_with_cpml(pressure, &div_v, grid, medium, dt);

        Ok(())
    }

    /// Compute pressure gradients using appropriate finite differences
    fn compute_pressure_gradients(&mut self, pressure: &Array3<f64>, grid: &Grid) {
        // X-gradient
        for i in 1..self.nx - 1 {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    self.grad_x[[i, j, k]] =
                        (pressure[[i + 1, j, k]] - pressure[[i - 1, j, k]]) / (2.0 * grid.dx);
                }
            }
        }

        // Y-gradient
        for i in 0..self.nx {
            for j in 1..self.ny - 1 {
                for k in 0..self.nz {
                    self.grad_y[[i, j, k]] =
                        (pressure[[i, j + 1, k]] - pressure[[i, j - 1, k]]) / (2.0 * grid.dy);
                }
            }
        }

        // Z-gradient
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 1..self.nz - 1 {
                    self.grad_z[[i, j, k]] =
                        (pressure[[i, j, k + 1]] - pressure[[i, j, k - 1]]) / (2.0 * grid.dz);
                }
            }
        }

        // Handle boundaries with one-sided differences
        self.apply_boundary_gradients(pressure, grid);
    }

    /// Apply one-sided differences at boundaries
    fn apply_boundary_gradients(&mut self, pressure: &Array3<f64>, grid: &Grid) {
        // X-boundaries
        for j in 0..self.ny {
            for k in 0..self.nz {
                // Left boundary
                self.grad_x[[0, j, k]] = (pressure[[1, j, k]] - pressure[[0, j, k]]) / grid.dx;
                // Right boundary
                let i = self.nx - 1;
                self.grad_x[[i, j, k]] = (pressure[[i, j, k]] - pressure[[i - 1, j, k]]) / grid.dx;
            }
        }

        // Y-boundaries
        for i in 0..self.nx {
            for k in 0..self.nz {
                // Bottom boundary
                self.grad_y[[i, 0, k]] = (pressure[[i, 1, k]] - pressure[[i, 0, k]]) / grid.dy;
                // Top boundary
                let j = self.ny - 1;
                self.grad_y[[i, j, k]] = (pressure[[i, j, k]] - pressure[[i, j - 1, k]]) / grid.dy;
            }
        }

        // Z-boundaries
        for i in 0..self.nx {
            for j in 0..self.ny {
                // Front boundary
                self.grad_z[[i, j, 0]] = (pressure[[i, j, 1]] - pressure[[i, j, 0]]) / grid.dz;
                // Back boundary
                let k = self.nz - 1;
                self.grad_z[[i, j, k]] = (pressure[[i, j, k]] - pressure[[i, j, k - 1]]) / grid.dz;
            }
        }
    }

    /// Update velocity field with C-PML modified gradients
    fn update_velocity_with_cpml(
        &self,
        velocity: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) {
        // Update velocity components separately to avoid borrow checker issues
        {
            let mut vx = velocity.index_axis_mut(Axis(0), 0);
            for i in 0..self.nx {
                for j in 0..self.ny {
                    for k in 0..self.nz {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        let rho = crate::medium::density_at(medium, x, y, z, grid);
                        vx[[i, j, k]] -= dt * self.grad_x[[i, j, k]] / rho;
                    }
                }
            }
        }

        {
            let mut vy = velocity.index_axis_mut(Axis(0), 1);
            for i in 0..self.nx {
                for j in 0..self.ny {
                    for k in 0..self.nz {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        let rho = crate::medium::density_at(medium, x, y, z, grid);
                        vy[[i, j, k]] -= dt * self.grad_y[[i, j, k]] / rho;
                    }
                }
            }
        }

        {
            let mut vz = velocity.index_axis_mut(Axis(0), 2);
            for i in 0..self.nx {
                for j in 0..self.ny {
                    for k in 0..self.nz {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        let rho = crate::medium::density_at(medium, x, y, z, grid);
                        vz[[i, j, k]] -= dt * self.grad_z[[i, j, k]] / rho;
                    }
                }
            }
        }
    }

    /// Compute velocity divergence for pressure update
    fn compute_velocity_divergence(&self, velocity: &Array4<f64>, grid: &Grid) -> Array3<f64> {
        let vx = velocity.index_axis(Axis(0), 0);
        let vy = velocity.index_axis(Axis(0), 1);
        let vz = velocity.index_axis(Axis(0), 2);

        let mut div_v = Array3::zeros((self.nx, self.ny, self.nz));

        // Compute divergence with central differences
        for i in 1..self.nx - 1 {
            for j in 1..self.ny - 1 {
                for k in 1..self.nz - 1 {
                    let dvx_dx = (vx[[i + 1, j, k]] - vx[[i - 1, j, k]]) / (2.0 * grid.dx);
                    let dvy_dy = (vy[[i, j + 1, k]] - vy[[i, j - 1, k]]) / (2.0 * grid.dy);
                    let dvz_dz = (vz[[i, j, k + 1]] - vz[[i, j, k - 1]]) / (2.0 * grid.dz);

                    div_v[[i, j, k]] = dvx_dx + dvy_dy + dvz_dz;
                }
            }
        }

        div_v
    }

    /// Update pressure field with C-PML
    fn update_pressure_with_cpml(
        &self,
        pressure: &mut Array3<f64>,
        div_v: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) {
        // Update pressure: dp/dt = -rho*c^2 * div(v)
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    let rho = crate::medium::density_at(medium, x, y, z, grid);
                    let c = crate::medium::sound_speed_at(medium, x, y, z, grid);
                    let rho_c2 = rho * c * c;
                    pressure[[i, j, k]] -= dt * rho_c2 * div_v[[i, j, k]];
                }
            }
        }
    }

    /// Get C-PML configuration
    pub fn config(&self) -> &CPMLConfig {
        self.cpml.config()
    }

    /// Enable dispersive media support
    pub fn enable_dispersive_support(
        &mut self,
        params: &crate::boundary::cpml::DispersiveParameters,
    ) {
        self.cpml.enable_dispersive_support(params.clone());
    }

    /// Estimate reflection coefficient at given angle
    pub fn estimate_reflection(&self, angle_degrees: f64) -> f64 {
        self.cpml.estimate_reflection(angle_degrees)
    }
}

/// Helper trait for C-PML integration with existing solvers
pub trait WithCPML {
    /// Apply C-PML boundary conditions
    fn apply_cpml(&mut self, solver: &mut CPMLSolver, grid: &Grid, dt: f64) -> KwaversResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpml_solver_creation() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::default();
        let dt = 1e-7;
        let sound_speed = 1500.0;
        let solver = CPMLSolver::new(config, &grid, dt, sound_speed).unwrap();

        assert_eq!(solver.nx, 64);
        assert_eq!(solver.grad_x.dim(), (64, 64, 64));
    }

    #[test]
    fn test_gradient_computation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::default();
        let dt = 1e-7;
        let sound_speed = 1500.0;
        let mut solver = CPMLSolver::new(config, &grid, dt, sound_speed).unwrap();

        // Create test pressure field
        let mut pressure = Array3::zeros((32, 32, 32));
        for i in 0..32 {
            for j in 0..32 {
                for k in 0..32 {
                    // Linear gradient in x
                    pressure[[i, j, k]] = i as f64;
                }
            }
        }

        solver.compute_pressure_gradients(&pressure, &grid);

        // Check gradient is approximately 1/dx in x-direction
        let expected_grad = 1.0 / grid.dx;
        for i in 1..31 {
            for j in 0..32 {
                for k in 0..32 {
                    assert!((solver.grad_x[[i, j, k]] - expected_grad).abs() < 1e-10);
                }
            }
        }
    }
}

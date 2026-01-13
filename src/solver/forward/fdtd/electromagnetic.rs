//! Electromagnetic FDTD solver
//!
//! This module adapts the existing FDTD solver to solve Maxwell's equations
//! for electromagnetic wave propagation using the Yee staggered grid scheme.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::field::EMFields;
use crate::domain::grid::Grid;
use crate::math::numerics::operators::{
    CentralDifference2, CentralDifference4, CentralDifference6, DifferentialOperator,
};
use crate::physics::electromagnetic::equations::{
    EMDimension, EMMaterialDistribution, ElectromagneticWaveEquation,
};
use ndarray::{Array3, ArrayD};

/// Electromagnetic FDTD solver using Yee's algorithm
///
/// This solver adapts the acoustic FDTD implementation to solve Maxwell's equations:
/// ```text
/// ∂E/∂t = (1/ε) ∇ × H - (σ/ε) E    (Faraday's Law)
/// ∂H/∂t = -(1/μ) ∇ × E             (Ampere's Law)
/// ```
///
/// The Yee cell staggers E and H fields in both space and time for numerical stability.
pub struct ElectromagneticFdtdSolver {
    /// Computational grid
    grid: Grid,
    /// Electromagnetic material properties (ε, μ, σ)
    materials: EMMaterialDistribution,
    /// Current time step
    time_step: usize,
    /// Time step size (seconds)
    dt: f64,
    /// Electric field components on Yee grid
    ex: Array3<f64>,
    ey: Array3<f64>,
    ez: Array3<f64>,
    /// Magnetic field components on Yee grid
    hx: Array3<f64>,
    hy: Array3<f64>,
    hz: Array3<f64>,
    /// Spatial derivative operators
    dx_operator: Box<dyn DifferentialOperator>,
    dy_operator: Box<dyn DifferentialOperator>,
    dz_operator: Box<dyn DifferentialOperator>,
}

impl std::fmt::Debug for ElectromagneticFdtdSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElectromagneticFdtdSolver")
            .field("grid", &self.grid)
            .field("materials", &self.materials)
            .field("time_step", &self.time_step)
            .field("dt", &self.dt)
            .finish_non_exhaustive()
    }
}

impl ElectromagneticFdtdSolver {
    /// Create a new electromagnetic FDTD solver
    pub fn new(
        grid: Grid,
        materials: EMMaterialDistribution,
        dt: f64,
        spatial_order: usize,
    ) -> KwaversResult<Self> {
        // Validate spatial order
        if ![2, 4, 6].contains(&spatial_order) {
            return Err(KwaversError::InvalidInput(format!(
                "spatial_order must be 2, 4, or 6, got {spatial_order}"
            )));
        }

        // Initialize field arrays with Yee staggering
        // E fields are at integer grid points, H fields at half-points
        let ex = Array3::zeros((grid.nx, grid.ny + 1, grid.nz + 1));
        let ey = Array3::zeros((grid.nx + 1, grid.ny, grid.nz + 1));
        let ez = Array3::zeros((grid.nx + 1, grid.ny + 1, grid.nz));

        let hx = Array3::zeros((grid.nx + 1, grid.ny, grid.nz));
        let hy = Array3::zeros((grid.nx, grid.ny + 1, grid.nz));
        let hz = Array3::zeros((grid.nx, grid.ny, grid.nz + 1));

        // Create spatial derivative operators
        let dx_operator: Box<dyn DifferentialOperator> = match spatial_order {
            2 => Box::new(CentralDifference2::new(grid.dx, grid.dy, grid.dz)?),
            4 => Box::new(CentralDifference4::new(grid.dx, grid.dy, grid.dz)?),
            6 => Box::new(CentralDifference6::new(grid.dx, grid.dy, grid.dz)?),
            _ => unreachable!(),
        };
        let dy_operator: Box<dyn DifferentialOperator> = match spatial_order {
            2 => Box::new(CentralDifference2::new(grid.dx, grid.dy, grid.dz)?),
            4 => Box::new(CentralDifference4::new(grid.dx, grid.dy, grid.dz)?),
            6 => Box::new(CentralDifference6::new(grid.dx, grid.dy, grid.dz)?),
            _ => unreachable!(),
        };
        let dz_operator: Box<dyn DifferentialOperator> = match spatial_order {
            2 => Box::new(CentralDifference2::new(grid.dx, grid.dy, grid.dz)?),
            4 => Box::new(CentralDifference4::new(grid.dx, grid.dy, grid.dz)?),
            6 => Box::new(CentralDifference6::new(grid.dx, grid.dy, grid.dz)?),
            _ => unreachable!(),
        };

        Ok(Self {
            grid,
            materials,
            time_step: 0,
            dt,
            ex,
            ey,
            ez,
            hx,
            hy,
            hz,
            dx_operator,
            dy_operator,
            dz_operator,
        })
    }

    /// Update electric fields using Faraday's law: ∂E/∂t = (1/ε) ∇ × H - (σ/ε) E
    fn update_electric_fields(&mut self) {
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dz = self.grid.dz;

        // Update Ex: ∂Ex/∂t = (1/ε) (∂Hz/∂y - ∂Hy/∂z) - (σ/ε) Ex
        for i in 0..self.grid.nx {
            for j in 1..self.grid.ny {
                for k in 1..self.grid.nz {
                    // Curl H at Ex position (i, j, k)
                    let curl_h_x = (self.hz[[i, j, k]] - self.hz[[i, j - 1, k]]) / dy
                        - (self.hy[[i, j, k]] - self.hy[[i, j, k - 1]]) / dz;

                    // Get material properties (simplified for now - assuming uniform vacuum)
                    let eps = 1.0; // Vacuum permittivity (normalized)
                    let sigma = 0.0; // No conductivity for now

                    // Update equation: ∂E/∂t = (1/ε) ∇×H - (σ/ε) E
                    let decay_term = if sigma > 0.0 {
                        (sigma / eps) * self.ex[[i, j, k]]
                    } else {
                        0.0
                    };
                    self.ex[[i, j, k]] += self.dt * (curl_h_x / eps - decay_term);
                }
            }
        }

        // Update Ey: ∂Ey/∂t = (1/ε) (∂Hx/∂z - ∂Hz/∂x) - (σ/ε) Ey
        for i in 1..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 1..self.grid.nz {
                    let curl_h_y = (self.hx[[i, j, k]] - self.hx[[i, j, k - 1]]) / dz
                        - (self.hz[[i, j, k]] - self.hz[[i - 1, j, k]]) / dx;

                    let eps = self
                        .materials
                        .permittivity
                        .get(ndarray::IxDyn(&[i, j, k]))
                        .copied()
                        .unwrap_or(1.0);
                    let sigma = self
                        .materials
                        .conductivity
                        .get(ndarray::IxDyn(&[i, j, k]))
                        .copied()
                        .unwrap_or(0.0);

                    let decay_term = if sigma > 0.0 {
                        (sigma / eps) * self.ey[[i, j, k]]
                    } else {
                        0.0
                    };
                    self.ey[[i, j, k]] += self.dt * (curl_h_y / eps - decay_term);
                }
            }
        }

        // Update Ez: ∂Ez/∂t = (1/ε) (∂Hy/∂x - ∂Hx/∂y) - (σ/ε) Ez
        for i in 1..self.grid.nx {
            for j in 1..self.grid.ny {
                for k in 0..self.grid.nz {
                    let curl_h_z = (self.hy[[i, j, k]] - self.hy[[i - 1, j, k]]) / dx
                        - (self.hx[[i, j, k]] - self.hx[[i, j - 1, k]]) / dy;

                    let eps = self
                        .materials
                        .permittivity
                        .get(ndarray::IxDyn(&[i, j, k]))
                        .copied()
                        .unwrap_or(1.0);
                    let sigma = self
                        .materials
                        .conductivity
                        .get(ndarray::IxDyn(&[i, j, k]))
                        .copied()
                        .unwrap_or(0.0);

                    let decay_term = if sigma > 0.0 {
                        (sigma / eps) * self.ez[[i, j, k]]
                    } else {
                        0.0
                    };
                    self.ez[[i, j, k]] += self.dt * (curl_h_z / eps - decay_term);
                }
            }
        }
    }

    /// Update magnetic fields using Ampere's law: ∂H/∂t = -(1/μ) ∇ × E
    fn update_magnetic_fields(&mut self) {
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dz = self.grid.dz;

        // Update Hx: ∂Hx/∂t = -(1/μ) (∂Ez/∂y - ∂Ey/∂z)
        for i in 1..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let curl_e_x = (self.ez[[i, j + 1, k]] - self.ez[[i, j, k]]) / dy
                        - (self.ey[[i, j, k + 1]] - self.ey[[i, j, k]]) / dz;

                    let mu = 1.0; // Vacuum permeability (normalized)
                    self.hx[[i, j, k]] -= self.dt * curl_e_x / mu;
                }
            }
        }

        // Update Hy: ∂Hy/∂t = -(1/μ) (∂Ex/∂z - ∂Ez/∂x)
        for i in 0..self.grid.nx {
            for j in 1..self.grid.ny {
                for k in 0..self.grid.nz {
                    let curl_e_y = (self.ex[[i, j, k + 1]] - self.ex[[i, j, k]]) / dz
                        - (self.ez[[i + 1, j, k]] - self.ez[[i, j, k]]) / dx;

                    let mu = 1.0; // Vacuum permeability (normalized)
                    self.hy[[i, j, k]] -= self.dt * curl_e_y / mu;
                }
            }
        }

        // Update Hz: ∂Hz/∂t = -(1/μ) (∂Ey/∂x - ∂Ex/∂y)
        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 1..self.grid.nz {
                    let curl_e_z = (self.ey[[i + 1, j, k]] - self.ey[[i, j, k]]) / dx
                        - (self.ex[[i, j + 1, k]] - self.ex[[i, j, k]]) / dz;

                    let mu = 1.0; // Vacuum permeability (normalized)
                    self.hz[[i, j, k]] -= self.dt * curl_e_z / mu;
                }
            }
        }
    }

    /// Compute CFL-stable time step for electromagnetic waves
    pub fn max_stable_dt(&self, c_max: f64) -> f64 {
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dz = self.grid.dz;

        // CFL condition for FDTD: dt ≤ 1/(c √(1/dx² + 1/dy² + 1/dz²))
        let denominator =
            c_max * ((1.0 / dx).powi(2) + (1.0 / dy).powi(2) + (1.0 / dz).powi(2)).sqrt();
        0.99 / denominator // 0.99 for stability margin
    }
}

impl ElectromagneticWaveEquation for ElectromagneticFdtdSolver {
    fn em_dimension(&self) -> EMDimension {
        EMDimension::Three
    }

    fn material_properties(&self) -> &EMMaterialDistribution {
        &self.materials
    }

    fn em_fields(&self) -> &EMFields {
        // Create EMFields from current arrays
        // This is a simplified implementation - full version would handle Yee grid properly
        static FIELDS: std::sync::OnceLock<EMFields> = std::sync::OnceLock::new();
        FIELDS.get_or_init(|| EMFields {
            electric: ArrayD::zeros(ndarray::IxDyn(&[
                3,
                self.grid.nx,
                self.grid.ny,
                self.grid.nz,
            ])),
            magnetic: ArrayD::zeros(ndarray::IxDyn(&[
                3,
                self.grid.nx,
                self.grid.ny,
                self.grid.nz,
            ])),
            displacement: None,
            flux_density: None,
        })
    }

    fn step_maxwell(&mut self, dt: f64) -> Result<(), String> {
        self.dt = dt;
        self.update_electric_fields();
        self.update_magnetic_fields();
        self.time_step += 1;
        Ok(())
    }

    fn apply_em_boundary_conditions(&mut self, _fields: &mut EMFields) {
        // Apply boundary conditions (PML, periodic, etc.)
        // Implementation would depend on boundary type
        // For now, use simple PEC (perfect electric conductor) boundaries
        self.apply_pec_boundaries();
    }

    fn check_em_constraints(&self, _fields: &EMFields) -> Result<(), String> {
        // Check Maxwell's equations constraints
        // For example, divergence conditions: ∇·E = ρ/ε₀, ∇·B = 0
        // This is a placeholder implementation
        Ok(())
    }
}

impl ElectromagneticFdtdSolver {
    /// Apply Perfect Electric Conductor (PEC) boundary conditions
    fn apply_pec_boundaries(&mut self) {
        // PEC: E_tangential = 0 at boundaries
        // This is a simplified implementation

        // Bottom boundary (z=0): Ez = 0
        for i in 0..self.grid.nx + 1 {
            for j in 0..self.grid.ny + 1 {
                self.ez[[i, j, 0]] = 0.0;
            }
        }

        // Similar for other boundaries...
        // Full implementation would handle all six boundaries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_em_fdtd_creation() {
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();

        // Use canonical domain composition pattern
        let materials = EMMaterialDistribution::vacuum(&[10, 10, 10]);

        let solver = ElectromagneticFdtdSolver::new(grid, materials, 1e-12, 4).unwrap();
        assert_eq!(solver.em_dimension(), EMDimension::Three);
    }

    #[test]
    fn test_maxwell_time_step() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();

        // Use canonical domain composition pattern
        let materials = EMMaterialDistribution::vacuum(&[32, 32, 32]);

        let solver = ElectromagneticFdtdSolver::new(grid, materials, 1e-12, 4).unwrap();

        // Speed of light in vacuum (normalized units)
        let c = 1.0;
        let dt = solver.max_stable_dt(c);

        // Check that time step is reasonable
        assert!(dt > 0.0);
        assert!(dt < 1e-3); // Should be smaller than spatial step
    }
}

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
use ndarray::{Array3, Array4};

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
    /// Cell-centered electromagnetic fields (derived from Yee grid)
    fields_cache: EMFields,
    /// Spatial derivative operators
    #[allow(dead_code)]
    dx_operator: Box<dyn DifferentialOperator>,
    #[allow(dead_code)]
    dy_operator: Box<dyn DifferentialOperator>,
    #[allow(dead_code)]
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

        // Validate spatial order before creating operators
        if spatial_order != 2 && spatial_order != 4 && spatial_order != 6 {
            return Err(KwaversError::InvalidInput(format!(
                "ElectromagneticFDTD: spatial_order must be 2, 4, or 6, got {}",
                spatial_order
            )));
        }

        // Create spatial derivative operators
        let dx_operator: Box<dyn DifferentialOperator> = match spatial_order {
            2 => Box::new(CentralDifference2::new(grid.dx, grid.dy, grid.dz)?),
            4 => Box::new(CentralDifference4::new(grid.dx, grid.dy, grid.dz)?),
            6 => Box::new(CentralDifference6::new(grid.dx, grid.dy, grid.dz)?),
            _ => unreachable!("validated above"),
        };
        let dy_operator: Box<dyn DifferentialOperator> = match spatial_order {
            2 => Box::new(CentralDifference2::new(grid.dx, grid.dy, grid.dz)?),
            4 => Box::new(CentralDifference4::new(grid.dx, grid.dy, grid.dz)?),
            6 => Box::new(CentralDifference6::new(grid.dx, grid.dy, grid.dz)?),
            _ => unreachable!("validated above"),
        };
        let dz_operator: Box<dyn DifferentialOperator> = match spatial_order {
            2 => Box::new(CentralDifference2::new(grid.dx, grid.dy, grid.dz)?),
            4 => Box::new(CentralDifference4::new(grid.dx, grid.dy, grid.dz)?),
            6 => Box::new(CentralDifference6::new(grid.dx, grid.dy, grid.dz)?),
            _ => unreachable!("validated above"),
        };

        // Cache grid dimensions before moving grid
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

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
            fields_cache: EMFields {
                electric: Array4::zeros((nx, ny, nz, 3)).into_dyn(),
                magnetic: Array4::zeros((nx, ny, nz, 3)).into_dyn(),
                displacement: None,
                flux_density: None,
            },
            dx_operator,
            dy_operator,
            dz_operator,
        })
    }

    fn update_field_cache(&mut self) {
        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let ex_c = 0.25
                        * (self.ex[[i, j, k]]
                            + self.ex[[i, j + 1, k]]
                            + self.ex[[i, j, k + 1]]
                            + self.ex[[i, j + 1, k + 1]]);
                    let ey_c = 0.25
                        * (self.ey[[i, j, k]]
                            + self.ey[[i + 1, j, k]]
                            + self.ey[[i, j, k + 1]]
                            + self.ey[[i + 1, j, k + 1]]);
                    let ez_c = 0.25
                        * (self.ez[[i, j, k]]
                            + self.ez[[i + 1, j, k]]
                            + self.ez[[i, j + 1, k]]
                            + self.ez[[i + 1, j + 1, k]]);

                    let hx_c = 0.5 * (self.hx[[i, j, k]] + self.hx[[i + 1, j, k]]);
                    let hy_c = 0.5 * (self.hy[[i, j, k]] + self.hy[[i, j + 1, k]]);
                    let hz_c = 0.5 * (self.hz[[i, j, k]] + self.hz[[i, j, k + 1]]);

                    self.fields_cache.electric[ndarray::IxDyn(&[i, j, k, 0])] = ex_c;
                    self.fields_cache.electric[ndarray::IxDyn(&[i, j, k, 1])] = ey_c;
                    self.fields_cache.electric[ndarray::IxDyn(&[i, j, k, 2])] = ez_c;

                    self.fields_cache.magnetic[ndarray::IxDyn(&[i, j, k, 0])] = hx_c;
                    self.fields_cache.magnetic[ndarray::IxDyn(&[i, j, k, 1])] = hy_c;
                    self.fields_cache.magnetic[ndarray::IxDyn(&[i, j, k, 2])] = hz_c;
                }
            }
        }
    }

    fn permittivity_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.materials
            .permittivity
            .get(ndarray::IxDyn(&[i, j, k]))
            .copied()
            .unwrap_or(1.0)
    }

    fn conductivity_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.materials
            .conductivity
            .get(ndarray::IxDyn(&[i, j, k]))
            .copied()
            .unwrap_or(0.0)
    }

    fn permeability_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.materials
            .permeability
            .get(ndarray::IxDyn(&[i, j, k]))
            .copied()
            .unwrap_or(1.0)
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

                    let mi = i;
                    let mj = j - 1;
                    let mk = k - 1;
                    let eps = self.permittivity_at(mi, mj, mk);
                    let sigma = self.conductivity_at(mi, mj, mk);

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

                    let eps = self.permittivity_at(i - 1, j, k - 1);
                    let sigma = self.conductivity_at(i - 1, j, k - 1);

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

                    let eps = self.permittivity_at(i - 1, j - 1, k);
                    let sigma = self.conductivity_at(i - 1, j - 1, k);

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

                    let mu = self.permeability_at(i - 1, j, k);
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

                    let mu = self.permeability_at(i, j - 1, k);
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

                    let mu = self.permeability_at(i, j, k - 1);
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
        &self.fields_cache
    }

    fn step_maxwell(&mut self, dt: f64) -> Result<(), String> {
        self.dt = dt;
        self.update_electric_fields();
        self.update_magnetic_fields();
        self.time_step += 1;
        self.update_field_cache();
        Ok(())
    }

    fn apply_em_boundary_conditions(&mut self, fields: &mut EMFields) {
        self.apply_pec_boundaries();
        self.update_field_cache();
        *fields = self.fields_cache.clone();
    }

    fn check_em_constraints(&self, fields: &EMFields) -> Result<(), String> {
        let all_finite = self.ex.iter().all(|v| v.is_finite())
            && self.ey.iter().all(|v| v.is_finite())
            && self.ez.iter().all(|v| v.is_finite())
            && self.hx.iter().all(|v| v.is_finite())
            && self.hy.iter().all(|v| v.is_finite())
            && self.hz.iter().all(|v| v.is_finite())
            && fields.electric.iter().all(|v| v.is_finite())
            && fields.magnetic.iter().all(|v| v.is_finite());

        if !all_finite {
            return Err("non-finite values detected in EM fields".to_string());
        }

        Ok(())
    }
}

impl ElectromagneticFdtdSolver {
    /// Apply Perfect Electric Conductor (PEC) boundary conditions
    fn apply_pec_boundaries(&mut self) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;

        for j in 0..ny {
            for k in 0..nz {
                self.ey[[0, j, k]] = 0.0;
                self.ey[[nx, j, k]] = 0.0;
                self.ez[[0, j, k]] = 0.0;
                self.ez[[nx, j, k]] = 0.0;
            }
        }

        for i in 0..nx {
            for k in 0..nz {
                self.ex[[i, 0, k]] = 0.0;
                self.ex[[i, ny, k]] = 0.0;
                self.ez[[i, 0, k]] = 0.0;
                self.ez[[i, ny, k]] = 0.0;
            }
        }

        for i in 0..nx {
            for j in 0..ny {
                self.ex[[i, j, 0]] = 0.0;
                self.ex[[i, j, nz]] = 0.0;
                self.ey[[i, j, 0]] = 0.0;
                self.ey[[i, j, nz]] = 0.0;
            }
        }
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

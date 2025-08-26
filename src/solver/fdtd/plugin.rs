//! FDTD solver plugin implementation

use ndarray::Array4;
use std::collections::HashMap;
use std::fmt::Debug;

use super::{FdtdConfig, FdtdSolver};
use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_mapping::UnifiedFieldType;
use crate::physics::plugin::{PhysicsPlugin, PluginContext, PluginMetadata, PluginState};

/// FDTD solver plugin
#[derive(Debug)]
pub struct FdtdPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    solver: FdtdSolver,
}

impl FdtdPlugin {
    /// Create a new FDTD plugin
    pub fn new(config: FdtdConfig, grid: &Grid) -> KwaversResult<Self> {
        let solver = FdtdSolver::new(config, grid)?;

        Ok(Self {
            metadata: PluginMetadata {
                id: "fdtd_solver".to_string(),
                name: "FDTD Solver".to_string(),
                version: "1.0.0".to_string(),
                description: "Finite-Difference Time Domain solver for acoustic wave propagation"
                    .to_string(),
                author: "Kwavers Team".to_string(),
                license: "MIT".to_string(),
            },
            state: PluginState::Initialized,
            solver,
        })
    }
}

impl PhysicsPlugin for FdtdPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![
            UnifiedFieldType::Pressure,
            UnifiedFieldType::VelocityX,
            UnifiedFieldType::VelocityY,
            UnifiedFieldType::VelocityZ,
        ]
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![
            UnifiedFieldType::Pressure,
            UnifiedFieldType::VelocityX,
            UnifiedFieldType::VelocityY,
            UnifiedFieldType::VelocityZ,
        ]
    }

    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        self.state = PluginState::Running;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        // Ensure fields have correct dimensions
        if fields.dim().0 < 4 {
            return Err(crate::error::KwaversError::Physics(
                crate::error::PhysicsError::InvalidFieldDimensions {
                    expected: "at least 4 field components".to_string(),
                    actual: format!("{} components", fields.dim().0),
                },
            ));
        }

        // Extract field slices
        let pressure_idx = UnifiedFieldType::Pressure.index();
        let vx_idx = UnifiedFieldType::VelocityX.index();
        let vy_idx = UnifiedFieldType::VelocityY.index();
        let vz_idx = UnifiedFieldType::VelocityZ.index();

        // Update pressure field using velocity divergence
        {
            let vx = fields.index_axis(ndarray::Axis(0), vx_idx).to_owned();
            let vy = fields.index_axis(ndarray::Axis(0), vy_idx).to_owned();
            let vz = fields.index_axis(ndarray::Axis(0), vz_idx).to_owned();

            let mut pressure = fields.index_axis_mut(ndarray::Axis(0), pressure_idx);
            let (nx, ny, nz) = pressure.dim();

            // Apply FDTD update for pressure
            for i in 1..nx - 1 {
                for j in 1..ny - 1 {
                    for k in 1..nz - 1 {
                        // Compute velocity divergence using central differences
                        let dvx_dx = (vx[[i + 1, j, k]] - vx[[i - 1, j, k]]) / (2.0 * grid.dx);
                        let dvy_dy = (vy[[i, j + 1, k]] - vy[[i, j - 1, k]]) / (2.0 * grid.dy);
                        let dvz_dz = (vz[[i, j, k + 1]] - vz[[i, j, k - 1]]) / (2.0 * grid.dz);

                        let divergence = dvx_dx + dvy_dy + dvz_dz;

                        // Get medium properties at this point
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        let rho = medium.density(x, y, z, grid);
                        let c = medium.sound_speed(x, y, z, grid);

                        // Update pressure
                        pressure[[i, j, k]] -= dt * rho * c * c * divergence;
                    }
                }
            }
        }

        // Update velocity fields using pressure gradient
        {
            let pressure = fields.index_axis(ndarray::Axis(0), pressure_idx).to_owned();

            // Update vx
            {
                let mut vx = fields.index_axis_mut(ndarray::Axis(0), vx_idx);
                let (nx, ny, nz) = vx.dim();

                for i in 1..nx - 1 {
                    for j in 0..ny {
                        for k in 0..nz {
                            let dp_dx = (pressure[[i + 1.min(nx - 1), j, k]]
                                - pressure[[i.saturating_sub(1), j, k]])
                                / (2.0 * grid.dx);

                            let x = i as f64 * grid.dx;
                            let y = j as f64 * grid.dy;
                            let z = k as f64 * grid.dz;
                            let rho = medium.density(x, y, z, grid);

                            vx[[i, j, k]] -= dt * dp_dx / rho;
                        }
                    }
                }
            }

            // Update vy
            {
                let mut vy = fields.index_axis_mut(ndarray::Axis(0), vy_idx);
                let (nx, ny, nz) = vy.dim();

                for i in 0..nx {
                    for j in 1..ny - 1 {
                        for k in 0..nz {
                            let dp_dy = (pressure[[i, j + 1.min(ny - 1), k]]
                                - pressure[[i, j.saturating_sub(1), k]])
                                / (2.0 * grid.dy);

                            let x = i as f64 * grid.dx;
                            let y = j as f64 * grid.dy;
                            let z = k as f64 * grid.dz;
                            let rho = medium.density(x, y, z, grid);

                            vy[[i, j, k]] -= dt * dp_dy / rho;
                        }
                    }
                }
            }

            // Update vz
            {
                let mut vz = fields.index_axis_mut(ndarray::Axis(0), vz_idx);
                let (nx, ny, nz) = vz.dim();

                for i in 0..nx {
                    for j in 0..ny {
                        for k in 1..nz - 1 {
                            let dp_dz = (pressure[[i, j, k + 1.min(nz - 1)]]
                                - pressure[[i, j, k.saturating_sub(1)]])
                                / (2.0 * grid.dz);

                            let x = i as f64 * grid.dx;
                            let y = j as f64 * grid.dy;
                            let z = k as f64 * grid.dz;
                            let rho = medium.density(x, y, z, grid);

                            vz[[i, j, k]] -= dt * dp_dz / rho;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        Ok(())
    }

    fn diagnostics(&self) -> HashMap<String, f64> {
        HashMap::new()
    }

    fn reset(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Initialized;
        Ok(())
    }
}

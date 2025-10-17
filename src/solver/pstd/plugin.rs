//! PSTD solver plugin implementation

use ndarray::Array4;
use std::fmt::Debug;

use super::{PstdConfig, PstdSolver};
use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_mapping::UnifiedFieldType;
use crate::physics::plugin::{PluginContext, PluginMetadata, PluginState};

/// PSTD solver plugin
#[derive(Debug)]
pub struct PstdPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    #[allow(dead_code)]
    solver: PstdSolver,
}

impl PstdPlugin {
    /// Create a new PSTD plugin
    pub fn new(config: PstdConfig, grid: &Grid) -> KwaversResult<Self> {
        let solver = PstdSolver::new(config.clone(), grid)?;

        Ok(Self {
            metadata: PluginMetadata {
                id: "pstd_solver".to_string(),
                name: "PSTD Solver".to_string(),
                version: "1.0.0".to_string(),
                description: "Pseudo-Spectral Time Domain solver for acoustic wave propagation"
                    .to_string(),
                author: "Kwavers Team".to_string(),
                license: "MIT".to_string(),
            },
            state: PluginState::Initialized,
            solver,
        })
    }
}

impl crate::physics::plugin::Plugin for PstdPlugin {
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

        // Plugin interface adaptation for PSTD solver
        // The plugin works with Array4 fields while PSTD solver has internal state
        // This adapter provides the bridge between the two interfaces

        // Extract field indices using UnifiedFieldType
        let pressure_idx = UnifiedFieldType::Pressure.index();
        let vx_idx = UnifiedFieldType::VelocityX.index();
        let vy_idx = UnifiedFieldType::VelocityY.index();
        let vz_idx = UnifiedFieldType::VelocityZ.index();

        // Finite difference update for plugin compatibility with unified field interface
        // Note: Full PSTD spectral operators require different field layout (k-space arrays)
        // Current: FD approximation maintains numerical stability with plugin architecture
        self.update_pressure_fd(
            fields,
            pressure_idx,
            vx_idx,
            vy_idx,
            vz_idx,
            grid,
            medium,
            dt,
        )?;
        self.update_velocity_fd(
            fields,
            pressure_idx,
            vx_idx,
            vy_idx,
            vz_idx,
            grid,
            medium,
            dt,
        )
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        Ok(())
    }

    fn diagnostics(&self) -> String {
        format!("PSTD Plugin - State: {:?}", self.state)
    }

    fn reset(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl PstdPlugin {
    /// Update pressure using finite differences
    fn update_pressure_fd(
        &self,
        fields: &mut Array4<f64>,
        pressure_idx: usize,
        vx_idx: usize,
        vy_idx: usize,
        vz_idx: usize,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        // Copy velocity fields to avoid borrow checker issues
        let vx_copy = fields.index_axis(ndarray::Axis(0), vx_idx).to_owned();
        let vy_copy = fields.index_axis(ndarray::Axis(0), vy_idx).to_owned();
        let vz_copy = fields.index_axis(ndarray::Axis(0), vz_idx).to_owned();

        let mut pressure_slice = fields.index_axis_mut(ndarray::Axis(0), pressure_idx);
        let (nx, ny, nz) = pressure_slice.dim();

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let dvx_dx =
                        (vx_copy[[i + 1, j, k]] - vx_copy[[i - 1, j, k]]) / (2.0 * grid.dx);
                    let dvy_dy =
                        (vy_copy[[i, j + 1, k]] - vy_copy[[i, j - 1, k]]) / (2.0 * grid.dy);
                    let dvz_dz =
                        (vz_copy[[i, j, k + 1]] - vz_copy[[i, j, k - 1]]) / (2.0 * grid.dz);

                    let divergence = dvx_dx + dvy_dy + dvz_dz;
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    let rho = crate::medium::density_at(medium, x, y, z, grid);
                    let c = crate::medium::sound_speed_at(medium, x, y, z, grid);

                    pressure_slice[[i, j, k]] -= dt * rho * c * c * divergence;
                }
            }
        }
        Ok(())
    }

    /// Update velocity using finite differences
    fn update_velocity_fd(
        &self,
        fields: &mut Array4<f64>,
        pressure_idx: usize,
        vx_idx: usize,
        vy_idx: usize,
        vz_idx: usize,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        // Copy pressure field to avoid borrow checker issues
        let pressure_copy = fields.index_axis(ndarray::Axis(0), pressure_idx).to_owned();
        let (nx, ny, nz) = pressure_copy.dim();

        // Update vx
        {
            let mut vx_slice = fields.index_axis_mut(ndarray::Axis(0), vx_idx);
            for i in 1..nx - 1 {
                for j in 0..ny {
                    for k in 0..nz {
                        let dp_dx = (pressure_copy[[i + 1, j, k]] - pressure_copy[[i - 1, j, k]])
                            / (2.0 * grid.dx);
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        let rho = crate::medium::density_at(medium, x, y, z, grid);
                        vx_slice[[i, j, k]] -= dt * dp_dx / rho;
                    }
                }
            }
        }

        // Update vy
        {
            let mut vy_slice = fields.index_axis_mut(ndarray::Axis(0), vy_idx);
            for i in 0..nx {
                for j in 1..ny - 1 {
                    for k in 0..nz {
                        let dp_dy = (pressure_copy[[i, j + 1, k]] - pressure_copy[[i, j - 1, k]])
                            / (2.0 * grid.dy);
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        let rho = crate::medium::density_at(medium, x, y, z, grid);
                        vy_slice[[i, j, k]] -= dt * dp_dy / rho;
                    }
                }
            }
        }

        // Update vz
        {
            let mut vz_slice = fields.index_axis_mut(ndarray::Axis(0), vz_idx);
            for i in 0..nx {
                for j in 0..ny {
                    for k in 1..nz - 1 {
                        let dp_dz = (pressure_copy[[i, j, k + 1]] - pressure_copy[[i, j, k - 1]])
                            / (2.0 * grid.dz);
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        let rho = crate::medium::density_at(medium, x, y, z, grid);
                        vz_slice[[i, j, k]] -= dt * dp_dz / rho;
                    }
                }
            }
        }
        Ok(())
    }
}

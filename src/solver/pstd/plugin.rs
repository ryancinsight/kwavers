//! PSTD solver plugin implementation

use ndarray::Array4;
use std::collections::HashMap;
use std::fmt::Debug;

use super::{PstdConfig, PstdSolver};
use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_mapping::UnifiedFieldType;
use crate::physics::plugin::{PhysicsPlugin, PluginContext, PluginMetadata, PluginState};

/// PSTD solver plugin
#[derive(Debug)]
pub struct PstdPlugin {
    metadata: PluginMetadata,
    state: PluginState,
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

impl PhysicsPlugin for PstdPlugin {
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
        t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        // Extract pressure and velocity fields from the unified field array
        // Assuming standard field indices for acoustic fields
        let pressure_idx = 0;
        let vx_idx = 1;
        let vy_idx = 2;
        let vz_idx = 3;
        
        // Get field slices and perform spectral operations
        let mut pressure = fields.index_axis(ndarray::Axis(0), pressure_idx).to_owned();
        let mut velocity_x = fields.index_axis(ndarray::Axis(0), vx_idx).to_owned();
        let mut velocity_y = fields.index_axis(ndarray::Axis(0), vy_idx).to_owned();
        let mut velocity_z = fields.index_axis(ndarray::Axis(0), vz_idx).to_owned();
        
        // Compute spectral divergence of velocity for pressure update
        let (dvx_dx, _, _) = self.solver.spectral.compute_gradient(&velocity_x, grid)?;
        let (_, dvy_dy, _) = self.solver.spectral.compute_gradient(&velocity_y, grid)?;
        let (_, _, dvz_dz) = self.solver.spectral.compute_gradient(&velocity_z, grid)?;
        
        // Update pressure field using divergence
        ndarray::Zip::indexed(&mut pressure)
            .and(&dvx_dx)
            .and(&dvy_dy)
            .and(&dvz_dz)
            .for_each(|(i, j, k), p, &dx, &dy, &dz| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let rho = medium.density(x, y, z, grid);
                let c = medium.sound_speed(x, y, z, grid);
                let divergence = dx + dy + dz;
                *p -= dt * rho * c * c * divergence;
            });
        
        // Compute pressure gradient for velocity update
        let (grad_x, grad_y, grad_z) = self.solver.spectral.compute_gradient(&pressure, grid)?;
        
        // Update velocity components
        ndarray::Zip::indexed(&mut velocity_x)
            .and(&grad_x)
            .for_each(|(i, j, k), v, &grad| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let rho = medium.density(x, y, z, grid);
                *v -= dt * grad / rho;
            });
            
        ndarray::Zip::indexed(&mut velocity_y)
            .and(&grad_y)
            .for_each(|(i, j, k), v, &grad| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let rho = medium.density(x, y, z, grid);
                *v -= dt * grad / rho;
            });
            
        ndarray::Zip::indexed(&mut velocity_z)
            .and(&grad_z)
            .for_each(|(i, j, k), v, &grad| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let rho = medium.density(x, y, z, grid);
                *v -= dt * grad / rho;
            });
        
        // Copy updated fields back to the plugin fields
        fields.index_axis_mut(ndarray::Axis(0), pressure_idx).assign(&pressure);
        fields.index_axis_mut(ndarray::Axis(0), vx_idx).assign(&velocity_x);
        fields.index_axis_mut(ndarray::Axis(0), vy_idx).assign(&velocity_y);
        fields.index_axis_mut(ndarray::Axis(0), vz_idx).assign(&velocity_z);
        
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

//! Acoustic wave propagation plugin
//!
//! This plugin implements acoustic wave physics following SOLID principles.
//! It depends only on abstractions and is completely decoupled from the solver.

use super::{Plugin, PluginContext, PluginMetadata, PluginState};
use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_mapping::UnifiedFieldType;
use ndarray::{Array3, Array4};

/// Acoustic wave propagation plugin
#[derive(Debug)]
pub struct AcousticWavePlugin {
    metadata: PluginMetadata,
    state: PluginState,
    /// Wave speed field (cached)
    sound_speed: Option<Array3<f64>>,
    /// Density field (cached)
    density: Option<Array3<f64>>,
    /// Previous pressure for time integration
    prev_pressure: Option<Array3<f64>>,
    /// CFL number for stability
    cfl_number: f64,
}

impl AcousticWavePlugin {
    /// Create a new acoustic wave plugin
    #[must_use]
    pub fn new(cfl_number: f64) -> Self {
        Self {
            metadata: PluginMetadata {
                id: "acoustic_wave".to_string(),
                name: "Acoustic Wave Propagation".to_string(),
                version: "1.0.0".to_string(),
                description: "Implements linear acoustic wave equation".to_string(),
                author: "Kwavers".to_string(),
                license: "MIT".to_string(),
            },
            state: PluginState::Created,
            sound_speed: None,
            density: None,
            prev_pressure: None,
            cfl_number,
        }
    }

    /// Update acoustic pressure using finite differences
    fn update_pressure(
        &self,
        pressure: &mut Array3<f64>,
        vx: &Array3<f64>,
        vy: &Array3<f64>,
        vz: &Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) {
        let dx_inv = 1.0 / grid.dx;
        let dy_inv = 1.0 / grid.dy;
        let dz_inv = 1.0 / grid.dz;

        let (nx, ny, nz) = pressure.dim();

        // Update pressure: ∂p/∂t = -ρc² (∇·v)
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let dvx_dx = (vx[[i + 1, j, k]] - vx[[i - 1, j, k]]) * 0.5 * dx_inv;
                    let dvy_dy = (vy[[i, j + 1, k]] - vy[[i, j - 1, k]]) * 0.5 * dy_inv;
                    let dvz_dz = (vz[[i, j, k + 1]] - vz[[i, j, k - 1]]) * 0.5 * dz_inv;

                    let divergence = dvx_dx + dvy_dy + dvz_dz;

                    let rho = self.density.as_ref().unwrap()[[i, j, k]];
                    let c = self.sound_speed.as_ref().unwrap()[[i, j, k]];

                    pressure[[i, j, k]] -= dt * rho * c * c * divergence;
                }
            }
        }
    }

    /// Update particle velocities using finite differences
    fn update_velocities(
        &self,
        vx: &mut Array3<f64>,
        vy: &mut Array3<f64>,
        vz: &mut Array3<f64>,
        pressure: &Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) {
        let dx_inv = 1.0 / grid.dx;
        let dy_inv = 1.0 / grid.dy;
        let dz_inv = 1.0 / grid.dz;

        let (nx, ny, nz) = pressure.dim();

        // Update velocities: ∂v/∂t = -(1/ρ) ∇p
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let rho_inv = 1.0 / self.density.as_ref().unwrap()[[i, j, k]];

                    // Update vx
                    let dp_dx = (pressure[[i + 1, j, k]] - pressure[[i - 1, j, k]]) * 0.5 * dx_inv;
                    vx[[i, j, k]] -= dt * rho_inv * dp_dx;

                    // Update vy
                    let dp_dy = (pressure[[i, j + 1, k]] - pressure[[i, j - 1, k]]) * 0.5 * dy_inv;
                    vy[[i, j, k]] -= dt * rho_inv * dp_dy;

                    // Update vz
                    let dp_dz = (pressure[[i, j, k + 1]] - pressure[[i, j, k - 1]]) * 0.5 * dz_inv;
                    vz[[i, j, k]] -= dt * rho_inv * dp_dz;
                }
            }
        }
    }
}

impl Plugin for AcousticWavePlugin {
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

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        // Cache medium properties
        self.sound_speed = Some(medium.sound_speed_array().to_owned());
        self.density = Some(medium.density_array().to_owned());

        // Initialize previous pressure
        self.prev_pressure = Some(Array3::zeros((grid.nx, grid.ny, grid.nz)));

        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        _medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        // Extract field slices
        let pressure_idx = UnifiedFieldType::Pressure.index();
        let vx_idx = UnifiedFieldType::VelocityX.index();
        let vy_idx = UnifiedFieldType::VelocityY.index();
        let vz_idx = UnifiedFieldType::VelocityZ.index();

        // Get views of the fields
        let mut pressure = fields
            .index_axis_mut(ndarray::Axis(0), pressure_idx)
            .to_owned();
        let mut vx = fields.index_axis_mut(ndarray::Axis(0), vx_idx).to_owned();
        let mut vy = fields.index_axis_mut(ndarray::Axis(0), vy_idx).to_owned();
        let mut vz = fields.index_axis_mut(ndarray::Axis(0), vz_idx).to_owned();

        // Leapfrog time integration
        // Step 1: Update velocities (half step)
        self.update_velocities(&mut vx, &mut vy, &mut vz, &pressure, grid, dt * 0.5);

        // Step 2: Update pressure (full step)
        self.update_pressure(&mut pressure, &vx, &vy, &vz, grid, dt);

        // Step 3: Update velocities (half step)
        self.update_velocities(&mut vx, &mut vy, &mut vz, &pressure, grid, dt * 0.5);

        // Write back to fields
        fields
            .index_axis_mut(ndarray::Axis(0), pressure_idx)
            .assign(&pressure);
        fields.index_axis_mut(ndarray::Axis(0), vx_idx).assign(&vx);
        fields.index_axis_mut(ndarray::Axis(0), vy_idx).assign(&vy);
        fields.index_axis_mut(ndarray::Axis(0), vz_idx).assign(&vz);

        // Store current pressure for next step
        self.prev_pressure = Some(pressure);

        self.state = PluginState::Running;
        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.sound_speed = None;
        self.density = None;
        self.prev_pressure = None;
        self.state = PluginState::Finalized;
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

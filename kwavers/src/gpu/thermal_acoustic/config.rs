//! `GpuThermalAcousticConfig` — grid parameters and coupling coefficients.
//!
//! SRP: changes when the uniform buffer layout or CFL bounds change.

use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
use crate::core::constants::thermodynamic::BODY_TEMPERATURE_C;
use crate::core::error::{KwaversError, KwaversResult};

/// Configuration for GPU thermal-acoustic coupling.
///
/// Field names use physics notation (dc_dT = ∂c/∂T) to match the
/// mathematical specification and WGSL shader variable names.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
pub struct GpuThermalAcousticConfig {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
    pub dt: f32,
    pub c_ref: f32,
    pub dc_dT: f32,
    pub rho_ref: f32,
    pub drho_dT: f32,
    pub T_ref: f32,
    pub alpha_thermal: f32,
    pub alpha_ac: f32,
    pub T_arterial: f32,
    pub w_b: f32,
    pub Q_met: f32,
}

impl Default for GpuThermalAcousticConfig {
    fn default() -> Self {
        Self {
            nx: 64,
            ny: 64,
            nz: 64,
            dx: 0.001,
            dy: 0.001,
            dz: 0.001,
            dt: 1e-8,
            c_ref: SOUND_SPEED_TISSUE as f32,
            dc_dT: 2.0,
            rho_ref: 1000.0,
            drho_dT: -0.2,
            T_ref: BODY_TEMPERATURE_C as f32,
            alpha_thermal: 1.5e-7,
            alpha_ac: 0.5,
            T_arterial: BODY_TEMPERATURE_C as f32,
            w_b: 5.0,
            Q_met: 0.0,
        }
    }
}

impl GpuThermalAcousticConfig {
    /// Validate CFL conditions for acoustic and thermal stability.
    ///
    /// Acoustic CFL: c_max · Δt / Δx < 0.3.
    /// Thermal stability: α · Δt / Δx² < 0.25.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err(KwaversError::InvalidInput(
                "Grid dimensions must be positive".to_string(),
            ));
        }
        if self.dx <= 0.0 || self.dy <= 0.0 || self.dz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Grid spacing must be positive".to_string(),
            ));
        }
        if self.dt <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Time step must be positive".to_string(),
            ));
        }

        let max_c = self.c_ref + self.dc_dT * 10.0;
        let cfl_acoustic = max_c * self.dt / self.dx.min(self.dy).min(self.dz);
        if cfl_acoustic >= 0.3 {
            return Err(KwaversError::InvalidInput(format!(
                "CFL condition violated (acoustic): {:.4} >= 0.3",
                cfl_acoustic
            )));
        }

        let cfl_thermal = self.alpha_thermal * self.dt / (self.dx * self.dx);
        if cfl_thermal >= 0.25 {
            return Err(KwaversError::InvalidInput(format!(
                "CFL condition violated (thermal): {:.4} >= 0.25",
                cfl_thermal
            )));
        }

        Ok(())
    }
}

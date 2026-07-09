//! Thermal-Acoustic Coupling Solver
//!
//! This module implements a monolithic thermal-acoustic solver where acoustic
//! propagation couples with temperature through material property variation.
//!
//! ## Physical Model
//!
//! **Acoustic System**:
//! ```text
//! ρ(T) ∂u/∂t = -∇p + f_ext
//! ∂p/∂t = -ρ(T) c²(T) ∇·u + Q_ac
//! ```
//!
//! **Thermal System** (Pennes bioheat), divided through by `ρc` so the perfusion
//! coefficient `w_b` is the lumped volumetric rate `[1/s]` (= ω_b·ρ_b·c_b/(ρc)):
//! ```text
//! ∂T/∂t = α ∇²T + w_b (T_a - T) + (Q_m + Q_ac)/(ρc)      [α = k/(ρc)]
//! ```
//!
//! **Coupling**:
//! - Sound speed varies with temperature: c(T) = c_ref + ∂c/∂T · (T - T_ref)
//! - Density varies: ρ(T) = ρ_ref + ∂ρ/∂T · (T - T_ref)
//! - Acoustic heating becomes thermal source: Q_ac = α|p|²/(ρc)
//!
//! ## Time Integration
//!
//! Uses forward Euler for simplicity, can be extended to RK2/RK4:
//! 1. Evaluate acoustic source from current pressure field
//! 2. Compute material properties from current temperature
//! 3. Advance acoustic equation one step
//! 4. Compute acoustic heating from pressure field
//! 5. Advance thermal equation one step
//! 6. Update material properties at new temperature
//!
//! ## Stability
//!
//! CFL condition for acoustic: c_max * dt / dx < 0.3 (FDTD)
//! Thermal stability: α * dt / dx² < 0.25
//! Combined constraint uses max of both
//!
//! ## References
//!
//! - Baysal et al. (2005): "Thermal-acoustic coupling in focused ultrasound therapy"
//! - Santos & Douglas (2008): "Monolithic formulation for coupled multiphase flow"
//! - Kolski-Andreaco et al. (2015): "Nonlinear acoustic heating in therapeutic ultrasound"

use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;
use kwavers_core::constants::{
    DC_DT_SOFT_TISSUE, DENSITY_WATER_NOMINAL, DRHO_DT_SOFT_TISSUE, SOUND_SPEED_TISSUE,
    THERMAL_DIFFUSIVITY_TISSUE,
};
use leto::Array3;

mod physics;
mod stepping;
#[cfg(test)]
mod tests;

/// Configuration for thermal-acoustic coupling solver
#[derive(Debug, Clone, Copy)]
pub struct ThermalAcousticConfig {
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,

    /// Grid spacing (m)
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,

    /// Time step (s)
    pub dt: f64,

    /// Reference sound speed (m/s)
    pub c_ref: f64,

    /// Temperature dependence of sound speed (m/s/°C)
    pub dc_d_t: f64,

    /// Reference density (kg/m³)
    pub rho_ref: f64,

    /// Temperature dependence of density (kg/m³/°C)
    pub drho_d_t: f64,

    /// Reference temperature (°C)
    pub t_ref: f64,

    /// Thermal diffusivity (m²/s)
    pub alpha_thermal: f64,

    /// Acoustic attenuation coefficient (Np/m)
    pub alpha_ac: f64,

    /// Arterial blood temperature (°C)
    pub t_arterial: f64,

    /// Blood perfusion rate (1/s)
    pub w_b: f64,

    /// Metabolic heat generation (W/m³)
    pub q_met: f64,

    /// Enable temperature-dependent material properties
    pub enable_temperature_coupling: bool,
}

impl Default for ThermalAcousticConfig {
    fn default() -> Self {
        Self {
            nx: 64,
            ny: 64,
            nz: 64,
            dx: 0.001,
            dy: 0.001,
            dz: 0.001,
            dt: 0.001e-6,
            c_ref: SOUND_SPEED_TISSUE,
            dc_d_t: DC_DT_SOFT_TISSUE,
            rho_ref: DENSITY_WATER_NOMINAL,
            drho_d_t: DRHO_DT_SOFT_TISSUE,
            t_ref: BODY_TEMPERATURE_C,
            alpha_thermal: THERMAL_DIFFUSIVITY_TISSUE,
            alpha_ac: 0.5,
            t_arterial: BODY_TEMPERATURE_C,
            w_b: 5.0,
            q_met: 0.0,
            enable_temperature_coupling: true,
        }
    }
}

/// Thermal-acoustic coupling solver
#[derive(Debug, Clone)]
pub struct ThermalAcousticCoupler {
    /// Configuration
    pub(super) config: ThermalAcousticConfig,

    /// Pressure field (Pa)
    pub(super) pressure: Array3<f64>,

    /// Velocity fields (m/s)
    pub(super) velocity_x: Array3<f64>,
    pub(super) velocity_y: Array3<f64>,
    pub(super) velocity_z: Array3<f64>,

    /// Temperature field (°C)
    pub(super) temperature: Array3<f64>,

    /// Previous state for time stepping
    pub(super) pressure_prev: Array3<f64>,
    pub(super) velocity_x_prev: Array3<f64>,
    pub(super) velocity_y_prev: Array3<f64>,
    pub(super) velocity_z_prev: Array3<f64>,
    pub(super) temperature_prev: Array3<f64>,

    /// Material properties (computed from temperature)
    pub(super) density: Array3<f64>,
    pub(super) sound_speed: Array3<f64>,

    /// Acoustic heating source (W/m³)
    pub(super) acoustic_heating: Array3<f64>,

    /// Time step counter
    pub(super) step_count: u64,

    /// Total simulation time (s)
    pub(super) total_time: f64,
}

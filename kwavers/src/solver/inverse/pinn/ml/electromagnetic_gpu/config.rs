use crate::core::constants::fundamental::{VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY};

#[derive(Debug, Clone)]
pub struct EMConfig {
    /// Grid dimensions [nx, ny, nz]
    pub grid_size: [usize; 3],
    /// Number of time steps
    pub time_steps: usize,
    /// Electric permittivity (F/m)
    pub permittivity: f64,
    /// Magnetic permeability (H/m)
    pub permeability: f64,
    /// Electrical conductivity (S/m)
    pub conductivity: f64,
    /// Spatial step sizes [dx, dy, dz] (m)
    pub spatial_steps: [f64; 3],
    /// Time step (s)
    pub time_step: f64,
    /// Courant stability factor
    pub courant_factor: f64,
    /// Boundary condition type
    pub boundary_condition: ElectromagneticBc,
}

impl Default for EMConfig {
    fn default() -> Self {
        Self {
            grid_size: [64, 64, 64],
            time_steps: 1000,
            permittivity: VACUUM_PERMITTIVITY,
            permeability: VACUUM_PERMEABILITY,
            conductivity: 0.0,                 // Perfect dielectric
            spatial_steps: [1e-3, 1e-3, 1e-3], // 1mm resolution
            time_step: 1e-12,                  // 1ps time step
            courant_factor: 0.99,              // Conservative stability
            boundary_condition: ElectromagneticBc::Absorbing,
        }
    }
}

/// Boundary condition types for electromagnetic simulations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElectromagneticBc {
    /// Perfect Electric Conductor (E = 0)
    PerfectElectricConductor,
    /// Perfect Magnetic Conductor (H = 0)
    PerfectMagneticConductor,
    /// Absorbing boundary condition
    Absorbing,
    /// Periodic boundary condition
    Periodic,
}

/// GPU buffer data structures matching WGSL
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EMConstants {
    permittivity: f32,
    permeability: f32,
    conductivity: f32,
    dt: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    nx: u32,
    ny: u32,
    nz: u32,
}

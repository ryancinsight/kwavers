//! Unified multi-physics solver
//!
//! This module provides a solver that can handle coupled acoustic-optical-thermal simulations.

pub mod acoustic_optical;
pub mod coupled_solver;
pub mod field_coupling;
pub mod photoacoustic;
pub mod thermal_optical;

pub use acoustic_optical::AcousticOpticalSolver;
pub use coupled_solver::MultiPhysicsSolver;
pub use field_coupling::{CouplingStrategy, FieldCoupler};
pub use photoacoustic::PhotoacousticSolver;
pub use thermal_optical::ThermalOpticalSolver;

/// Multi-physics field indices
pub mod field_indices {
    /// Acoustic pressure field index
    pub const ACOUSTIC_PRESSURE: usize = 0;
    /// Optical intensity field index
    pub const OPTICAL_INTENSITY: usize = 1;
    /// Temperature field index
    pub const TEMPERATURE: usize = 2;
    /// Total number of fields in multi-physics simulation
    pub const TOTAL_FIELDS: usize = 3;
}

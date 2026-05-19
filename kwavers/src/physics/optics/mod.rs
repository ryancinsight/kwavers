// src/physics/optics/mod.rs
//! # Optics Module
//!
//! Implements optical physics for biomedical imaging and light-matter interactions.
//! Classical models (blackbody, bremsstrahlung, Cherenkov, Mie, Monte Carlo photon
//! transport) are fully implemented.  Full quantum electrodynamics (photon number
//! states, Jaynes-Cummings, Lamb shift) is out of scope for this classical/
//! semiclassical acoustic simulator and is not planned.

pub mod diffusion;
pub mod map_builder;
pub mod monte_carlo;
pub mod nonlinear;
pub mod polarization;
pub mod quantum_optics;
pub mod scattering;
pub mod sonoluminescence;

// Re-export commonly used types
pub use diffusion::{LightDiffusion, DiffusionOpticalProperties};

// Re-export domain types for backwards compatibility
pub use crate::domain::medium::optical_map::{
    Layer, OpticalPropertyMap, OpticalPropertyMapBuilder, Region,
};

// Re-export physics-specific analysis types
pub use map_builder::PropertyStats;

pub use monte_carlo::{MCResult, MonteCarloSolver, PhotonSource, SimulationConfig};
pub use nonlinear::{KerrEffect, PhotoacousticConversion};
pub use polarization::{
    JonesMatrix, JonesPolarizationModel, JonesVector, LinearPolarization, PolarizationModel,
};
pub use quantum_optics::{
    gaunt_factor_ff, lamb_shift_ev, relativistic_parameter, EinsteinCoefficients,
    QuantumCorrectionAssessment,
};
pub use scattering::{MieCalculator, MieParameters, MieResult, RayleighScattering};
pub use sonoluminescence::{EmissionParameters, SonoluminescenceEmission};

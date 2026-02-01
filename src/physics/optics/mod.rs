// src/physics/optics/mod.rs
//! # Optics Module
//!
//! Implements optical physics for biomedical imaging and light-matter interactions.
//! TODO_AUDIT: P1 - Quantum Optics Framework - Implement quantum electrodynamics framework for extreme light-matter interactions in sonoluminescence
//! DEPENDS ON: physics/quantum/qed.rs, physics/optics/quantum_emission.rs
//! MISSING: Photon number states and quantum field operators for intense light fields
//! MISSING: Jaynes-Cummings model for atom-light coupling in hot plasma
//! MISSING: Quantum cascade effects in multi-photon emission processes
//! MISSING: Entanglement generation in Cherenkov radiation from coherent electron beams
//! MISSING: Quantum corrections to classical emission spectra (Lamb shift, spontaneous emission)

pub mod diffusion;
pub mod map_builder;
pub mod monte_carlo;
pub mod nonlinear;
pub mod polarization;
pub mod scattering;
pub mod sonoluminescence;

// Re-export commonly used types
pub use diffusion::{LightDiffusion, OpticalProperties};

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
pub use scattering::{MieCalculator, MieParameters, MieResult, RayleighScattering};
pub use sonoluminescence::{EmissionParameters, SonoluminescenceEmission};

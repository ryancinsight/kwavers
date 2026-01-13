//! Shear Wave Elastography (SWE) Physics
//!
//! Implements elastic wave propagation, acoustic radiation force, and elasticity inversion.
//!
//! # Components
//!
//! - **Displacement**: Tracking tissue motion
//! - **Elastic Wave Solver**: FDTD simulation of shear waves
//! - **Inversion**: Reconstructing elasticity from wave speed
//! - **Radiation Force**: ARFI force generation
//! - **Harmonic Detection**: For nonlinear parameter estimation

pub mod displacement;
pub mod harmonic_detection;
pub mod nonlinear;
pub mod radiation_force;

// Re-exports
pub use displacement::DisplacementField;

// Note: Solver components (ElasticWaveSolver, GPU solvers, Inversion)
// have been moved to crate::solver::forward::elastic::swe
// and crate::solver::inverse::elastography to enforce architectural boundaries.

pub use harmonic_detection::{
    HarmonicDetectionConfig, HarmonicDetector, HarmonicDisplacementField,
};
pub use nonlinear::{HyperelasticModel, NonlinearElasticWaveSolver, NonlinearSWEConfig};
pub use radiation_force::{AcousticRadiationForce, MultiDirectionalPush};

// Note: ShearWaveElastography orchestrator has moved to crate::simulation::imaging::elastography

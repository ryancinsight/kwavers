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
//!
pub mod displacement;
pub mod harmonic_detection;
pub mod radiation_force;

// Re-exports
pub use displacement::DisplacementField;

// Note: Solver components have been moved to enforce architectural boundaries:
// - Linear elastic solver: crate::solver::forward::elastic::swe
// - Nonlinear elastic solver: crate::solver::forward::elastic::nonlinear
// - Inversion methods: crate::solver::inverse::elastography
//
pub use harmonic_detection::{
    HarmonicDetectionConfig, HarmonicDetector, HarmonicDisplacementField,
};
pub use radiation_force::{AcousticRadiationForce, MultiDirectionalPush};

// Note: ShearWaveElastography orchestrator has moved to crate::simulation::imaging::elastography
// Removed backward compatibility re-export to avoid circular dependency (physics → solver)
// Users should import directly from solver layer:
// - NonlinearElasticWaveSolver: crate::solver::forward::elastic::nonlinear

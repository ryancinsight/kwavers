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
pub mod elastic_wave_solver;
pub mod gpu_accelerated_3d;
pub mod harmonic_detection;
pub mod inversion;
pub mod nonlinear;
pub mod radiation_force;

// Re-exports
pub use displacement::DisplacementField;
pub use elastic_wave_solver::{
    ArrivalDetection, ElasticBodyForceConfig, ElasticWaveConfig, ElasticWaveField,
    ElasticWaveSolver, VolumetricSource, VolumetricWaveConfig, WaveFrontTracker,
};
pub use gpu_accelerated_3d::{AdaptiveResolution, GPUDevice, GPUElasticWaveSolver3D};
pub use harmonic_detection::{
    HarmonicDetectionConfig, HarmonicDetector, HarmonicDisplacementField,
};
pub use inversion::{NonlinearInversion, ShearWaveInversion};
pub use nonlinear::{HyperelasticModel, NonlinearElasticWaveSolver, NonlinearSWEConfig};
pub use radiation_force::{AcousticRadiationForce, MultiDirectionalPush};

// Note: ShearWaveElastography orchestrator has moved to crate::simulation::imaging::elastography

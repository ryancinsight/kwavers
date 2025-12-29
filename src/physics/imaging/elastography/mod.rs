//! Shear Wave Elastography (SWE) Module
//!
//! Implements clinical shear wave elastography for tissue characterization.
//!
//! ## Overview
//!
//! Shear wave elastography measures tissue stiffness by:
//! 1. Generating shear waves via acoustic radiation force impulse (ARFI)
//! 2. Tracking wave propagation with ultrafast imaging
//! 3. Reconstructing elasticity from shear wave speed
//!
//! ## Literature References
//!
//! - Sarvazyan, A. P., et al. (1998). "Shear wave elasticity imaging: a new ultrasonic
//!   technology of medical diagnostics." *Ultrasound in Medicine & Biology*, 24(9), 1419-1435.
//! - Bercoff, J., et al. (2004). "Supersonic shear imaging: a new technique for soft tissue
//!   elasticity mapping." *IEEE TUFFC*, 51(4), 396-409.
//! - Deffieux, T., et al. (2009). "Shear wave spectroscopy for in vivo quantification of
//!   human soft tissues visco-elasticity." *IEEE TMI*, 28(3), 313-322.
//!
//! ## Clinical Applications
//!
//! - Liver fibrosis assessment (non-invasive)
//! - Breast tumor differentiation (benign vs malignant)
//! - Prostate cancer detection
//! - Thyroid nodule characterization

pub mod displacement;
pub mod elastic_wave_solver;
pub mod harmonic_detection;
pub mod inversion;
pub mod nonlinear;
pub mod radiation_force;

pub use displacement::{DisplacementEstimator, DisplacementField};
pub use elastic_wave_solver::{
    ElasticWaveConfig, ElasticWaveField, ElasticWaveSolver, VolumetricQualityMetrics,
    VolumetricWaveConfig, WaveFrontTracker,
};

// GPU acceleration for 3D SWE
pub mod gpu_accelerated_3d;
pub use gpu_accelerated_3d::{
    AdaptiveResolution, AdaptiveSolution, AdaptiveSolutionStep, GPUDevice, GPUElasticWaveSolver3D,
    GPUInversionResult, GPUMemoryPool, GPUPropagationResult, MemoryStats, PerformanceMetrics,
    PerformanceStatistics,
};
pub use harmonic_detection::{
    HarmonicDetectionConfig, HarmonicDetector, HarmonicDisplacementField,
};
pub use inversion::{
    ElasticityMap, InversionMethod, NonlinearInversion, NonlinearInversionMethod,
    NonlinearParameterMap, ShearWaveInversion,
};
pub use nonlinear::{HyperelasticModel, NonlinearElasticWaveSolver, NonlinearSWEConfig};
pub use radiation_force::{
    AcousticRadiationForce, DirectionalPush, DirectionalQuality, DirectionalWaveTracker,
    MultiDirectionalPush, PushPulseParameters, TrackingRegion, ValidationResult,
};

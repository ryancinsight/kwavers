//! Beamforming Algorithms for Ultrasound Arrays
//!
//! This module implements state-of-the-art beamforming algorithms for ultrasound
//! imaging and passive acoustic mapping, following established literature and
//! designed for large-scale array processing.
//!
//! # Design Principles
//! - **Literature-Based**: All algorithms follow established papers
//! - **Zero-Copy**: Efficient `ArrayView` usage throughout
//! - **Sparse Operations**: Designed for large arrays with sparse matrices
//! - **Modular Design**: Plugin-compatible architecture
//!
//! # Field jargon / capability map
//! - **A (broadband / transient)**: time-domain **DAS / SRP-DAS** via `time_domain` with an explicit
//!   **delay datum / delay reference** (recommended default: reference sensor 0).
//! - **B (narrowband / adaptive)**: point-steered **MVDR/Capon spatial spectrum** via `narrowband`
//!   plus subspace methods (MUSIC/ESMV) via `adaptive` + steering + covariance estimation.
//!
//! # Literature References
//! - Van Veen & Buckley (1988): "Beamforming: A versatile approach to spatial filtering"
//! - Li et al. (2003): "Robust Capon beamforming"
//! - Schmidt (1986): "Multiple emitter location and signal parameter estimation"
//! - Capon (1969): "High-resolution frequency-wavenumber spectrum analysis"
//! - Frost (1972): "An algorithm for linearly constrained adaptive array processing"

pub mod adaptive;
#[cfg(any(feature = "experimental_neural", feature = "pinn"))]
pub mod ai_integration;
mod beamforming_3d;
mod config;
mod covariance;
#[cfg(feature = "experimental_neural")]
pub mod experimental;
pub mod narrowband;
mod processor;
#[cfg(feature = "gpu")]
mod shaders;
mod steering;
pub mod time_domain;

pub use adaptive::{
    AdaptiveBeamformer, ArrayGeometry, BeamformingAlgorithm as AdaptiveBeamformingAlgorithm,
    CovarianceTaper, DelayAndSum, MinimumVariance, SteeringMatrix,
    SteeringVector as AdaptiveSteeringVector, WeightCalculator, WeightingScheme,
};
pub use beamforming_3d::{
    ApodizationWindow, BeamformingAlgorithm3D, BeamformingConfig3D, BeamformingMetrics,
    BeamformingProcessor3D,
};
pub use config::{BeamformingConfig, BeamformingCoreConfig};
pub use covariance::{
    CovarianceEstimator, CovariancePostProcess, SpatialSmoothing, SpatialSmoothingComplex,
};
pub use narrowband::{
    capon_spatial_spectrum_point, capon_spatial_spectrum_point_complex_baseband,
    extract_complex_baseband_snapshots, extract_narrowband_snapshots, BasebandSnapshotConfig,
    CaponSpectrumConfig, NarrowbandSteering, NarrowbandSteeringVector, SnapshotMethod,
    SnapshotScenario, SnapshotSelection, StftBinConfig, WindowFunction,
};
pub use processor::BeamformingProcessor;
pub use steering::{SteeringVector, SteeringVectorMethod};
pub use time_domain::das::{delay_and_sum_time_domain_with_reference, DEFAULT_DELAY_REFERENCE};
pub use time_domain::{
    relative_delays_s as relative_tof_delays_s, DelayReference as TimeDomainDelayReference,
};

#[cfg(feature = "experimental_neural")]
pub use experimental::neural::{
    // PinnBeamformingResult,
    // DistributedNeuralBeamformingProcessor,
    // DistributedNeuralBeamformingResult,
    // DistributedNeuralBeamformingMetrics,
    BeamformingFeedback,
    // NeuralBeamformingProcessor, // Only with pinn
    // PINNBeamformingConfig,
    HybridBeamformingResult,
    NeuralBeamformer,
    NeuralBeamformingConfig,
    NeuralBeamformingNetwork,
    NeuralLayer,
    PhysicsConstraints,
    UncertaintyEstimator,
};

#[cfg(any(feature = "experimental_neural", feature = "pinn"))]
pub use ai_integration::{
    AIBeamformingConfig, AIBeamformingResult, AIEnhancedBeamformingProcessor,
    ClinicalDecisionSupport, DiagnosisAlgorithm, FeatureExtractor, RealTimeWorkflow,
};
#[cfg(feature = "pinn")]
pub use experimental::neural::{
    DistributedNeuralBeamformingMetrics, DistributedNeuralBeamformingProcessor,
    DistributedNeuralBeamformingResult, FaultToleranceState, ModelParallelConfig,
    NeuralBeamformingProcessor, PINNBeamformingConfig, PinnBeamformingResult, PipelineStage,
};

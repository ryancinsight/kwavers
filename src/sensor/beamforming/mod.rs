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
//! # Literature References
//! - Van Veen & Buckley (1988): "Beamforming: A versatile approach to spatial filtering"
//! - Li et al. (2003): "Robust Capon beamforming"
//! - Schmidt (1986): "Multiple emitter location and signal parameter estimation"
//! - Capon (1969): "High-resolution frequency-wavenumber spectrum analysis"
//! - Frost (1972): "An algorithm for linearly constrained adaptive array processing"

#[cfg(any(feature = "experimental_neural", feature = "pinn"))]
pub mod ai_integration;
mod algorithms;
mod beamforming_3d;
mod config;
mod covariance;
#[cfg(any(feature = "experimental_neural", feature = "pinn"))]
mod neural;
mod processor;
#[cfg(feature = "gpu")]
mod shaders;
mod steering;

pub use algorithms::{AlgorithmImplementation, BeamformingAlgorithm, MVDRBeamformer};
pub use beamforming_3d::{
    ApodizationWindow, BeamformingAlgorithm3D, BeamformingConfig3D, BeamformingMetrics,
    BeamformingProcessor3D,
};
pub use config::{BeamformingConfig, BeamformingCoreConfig};
pub use covariance::{CovarianceEstimator, SpatialSmoothing};
pub use processor::BeamformingProcessor;
pub use steering::{SteeringVector, SteeringVectorMethod};

#[cfg(any(feature = "experimental_neural", feature = "pinn"))]
pub use neural::{
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
pub use neural::{
    DistributedNeuralBeamformingMetrics, DistributedNeuralBeamformingProcessor,
    DistributedNeuralBeamformingResult, FaultToleranceState, ModelParallelConfig,
    NeuralBeamformingProcessor, PINNBeamformingConfig, PinnBeamformingResult, PipelineStage,
};

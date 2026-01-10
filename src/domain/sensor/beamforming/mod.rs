//! Beamforming Algorithms for Ultrasound Arrays
//!
//! ⚠️ **DEPRECATION NOTICE** ⚠️
//!
//! This module is **deprecated** and will be removed in version 3.0.0.
//!
//! **New Location:** [`crate::analysis::signal_processing::beamforming`]
//!
//! # Migration Guide
//!
//! Beamforming algorithms have been moved to the analysis layer to enforce proper
//! architectural layering. The domain layer should only contain sensor geometry
//! and hardware-specific primitives, not signal processing algorithms.
//!
//! ## Quick Migration
//!
//! **Old (deprecated):**
//! ```rust,ignore
//! use kwavers::domain::sensor::beamforming::adaptive::MinimumVariance;
//! use kwavers::domain::sensor::beamforming::time_domain::DelayAndSum;
//! ```
//!
//! **New (canonical):**
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::beamforming::adaptive::MinimumVariance;
//! use kwavers::analysis::signal_processing::beamforming::time_domain::DelayAndSum;
//! ```
//!
//! ## Migration Timeline
//!
//! - **Version 2.1.0** (current): Deprecation warnings, backward compatibility maintained
//! - **Version 2.2.0**: Continued support with deprecation warnings
//! - **Version 3.0.0**: Old location removed entirely
//!
//! ## Documentation
//!
//! See `docs/refactor/BEAMFORMING_MIGRATION_GUIDE.md` for complete migration instructions,
//! examples, and architectural rationale.
//!
//! ---
//!
//! # Original Documentation (Deprecated)
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
pub mod covariance;
#[cfg(feature = "experimental_neural")]
pub mod experimental;
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
pub use processor::BeamformingProcessor;
pub use steering::{SteeringVector, SteeringVectorMethod};
pub use time_domain::das::{delay_and_sum_time_domain_with_reference, DEFAULT_DELAY_REFERENCE};
pub use time_domain::{
    relative_delays_s as relative_tof_delays_s, DelayReference as TimeDomainDelayReference,
};

#[cfg(feature = "experimental_neural")]
pub use experimental::{
    BeamformingFeedback, HybridBeamformingMetrics, HybridBeamformingResult,
    NeuralBeamformingNetwork, NeuralLayer, PhysicsConstraints, UncertaintyEstimator,
};

// Note: NeuralBeamformer and NeuralBeamformingConfig are NOT YET MIGRATED.
// These high-level API types remain in the old monolithic file and will be
// extracted in a future sprint. Use the lower-level primitives directly:
// - NeuralBeamformingNetwork (for network operations)
// - PhysicsConstraints (for physics-informed constraints)
// - UncertaintyEstimator (for uncertainty quantification)

#[cfg(any(feature = "experimental_neural", feature = "pinn"))]
pub use ai_integration::{
    AIBeamformingConfig, AIBeamformingResult, AIEnhancedBeamformingProcessor,
    ClinicalDecisionSupport, DiagnosisAlgorithm, FeatureExtractor, RealTimeWorkflow,
};
#[cfg(feature = "pinn")]
pub use experimental::{
    DistributedNeuralBeamformingMetrics, DistributedNeuralBeamformingProcessor,
    DistributedNeuralBeamformingResult, FaultToleranceState, ModelParallelConfig,
    NeuralBeamformingProcessor, PINNBeamformingConfig, PinnBeamformingResult, PipelineStage,
};

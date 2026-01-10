//! Experimental beamforming algorithms and features
//!
//! ⚠️ **DEPRECATION NOTICE** ⚠️
//!
//! This module is **deprecated** and will be removed in version 3.0.0.
//!
//! **New Location:** [`crate::analysis::signal_processing::beamforming::neural`]
//!
//! All neural beamforming implementations have been moved to the analysis layer
//! to enforce proper architectural layering and eliminate duplication.
//!
//! ## Migration Guide
//!
//! **Old (deprecated):**
//! ```rust,ignore
//! use kwavers::domain::sensor::beamforming::experimental::neural::{
//!     NeuralBeamformer,
//!     NeuralBeamformingConfig,
//! };
//! ```
//!
//! **New (canonical):**
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::beamforming::neural::{
//!     NeuralBeamformingNetwork,
//!     PhysicsConstraints,
//!     UncertaintyEstimator,
//! };
//! ```
//!
//! ## Breaking Changes
//!
//! The high-level `NeuralBeamformer` and `NeuralBeamformingConfig` types from the
//! old monolithic implementation are **not yet migrated**. If you depend on these,
//! please file an issue or use the lower-level primitives directly:
//!
//! - [`NeuralBeamformingNetwork`](crate::analysis::signal_processing::beamforming::neural::NeuralBeamformingNetwork)
//! - [`NeuralLayer`](crate::analysis::signal_processing::beamforming::neural::NeuralLayer)
//! - [`PhysicsConstraints`](crate::analysis::signal_processing::beamforming::neural::PhysicsConstraints)
//! - [`UncertaintyEstimator`](crate::analysis::signal_processing::beamforming::neural::UncertaintyEstimator)
//!
//! ## Architectural Rationale
//!
//! Beamforming algorithms are **signal processing operations**, not domain primitives.
//! The domain layer should contain only:
//! - Sensor geometry definitions
//! - Hardware-specific configurations
//! - Physical transducer models
//!
//! Signal processing belongs in the analysis layer where it can be:
//! - Tested independently of hardware
//! - Reused across different sensor types
//! - Composed with other analysis algorithms
//!
//! See `docs/refactor/BEAMFORMING_MIGRATION_GUIDE.md` for complete details.

// Re-export from new canonical location
// Note: This provides backward compatibility shims during the deprecation period
pub use crate::analysis::signal_processing::beamforming::neural::{
    BeamformingFeedback, HybridBeamformingMetrics, HybridBeamformingResult,
    NeuralBeamformingNetwork, NeuralLayer, PhysicsConstraints, UncertaintyEstimator,
};

// PINN-specific types (feature-gated)
#[cfg(feature = "pinn")]
pub use crate::analysis::signal_processing::beamforming::neural::{
    distributed::{
        DistributedNeuralBeamformingProcessor, FaultToleranceState, ModelParallelConfig,
        PipelineStage,
    },
    pinn::NeuralBeamformingProcessor,
    types::{
        DistributedNeuralBeamformingMetrics, DistributedNeuralBeamformingResult,
        PINNBeamformingConfig, PinnBeamformingResult,
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward_compatibility_exports() {
        // Verify re-exports are accessible (compilation check)
        let _ = std::any::type_name::<NeuralBeamformingNetwork>();
        let _ = std::any::type_name::<NeuralLayer>();
        let _ = std::any::type_name::<PhysicsConstraints>();
        let _ = std::any::type_name::<UncertaintyEstimator>();
    }

    #[test]
    #[cfg(feature = "pinn")]
    fn test_pinn_backward_compatibility() {
        let _ = std::any::type_name::<NeuralBeamformingProcessor>();
        let _ = std::any::type_name::<PINNBeamformingConfig>();
    }
}

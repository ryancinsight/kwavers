//! Experimental beamforming algorithms and features
//!
//! This module contains cutting-edge, experimental beamforming implementations
//! that may not be stable or fully validated. Use with caution.
//!
//! Currently includes neural network-based beamforming approaches.

pub mod neural;

// Re-export neural beamforming components
pub use neural::{
    BeamformingFeedback, HybridBeamformingResult, NeuralBeamformer, NeuralBeamformingConfig,
    NeuralBeamformingNetwork, NeuralLayer, PhysicsConstraints, UncertaintyEstimator,
};

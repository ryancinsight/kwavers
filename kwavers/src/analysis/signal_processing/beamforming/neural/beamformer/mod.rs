//! High-level neural beamformer implementation.
//!
//! This module provides the main `NeuralBeamformer` struct that orchestrates
//! all neural beamforming operations, including traditional delay-and-sum,
//! neural network refinement, physics constraints, and uncertainty quantification.
//!
//! ## Architecture
//!
//! ```text
//! NeuralBeamformer
//! ├── Configuration (mode, network, physics, adaptation)
//! ├── Neural Network (optional, mode-dependent)
//! ├── Physics Constraints (reciprocity, coherence, sparsity)
//! ├── Uncertainty Estimator (dropout MC, local variance)
//! └── Performance Metrics (processing time, confidence, quality)
//! ```
//!
//! ## Processing Pipeline
//!
//! ```text
//! RF Data → Traditional DAS → Feature Extraction → Neural Network
//!                                                         ↓
//!           ← Uncertainty Estimation ← Physics Constraints ←
//! ```

use crate::core::error::KwaversResult;

use super::config::{NeuralBeamformingConfig, NeuralBeamformingMode};
use super::network::NeuralBeamformingNetwork;
use super::physics::PhysicsConstraints;
use super::types::{BeamformingFeedback, HybridBeamformingMetrics};
use super::uncertainty::UncertaintyEstimator;

/// Main neural beamformer processor.
#[derive(Debug)]
pub struct NeuralBeamformer {
    pub(super) config: NeuralBeamformingConfig,
    pub(super) neural_network: Option<NeuralBeamformingNetwork>,
    pub(super) physics_constraints: PhysicsConstraints,
    pub(super) uncertainty_estimator: UncertaintyEstimator,
    pub(super) metrics: HybridBeamformingMetrics,
}

impl NeuralBeamformer {
    /// New.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: NeuralBeamformingConfig) -> KwaversResult<Self> {
        config.validate()?;

        let neural_network = match config.mode {
            NeuralBeamformingMode::NeuralOnly
            | NeuralBeamformingMode::Hybrid
            | NeuralBeamformingMode::Adaptive => {
                Some(NeuralBeamformingNetwork::new(&config.network_architecture)?)
            }
            #[cfg(feature = "pinn")]
            NeuralBeamformingMode::PhysicsInformed => {
                Some(NeuralBeamformingNetwork::new(&config.network_architecture)?)
            }
        };

        let physics_constraints = PhysicsConstraints::new(
            config.physics_parameters.reciprocity_weight,
            config.physics_parameters.coherence_weight,
            config.physics_parameters.sparsity_weight,
        );

        let uncertainty_estimator = UncertaintyEstimator::default();

        Ok(Self {
            config,
            neural_network,
            physics_constraints,
            uncertainty_estimator,
            metrics: HybridBeamformingMetrics::default(),
        })
    }
    /// Adapt.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn adapt(&mut self, feedback: &BeamformingFeedback) -> KwaversResult<()> {
        if self.config.adaptation_parameters.enable_online_learning {
            if let Some(network) = &mut self.neural_network {
                network.adapt(feedback, self.config.adaptation_parameters.learning_rate)?;
            }
        }

        self.physics_constraints.update(feedback)?;

        Ok(())
    }

    #[must_use]
    pub fn metrics(&self) -> &HybridBeamformingMetrics {
        &self.metrics
    }

    #[must_use]
    pub fn config(&self) -> &NeuralBeamformingConfig {
        &self.config
    }
}

mod das;
#[cfg(feature = "pinn")]
mod physics_informed;
mod pipeline;
#[cfg(test)]
mod tests;

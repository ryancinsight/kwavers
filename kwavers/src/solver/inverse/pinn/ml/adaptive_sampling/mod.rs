//! Adaptive Sampling for PINN Training
//!
//! This module implements physics-aware adaptive collocation point sampling
//! to optimize PINN training efficiency. The system dynamically redistributes
//! collocation points based on PDE residual magnitude and solution complexity.
//!
//! ## Algorithm
//!
//! The adaptive sampling uses:
//! - Residual-based importance weighting
//! - Hierarchical refinement criteria
//! - Uncertainty quantification for sampling priority
//! - Memory-efficient point redistribution

mod evaluation;
mod refinement;
mod sampler;
#[cfg(test)]
mod tests;

use burn::tensor::{backend::AutodiffBackend, Tensor};

/// High residual region for targeted sampling
#[derive(Debug, Clone)]
struct HighResidualRegion {
    center_x: f32,
    center_y: f32,
    center_t: f32,
    size_x: f32,
    size_y: f32,
    size_t: f32,
    residual_magnitude: f32,
}

/// Sampling strategy configuration
#[derive(Debug, Clone)]
pub struct AdaptiveRefinementConfig {
    /// Minimum priority threshold for refinement
    pub refinement_threshold: f64,
    /// Maximum priority threshold for coarsening
    pub coarsening_threshold: f64,
    /// Fraction of points to refine per iteration
    pub refinement_fraction: f64,
    /// Fraction of points to coarsen per iteration
    pub coarsening_fraction: f64,
    /// Hierarchical refinement levels
    pub hierarchy_levels: usize,
    /// Uncertainty-based sampling weight
    pub uncertainty_weight: f64,
    /// Residual-based sampling weight
    pub residual_weight: f64,
}

/// Sampling statistics
#[derive(Debug, Clone)]
pub struct SamplingStats {
    /// Total sampling iterations performed
    pub iterations: usize,
    /// Points refined in last iteration
    pub points_refined: usize,
    /// Points coarsened in last iteration
    pub points_coarsened: usize,
    /// Current point distribution entropy
    pub distribution_entropy: f64,
    /// Average priority value
    pub avg_priority: f64,
    /// Maximum priority value
    pub max_priority: f64,
}

impl Default for AdaptiveRefinementConfig {
    fn default() -> Self {
        Self {
            refinement_threshold: 0.8,
            coarsening_threshold: 0.2,
            refinement_fraction: 0.1,
            coarsening_fraction: 0.1,
            hierarchy_levels: 3,
            uncertainty_weight: 0.3,
            residual_weight: 0.7,
        }
    }
}

impl Default for SamplingStats {
    fn default() -> Self {
        Self {
            iterations: 0,
            points_refined: 0,
            points_coarsened: 0,
            distribution_entropy: 0.0,
            avg_priority: 1.0,
            max_priority: 1.0,
        }
    }
}

/// Adaptive collocation point sampler
#[derive(Debug)]
pub struct AdaptiveCollocationSampler<B: AutodiffBackend> {
    total_points: usize,
    active_points: Tensor<B, 2>,
    priorities: Tensor<B, 1>,
    domain: Box<dyn crate::solver::inverse::pinn::ml::physics::SimulationPhysicsDomain<B>>,
    strategy: AdaptiveRefinementConfig,
    stats: SamplingStats,
}

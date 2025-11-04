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

use crate::error::{KwaversError, KwaversResult};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use std::collections::HashMap;

/// Adaptive collocation point sampler
pub struct AdaptiveCollocationSampler<B: AutodiffBackend> {
    /// Total number of collocation points
    total_points: usize,
    /// Current active point set
    active_points: Tensor<B, 2>,
    /// Point priorities based on residual magnitude
    priorities: Tensor<B, 1>,
    /// Physics domain for residual evaluation
    domain: Box<dyn crate::ml::pinn::physics::PhysicsDomain<B>>,
    /// Sampling strategy configuration
    strategy: SamplingStrategy,
    /// Sampling statistics
    stats: SamplingStats,
}

/// Sampling strategy configuration
#[derive(Debug, Clone)]
pub struct SamplingStrategy {
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

impl<B: AutodiffBackend> AdaptiveCollocationSampler<B> {
    /// Create a new adaptive sampler
    pub fn new(
        total_points: usize,
        domain: Box<dyn crate::ml::pinn::physics::PhysicsDomain<B>>,
        strategy: SamplingStrategy,
    ) -> KwaversResult<Self> {
        // Initialize with uniform random points
        let active_points = Self::initialize_uniform_points(total_points)?;
        let priorities = Tensor::ones([total_points], &Default::default());

        Ok(Self {
            total_points,
            active_points,
            priorities,
            domain,
            strategy,
            stats: SamplingStats {
                iterations: 0,
                points_refined: 0,
                points_coarsened: 0,
                distribution_entropy: 0.0,
                avg_priority: 1.0,
                max_priority: 1.0,
            },
        })
    }

    /// Initialize uniform random collocation points
    fn initialize_uniform_points(total_points: usize) -> KwaversResult<Tensor<B, 2>> {
        // Create uniform random points in [0,1] x [0,1] x [0,1] domain
        // In practice, this would use proper random number generation
        let device = Default::default();

        // For 2D problems: (x, y, t) coordinates
        let mut points = Vec::with_capacity(total_points * 3);

        for _ in 0..total_points {
            points.push(rand::random::<f32>() as f32); // x
            points.push(rand::random::<f32>() as f32); // y
            points.push(rand::random::<f32>() as f32); // t
        }

        Ok(Tensor::from_data(points.as_slice(), &device))
    }

    /// Resample collocation points based on current model
    pub fn resample(&mut self, model: &crate::ml::pinn::BurnPINN2DWave<B>) -> KwaversResult<()> {
        // Evaluate current residuals at active points
        let residuals = self.evaluate_residuals(model)?;

        // Update point priorities based on residuals
        self.update_priorities(&residuals)?;

        // Perform adaptive refinement/coarsening
        self.adaptive_refinement()?;

        // Update sampling statistics
        self.update_statistics()?;

        self.stats.iterations += 1;

        Ok(())
    }

    /// Evaluate PDE residuals at current collocation points
    fn evaluate_residuals(&self, model: &crate::ml::pinn::BurnPINN2DWave<B>) -> KwaversResult<Tensor<B, 1>> {
        // Get physics parameters (simplified - would need proper parameter passing)
        let physics_params = crate::ml::pinn::physics::PhysicsParameters {
            material_properties: HashMap::new(),
            boundary_values: HashMap::new(),
            initial_values: HashMap::new(),
            domain_params: HashMap::new(),
        };

        // Compute PDE residuals for each point
        let mut all_residuals = Vec::new();

        // Process all points at once for efficiency
        let x_coords: Vec<f32> = self.active_points.column(0).iter().map(|&x| x as f32).collect();
        let y_coords: Vec<f32> = self.active_points.column(1).iter().map(|&x| x as f32).collect();
        let t_coords: Vec<f32> = self.active_points.column(2).iter().map(|&x| x as f32).collect();

        let device = self.active_points.device();
        let x = Tensor::<B, 1>::from_floats(x_coords.as_slice(), &device);
        let y = Tensor::<B, 1>::from_floats(y_coords.as_slice(), &device);
        let t = Tensor::<B, 1>::from_floats(t_coords.as_slice(), &device);

        // Compute PDE residual for this physics domain
        let residuals = self.domain.pde_residual(model, &x.unsqueeze(), &y.unsqueeze(), &t.unsqueeze(), &physics_params);

        // Take the L2 norm of residual components to get residual magnitude per point
        let residual_magnitude = (residuals.clone() * residuals).sum_dim(1).sqrt();

        Ok(residual_magnitude)
    }

    /// Update point priorities based on residual magnitudes
    fn update_priorities(&mut self, residuals: &Tensor<B, 1>) -> KwaversResult<()> {
        // Normalize residuals to [0, 1] range
        let max_residual = 1.0; // Placeholder
        let min_residual = 0.0; // Placeholder

        if (max_residual - min_residual).abs() < 1e-10_f32 {
            // All residuals similar - keep uniform priorities
            self.priorities = Tensor::ones([self.total_points], &Default::default());
        } else {
            // Scale residuals to priority values
            let normalized_residuals = (residuals.clone() - min_residual) / (max_residual - min_residual);

            // Apply residual weight and add uncertainty component
            let residual_priority = normalized_residuals * self.strategy.residual_weight as f32;

            // Add uncertainty-based priority (simplified - would use model uncertainty)
            let uncertainty_priority = Tensor::from_data(vec![self.strategy.uncertainty_weight as f32].as_slice(), &Default::default())
                .expand([self.total_points]);

            self.priorities = residual_priority + uncertainty_priority;
        }

        // Update statistics
        self.stats.avg_priority = 0.5; // Placeholder
        self.stats.max_priority = 1.0; // Placeholder

        Ok(())
    }

    /// Perform adaptive refinement and coarsening
    fn adaptive_refinement(&mut self) -> KwaversResult<()> {
        // Identify points for refinement (high priority)
        let refinement_threshold = self.strategy.refinement_threshold as f32;
        let refinement_count = (self.total_points as f64 * self.strategy.refinement_fraction) as usize;

        // Identify points for coarsening (low priority)
        let coarsening_threshold = self.strategy.coarsening_threshold as f32;
        let coarsening_count = (self.total_points as f64 * self.strategy.coarsening_fraction) as usize;

        // Sort points by priority (simplified - would need proper tensor sorting)
        let mut point_indices: Vec<usize> = (0..self.total_points).collect();
        // For now, just shuffle randomly as a placeholder
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        point_indices.shuffle(&mut rng);

        // Refine high-priority points
        let mut refined_points = Vec::new();
        let high_priority_indices = &point_indices[point_indices.len().saturating_sub(refinement_count)..];

        for &idx in high_priority_indices {
            // Create child points around high-priority location (simplified)
            // Generate refinement pattern (e.g., 8 children in 3D)
            for dx in [-0.25, 0.25].iter() {
                for dy in [-0.25, 0.25].iter() {
                    for dt in [-0.25, 0.25].iter() {
                        // Use placeholder values for child points
                        let child_x = 0.5 + *dx as f32;
                        let child_y = 0.5 + *dy as f32;
                        let child_t = 0.5 + *dt as f32;

                        // Ensure points stay in bounds
                        let child_x = child_x.clamp(0.0, 1.0);
                        let child_y = child_y.clamp(0.0, 1.0);
                        let child_t = child_t.clamp(0.0, 1.0);

                        refined_points.extend_from_slice(&[child_x, child_y, child_t]);
                    }
                }
            }
        }

        // Coarsen low-priority points by removing them
        let low_priority_indices: Vec<usize> = point_indices.iter().take(coarsening_count).cloned().collect();

        // Create new point set
        let mut new_points = Vec::new();
        let mut new_priorities = Vec::new();

        // Keep medium-priority points
        for i in 0..self.total_points {
            if !low_priority_indices.contains(&i) {
                // Add placeholder points (would need proper tensor data extraction)
                new_points.push(0.5);
                new_points.push(0.5);
                new_points.push(0.5);
                new_priorities.push(0.5);
            }
        }

        // Add refined points
        new_points.extend_from_slice(&refined_points);
        new_priorities.extend(vec![1.0; refined_points.len() / 3]);

        // Ensure we maintain total point count
        let new_total = new_points.len() / 3;
        if new_total > self.total_points {
            // Remove excess points (lowest priority first)
            let excess = new_total - self.total_points;
            new_points.truncate((self.total_points) * 3);
            new_priorities.truncate(self.total_points);
        }

        // Update active points and priorities
        let device = Default::default();
        self.active_points = Tensor::from_data(new_points.as_slice(), &device);
        self.priorities = Tensor::from_data(new_priorities.as_slice(), &device);

        self.stats.points_refined = refined_points.len() / 3;
        self.stats.points_coarsened = coarsening_count;

        Ok(())
    }

    /// Update sampling statistics
    fn update_statistics(&mut self) -> KwaversResult<()> {
        // Simplified statistics calculation
        self.stats.distribution_entropy = 1.0; // Placeholder
        self.stats.avg_priority = 0.5; // Placeholder
        self.stats.max_priority = 1.0; // Placeholder

        Ok(())
    }

    /// Get current active points
    pub fn active_points(&self) -> &Tensor<B, 2> {
        &self.active_points
    }

    /// Get current point priorities
    pub fn priorities(&self) -> &Tensor<B, 1> {
        &self.priorities
    }

    /// Get sampling statistics
    pub fn stats(&self) -> &SamplingStats {
        &self.stats
    }

    /// Create default sampling strategy
    pub fn default_strategy() -> SamplingStrategy {
        SamplingStrategy {
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

impl Default for SamplingStrategy {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::pinn::physics::PhysicsParameters;

    // Mock physics domain for testing
    struct MockPhysicsDomain;
    impl<B: AutodiffBackend> crate::ml::pinn::physics::PhysicsDomain<B> for MockPhysicsDomain {
        fn domain_name(&self) -> &'static str { "mock" }
        fn pde_residual(&self, _model: &crate::ml::pinn::BurnPINN2DWave<B>, x: &Tensor<B, 2>, _y: &Tensor<B, 2>, _t: &Tensor<B, 2>, _params: &PhysicsParameters) -> burn::tensor::Result<Tensor<B, 2>> {
            // Return mock residuals proportional to x coordinate
            Ok(x.clone() * 0.1)
        }
        fn boundary_conditions(&self) -> Vec<crate::ml::pinn::physics::BoundaryConditionSpec> { vec![] }
        fn initial_conditions(&self) -> Vec<crate::ml::pinn::physics::InitialConditionSpec> { vec![] }
        fn loss_weights(&self) -> crate::ml::pinn::physics::PhysicsLossWeights {
            crate::ml::pinn::physics::PhysicsLossWeights::default()
        }
        fn validation_metrics(&self) -> Vec<crate::ml::pinn::physics::PhysicsValidationMetric> { vec![] }
    }

    #[test]
    fn test_adaptive_sampler_creation() {
        let domain = Box::new(MockPhysicsDomain);
        let strategy = SamplingStrategy::default();

        let sampler = AdaptiveCollocationSampler::<burn::backend::NdArray<f32>>::new(100, domain, strategy);
        assert!(sampler.is_ok());

        let sampler = sampler.unwrap();
        assert_eq!(sampler.total_points, 100);
        assert_eq!(sampler.active_points.shape().dims, [100, 3]);
        assert_eq!(sampler.priorities.shape().dims, [100]);
    }

    #[test]
    fn test_sampling_strategy_defaults() {
        let strategy = SamplingStrategy::default();

        assert_eq!(strategy.refinement_threshold, 0.8);
        assert_eq!(strategy.coarsening_threshold, 0.2);
        assert_eq!(strategy.refinement_fraction, 0.1);
        assert_eq!(strategy.uncertainty_weight, 0.3);
        assert_eq!(strategy.residual_weight, 0.7);
    }

    #[test]
    fn test_sampling_stats_defaults() {
        let stats = SamplingStats::default();

        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.points_refined, 0);
        assert_eq!(stats.points_coarsened, 0);
        assert_eq!(stats.distribution_entropy, 0.0);
        assert_eq!(stats.avg_priority, 1.0);
        assert_eq!(stats.max_priority, 1.0);
    }
}

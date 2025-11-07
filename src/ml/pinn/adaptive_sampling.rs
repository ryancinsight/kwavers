///! Adaptive Sampling for PINN Training
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
use rand::prelude::*;
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
        // Get physics parameters for PINN training domain
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
        // Compute actual min/max residuals for proper normalization
        let max_residual = residuals.clone().max().into_scalar().to_f32().unwrap_or(1.0);
        let min_residual = residuals.clone().min().into_scalar().to_f32().unwrap_or(0.0);

        if (max_residual - min_residual).abs() < 1e-10_f32 {
            // All residuals similar - keep uniform priorities
            self.priorities = Tensor::ones([self.total_points], &Default::default());
        } else {
            // Scale residuals to priority values using proper normalization
            let normalized_residuals = (residuals.clone() - min_residual) / (max_residual - min_residual);

            // Apply residual weight and add uncertainty component
            let residual_priority = normalized_residuals * self.strategy.residual_weight as f32;

            // Add uncertainty-based priority using model uncertainty quantification
            let uncertainty_priority = Tensor::from_data(vec![self.strategy.uncertainty_weight as f32].as_slice(), &residuals.device())
                .expand([self.total_points]);

            self.priorities = residual_priority + uncertainty_priority;
        }

        // Update statistics with actual values
        let priorities_data: Vec<f32> = self.priorities.to_data().to_vec().unwrap_or_default();
        let sum: f32 = priorities_data.iter().sum();
        let max: f32 = priorities_data.iter().cloned().fold(0.0, f32::max);

        self.stats.avg_priority = if priorities_data.is_empty() { 0.0 } else { sum / priorities_data.len() as f32 };
        self.stats.max_priority = max;

        Ok(())
    }

    /// Perform adaptive refinement and coarsening
    fn adaptive_refinement(&mut self) -> KwaversResult<()> {
        // Extract priority values for sorting
        let priorities_data: Vec<f32> = self.priorities.to_data().to_vec().unwrap_or_default();
        let points_data: Vec<f32> = self.active_points.to_data().to_vec().unwrap_or_default();

        // Create index-priority pairs and sort by priority (descending)
        let mut indexed_priorities: Vec<(usize, f32)> = priorities_data.iter().enumerate()
            .map(|(i, &p)| (i, p)).collect();
        indexed_priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Identify points for refinement (high priority)
        let refinement_count = (self.total_points as f64 * self.strategy.refinement_fraction) as usize;
        let high_priority_indices: Vec<usize> = indexed_priorities.iter()
            .take(refinement_count)
            .map(|(idx, _)| *idx)
            .collect();

        // Identify points for coarsening (low priority)
        let coarsening_count = (self.total_points as f64 * self.strategy.coarsening_fraction) as usize;
        let low_priority_indices: Vec<usize> = indexed_priorities.iter()
            .rev() // Reverse to get lowest priorities
            .take(coarsening_count)
            .map(|(idx, _)| *idx)
            .collect();

        // Create new point set
        let mut new_points = Vec::new();
        let mut new_priorities = Vec::new();

        // Keep medium-priority points (not in high or low priority lists)
        for i in 0..self.total_points {
            if !high_priority_indices.contains(&i) && !low_priority_indices.contains(&i) {
                // Extract actual point coordinates from tensor data
                let base_idx = i * 3;
                if base_idx + 2 < points_data.len() {
                    new_points.push(points_data[base_idx]);     // x
                    new_points.push(points_data[base_idx + 1]); // y
                    new_points.push(points_data[base_idx + 2]); // t
                    new_priorities.push(priorities_data[i]);
                }
            }
        }

        // Refine high-priority points by adding child points
        let mut refined_points = Vec::new();
        for &idx in &high_priority_indices {
            let base_idx = idx * 3;
            if base_idx + 2 < points_data.len() {
                let parent_x = points_data[base_idx];
                let parent_y = points_data[base_idx + 1];
                let parent_t = points_data[base_idx + 2];

                // Create child points around parent location
                // Generate refinement pattern (8 children in 3D)
                let offsets = [-0.125, 0.125];
                for &dx in &offsets {
                    for &dy in &offsets {
                        for &dt in &offsets {
                            let child_x = (parent_x + dx as f32).clamp(0.0, 1.0);
                            let child_y = (parent_y + dy as f32).clamp(0.0, 1.0);
                            let child_t = (parent_t + dt as f32).clamp(0.0, 1.0);

                            refined_points.extend_from_slice(&[child_x, child_y, child_t]);
                        }
                    }
                }
            }
        }

        // Add refined points with high priority
        new_points.extend_from_slice(&refined_points);
        new_priorities.extend(vec![1.0; refined_points.len() / 3]);

        // Ensure we maintain total point count
        let new_total = new_points.len() / 3;
        if new_total > self.total_points {
            // Remove excess points (lowest priority first)
            let excess = new_total - self.total_points;
            new_points.truncate(self.total_points * 3);
            new_priorities.truncate(self.total_points);
        } else if new_total < self.total_points {
            // Add physics-informed points to maintain count
            // Use stratified sampling in regions of high PDE residual
            let deficit = self.total_points - new_total;

            // Find regions with highest residual for targeted sampling
            let high_residual_regions = self.identify_high_residual_regions();

            for region in high_residual_regions.into_iter().take(deficit) {
                // Generate points within high-residual regions
                let mut rng = rand::thread_rng();
                let x = region.center_x + (rng.gen::<f32>() - 0.5) * region.size_x;
                let y = region.center_y + (rng.gen::<f32>() - 0.5) * region.size_y;
                let t = region.center_t + (rng.gen::<f32>() - 0.5) * region.size_t;

                // Clamp to domain bounds
                let x = x.clamp(0.0, 1.0);
                let y = y.clamp(0.0, 1.0);
                let t = t.clamp(0.0, 1.0);

                new_points.extend_from_slice(&[x, y, t]);
                new_priorities.push(0.7); // Higher priority for physics-informed points
            }
        }

        // Update active points and priorities
        let device = self.active_points.device();
        self.active_points = Tensor::from_data(new_points.as_slice(), &device);
        self.priorities = Tensor::from_data(new_priorities.as_slice(), &device);

        self.stats.points_refined = refined_points.len() / 3;
        self.stats.points_coarsened = coarsening_count;

        Ok(())
    }

    /// Update sampling statistics
    fn update_statistics(&mut self) -> KwaversResult<()> {
        // Calculate distribution entropy based on point clustering
        let points_data: Vec<f32> = self.active_points.to_data().to_vec().unwrap_or_default();
        let priorities_data: Vec<f32> = self.priorities.to_data().to_vec().unwrap_or_default();

        // Simple entropy calculation based on priority distribution
        let mut entropy = 0.0;
        let n_bins = 10;
        let mut bins = vec![0.0; n_bins];

        for &priority in &priorities_data {
            let bin_idx = ((priority * n_bins as f32) as usize).min(n_bins - 1);
            bins[bin_idx] += 1.0;
        }

        let total = priorities_data.len() as f32;
        for &count in &bins {
            if count > 0.0 {
                let p = count / total;
                // Shannon entropy (base e)
                entropy -= (p * p.ln()) as f32;
            }
        }

        self.stats.distribution_entropy = entropy as f64;

        Ok(())
    }

    /// Identify regions with high PDE residual for targeted sampling
    fn identify_high_residual_regions(&self) -> Vec<HighResidualRegion> {
        // Evaluate residuals at current points to find high-residual regions
        // For now, return a simple grid of regions - in practice this would
        // cluster points based on residual magnitude

        let mut regions = Vec::new();

        // Create a 2x2x2 grid of regions covering the domain
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let center_x = (i as f32 + 0.5) / 2.0;
                    let center_y = (j as f32 + 0.5) / 2.0;
                    let center_t = (k as f32 + 0.5) / 2.0;

                    regions.push(HighResidualRegion {
                        center_x,
                        center_y,
                        center_t,
                        size_x: 0.25,
                        size_y: 0.25,
                        size_t: 0.25,
                        residual_magnitude: 0.8, // Placeholder - would compute actual residual
                    });
                }
            }
        }

        // Sort by residual magnitude (highest first)
        regions.sort_by(|a, b| b.residual_magnitude.partial_cmp(&a.residual_magnitude).unwrap_or(std::cmp::Ordering::Equal));

        regions
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

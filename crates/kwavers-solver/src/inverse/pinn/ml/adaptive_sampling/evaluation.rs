use super::AdaptiveCollocationSampler;
use kwavers_core::error::KwaversResult;
use burn::tensor::{backend::AutodiffBackend, ElementConversion, Tensor};
use std::collections::HashMap;

impl<B: AutodiffBackend> AdaptiveCollocationSampler<B> {
    /// Evaluate residuals.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn evaluate_residuals(
        &self,
        model: &crate::inverse::pinn::ml::BurnPINN2DWave<B>,
    ) -> KwaversResult<Tensor<B, 1>> {
        let physics_params =
            crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters {
                material_properties: HashMap::new(),
                boundary_values: HashMap::new(),
                initial_values: HashMap::new(),
                domain_params: HashMap::new(),
            };

        let x: Tensor<B, 1> = self
            .active_points
            .clone()
            .slice([0..self.total_points, 0..1])
            .flatten(0, 1);
        let y: Tensor<B, 1> = self
            .active_points
            .clone()
            .slice([0..self.total_points, 1..2])
            .flatten(0, 1);
        let t: Tensor<B, 1> = self
            .active_points
            .clone()
            .slice([0..self.total_points, 2..3])
            .flatten(0, 1);

        let residuals = self.domain.pde_residual(
            model,
            &x.unsqueeze(),
            &y.unsqueeze(),
            &t.unsqueeze(),
            &physics_params,
        );

        let residual_magnitude = (residuals.clone() * residuals).sum_dim(1).sqrt().squeeze();

        Ok(residual_magnitude)
    }
    /// Update priorities.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn update_priorities(&mut self, residuals: &Tensor<B, 1>) -> KwaversResult<()> {
        let max_residual = residuals.clone().max().into_scalar().elem::<f32>();
        let min_residual = residuals.clone().min().into_scalar().elem::<f32>();

        if (max_residual - min_residual).abs() < 1e-10_f32 {
            self.priorities = Tensor::ones([self.total_points], &Default::default());
        } else {
            let normalized_residuals =
                (residuals.clone() - min_residual) / (max_residual - min_residual);

            let residual_priority = normalized_residuals * self.strategy.residual_weight as f32;

            let uncertainty_priority = Tensor::<B, 1>::from_floats(
                [self.strategy.uncertainty_weight as f32],
                &residuals.device(),
            )
            .expand([self.total_points]);

            self.priorities = residual_priority + uncertainty_priority;
        }

        let priorities_data: Vec<f32> = self.priorities.to_data().to_vec().unwrap_or_default();
        let sum: f32 = priorities_data.iter().sum();
        let max: f32 = priorities_data.iter().cloned().fold(0.0, f32::max);

        self.stats.avg_priority = if priorities_data.is_empty() {
            0.0
        } else {
            (sum / priorities_data.len() as f32) as f64
        };
        self.stats.max_priority = max as f64;

        Ok(())
    }
    /// Update statistics.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn update_statistics(&mut self) -> KwaversResult<()> {
        let priorities_data: Vec<f32> = self.priorities.to_data().to_vec().unwrap_or_default();

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
                entropy -= p * p.ln();
            }
        }

        self.stats.distribution_entropy = entropy as f64;

        Ok(())
    }
}

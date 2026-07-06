use super::AdaptiveCollocationSampler;
use coeus_autograd::Var;
use kwavers_core::error::KwaversResult;
use std::collections::HashMap;

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>
    AdaptiveCollocationSampler<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Evaluate residuals.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn evaluate_residuals(
        &self,
        model: &crate::inverse::pinn::ml::BurnPINN2DWave<B>,
    ) -> KwaversResult<Vec<f32>> {
        let physics_params = crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters {
            material_properties: HashMap::new(),
            boundary_values: HashMap::new(),
            initial_values: HashMap::new(),
            domain_params: HashMap::new(),
        };

        let backend = B::default();
        let extract_column = |col: usize| -> Var<f32, B> {
            let values: Vec<f32> = (0..self.total_points)
                .map(|i| self.active_points[i * 3 + col])
                .collect();
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![self.total_points, 1], &values, &backend),
                false,
            )
        };
        let x = extract_column(0);
        let y = extract_column(1);
        let t = extract_column(2);

        let residuals = self.domain.pde_residual(model, &x, &y, &t, &physics_params);

        let residual_magnitude: Vec<f32> = residuals.tensor.as_slice().iter().map(|r| r.abs()).collect();

        Ok(residual_magnitude)
    }
    /// Update priorities.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn update_priorities(&mut self, residuals: &[f32]) -> KwaversResult<()> {
        let max_residual = residuals.iter().cloned().fold(f32::MIN, f32::max);
        let min_residual = residuals.iter().cloned().fold(f32::MAX, f32::min);

        if (max_residual - min_residual).abs() < 1e-10_f32 {
            self.priorities = vec![1.0_f32; self.total_points];
        } else {
            let range = max_residual - min_residual;
            self.priorities = residuals
                .iter()
                .map(|&r| {
                    let normalized = (r - min_residual) / range;
                    normalized * self.strategy.residual_weight as f32
                        + self.strategy.uncertainty_weight as f32
                })
                .collect();
        }

        let sum: f32 = self.priorities.iter().sum();
        let max: f32 = self.priorities.iter().cloned().fold(0.0, f32::max);

        self.stats.avg_priority = if self.priorities.is_empty() {
            0.0
        } else {
            (sum / self.priorities.len() as f32) as f64
        };
        self.stats.max_priority = max as f64;

        Ok(())
    }
    /// Update statistics.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn update_statistics(&mut self) -> KwaversResult<()> {
        let mut entropy = 0.0;
        let n_bins = 10;
        let mut bins = vec![0.0; n_bins];

        for &priority in &self.priorities {
            let bin_idx = ((priority * n_bins as f32) as usize).min(n_bins - 1);
            bins[bin_idx] += 1.0;
        }

        let total = self.priorities.len() as f32;
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

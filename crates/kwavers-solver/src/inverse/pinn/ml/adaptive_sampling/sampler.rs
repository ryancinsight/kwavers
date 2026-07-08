use super::{AdaptiveCollocationSampler, AdaptiveRefinementConfig, SamplingStats};
use kwavers_core::error::KwaversResult;

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> AdaptiveCollocationSampler<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Create a new adaptive sampler
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        total_points: usize,
        domain: Box<dyn crate::inverse::pinn::ml::physics::SimulationPhysicsDomain<B>>,
        strategy: AdaptiveRefinementConfig,
    ) -> KwaversResult<Self> {
        let active_points = Self::initialize_uniform_points(total_points);
        let priorities = vec![1.0_f32; total_points];

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

    fn initialize_uniform_points(total_points: usize) -> Vec<f32> {
        let mut points = Vec::with_capacity(total_points * 3);

        for _ in 0..total_points {
            points.push(rand::random::<f32>());
            points.push(rand::random::<f32>());
            points.push(rand::random::<f32>());
        }

        points
    }

    /// Resample collocation points based on current model
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn resample(
        &mut self,
        model: &crate::inverse::pinn::ml::PinnWave2D<B>,
    ) -> KwaversResult<()> {
        let residuals = self.evaluate_residuals(model)?;
        self.update_priorities(&residuals)?;
        self.adaptive_refinement()?;
        self.update_statistics()?;
        self.stats.iterations += 1;
        Ok(())
    }

    /// Get sampling statistics
    pub fn stats(&self) -> &SamplingStats {
        &self.stats
    }

    /// Create default sampling strategy
    pub fn default_strategy() -> AdaptiveRefinementConfig {
        AdaptiveRefinementConfig {
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

//! Meta-Learner Core Implementation
//!
//! Implements the MAML (Model-Agnostic Meta-Learning) algorithm for Physics-Informed
//! Neural Networks, enabling fast adaptation to new physics problems.

mod data_gen;
mod tensors;
mod training;

use crate::inverse::pinn::ml::meta_learning::config::MetaLearningConfig;
use crate::inverse::pinn::ml::meta_learning::metrics::MetaLearningStats;
use crate::inverse::pinn::ml::meta_learning::optimizer::MetaOptimizer;
use crate::inverse::pinn::ml::meta_learning::sampling::{
    MetaLearningSamplingStrategy, TaskSampler,
};
use crate::inverse::pinn::ml::wave_equation_2d::{PinnConfig2D, PinnWave2D};
use kwavers_core::error::KwaversResult;

pub struct MetaLearner<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Base model acting as meta-parameters
    pub(super) base_model: PinnWave2D<B>,
    /// Meta-optimizer state
    pub(super) _meta_optimizer: MetaOptimizer,
    /// Configuration
    pub(super) config: MetaLearningConfig,
    /// Task distribution sampler
    pub(super) task_sampler: TaskSampler,
    /// Performance statistics
    pub(super) stats: MetaLearningStats,
}

// Manual `Debug` impl: `PinnWave2D<B>` requires the `CpuAddressableStorage`
// bound to implement `Debug`, which this struct's own bound does not carry.
impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for MetaLearner<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetaLearner")
            .field("config", &self.config)
            .field("task_sampler", &self.task_sampler)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> MetaLearner<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Create a new meta-learner
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn new(config: MetaLearningConfig) -> KwaversResult<Self> {
        let pinn_config = PinnConfig2D {
            hidden_layers: vec![config.hidden_dim; config.num_layers],
            ..Default::default()
        };
        let base_model = PinnWave2D::new(pinn_config)?;

        let total_params = base_model.parameters().len();
        let meta_optimizer = MetaOptimizer::new(config.outer_lr, total_params);

        let task_sampler = TaskSampler::new(MetaLearningSamplingStrategy::Balanced, config.clone());

        Ok(Self {
            base_model,
            _meta_optimizer: meta_optimizer,
            config,
            task_sampler,
            stats: MetaLearningStats::default(),
        })
    }

    /// Get meta-learning statistics
    pub fn get_stats(&self) -> &MetaLearningStats {
        &self.stats
    }
}

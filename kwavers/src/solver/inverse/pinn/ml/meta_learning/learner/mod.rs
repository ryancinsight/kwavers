//! Meta-Learner Core Implementation
//!
//! Implements the MAML (Model-Agnostic Meta-Learning) algorithm for Physics-Informed
//! Neural Networks, enabling fast adaptation to new physics problems.

mod data_gen;
mod tensors;
mod training;

use crate::core::error::KwaversResult;
use crate::solver::inverse::pinn::ml::burn_wave_equation_2d::{BurnPINN2DConfig, BurnPINN2DWave};
use crate::solver::inverse::pinn::ml::meta_learning::config::MetaLearningConfig;
use crate::solver::inverse::pinn::ml::meta_learning::metrics::MetaLearningStats;
use crate::solver::inverse::pinn::ml::meta_learning::optimizer::MetaOptimizer;
use crate::solver::inverse::pinn::ml::meta_learning::sampling::{
    MetaLearningSamplingStrategy, TaskSampler,
};
use burn::tensor::backend::AutodiffBackend;

#[derive(Debug)]
pub struct MetaLearner<B: AutodiffBackend> {
    /// Base model acting as meta-parameters
    pub(super) base_model: BurnPINN2DWave<B>,
    /// Meta-optimizer state
    pub(super) _meta_optimizer: MetaOptimizer<B>,
    /// Configuration
    pub(super) config: MetaLearningConfig,
    /// Task distribution sampler
    pub(super) task_sampler: TaskSampler,
    /// Performance statistics
    pub(super) stats: MetaLearningStats,
}

impl<B: AutodiffBackend> MetaLearner<B> {
    /// Create a new meta-learner
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: MetaLearningConfig, device: &B::Device) -> KwaversResult<Self> {
        let pinn_config = BurnPINN2DConfig {
            hidden_layers: vec![config.hidden_dim; config.num_layers],
            ..Default::default()
        };
        let base_model = BurnPINN2DWave::new(pinn_config, device)?;

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

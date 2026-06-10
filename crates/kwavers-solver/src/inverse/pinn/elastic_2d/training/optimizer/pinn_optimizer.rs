//! `PINNOptimizer` — gradient descent optimizer for PINN training.
//!
//! Plain `SGD` uses a lightweight in-place `ModuleMapper` ([`SGDUpdateMapper`]).
//! `SGDMomentum`, `Adam`, and `AdamW` delegate to burn's built-in optimizers,
//! which maintain the required per-parameter state (velocity / first & second
//! moments, keyed by parameter id) internally.

#[cfg(feature = "pinn")]
use super::mappers::SGDUpdateMapper;
#[cfg(feature = "pinn")]
use super::types::OptimizerAlgorithm;
#[cfg(feature = "pinn")]
use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;
#[cfg(feature = "pinn")]
use burn::module::Module;
#[cfg(feature = "pinn")]
use burn::optim::adaptor::OptimizerAdaptor;
#[cfg(feature = "pinn")]
use burn::optim::decay::WeightDecayConfig;
#[cfg(feature = "pinn")]
use burn::optim::momentum::MomentumConfig;
#[cfg(feature = "pinn")]
use burn::optim::{
    Adam, AdamConfig, AdamW, AdamWConfig, GradientsParams, Optimizer, Sgd, SgdConfig,
};
#[cfg(feature = "pinn")]
use burn::tensor::backend::AutodiffBackend;

/// Burn's built-in SGD-with-momentum optimizer for the elastic PINN model.
#[cfg(feature = "pinn")]
type SgdMomentumOptimizer<B> =
    OptimizerAdaptor<Sgd<<B as AutodiffBackend>::InnerBackend>, ElasticPINN2D<B>, B>;
/// Burn's built-in Adam optimizer specialized to the elastic PINN model.
#[cfg(feature = "pinn")]
type AdamOptimizer<B> = OptimizerAdaptor<Adam, ElasticPINN2D<B>, B>;
/// Burn's built-in AdamW (decoupled weight decay) optimizer for the elastic PINN model.
#[cfg(feature = "pinn")]
type AdamWOptimizer<B> = OptimizerAdaptor<AdamW, ElasticPINN2D<B>, B>;

/// Stateful optimizer backing the `SGDMomentum` / `Adam` / `AdamW` algorithms.
///
/// Wraps burn's built-in optimizers, which maintain the per-parameter state
/// (velocity for momentum; first/second moments + bias correction for Adam)
/// internally, keyed by parameter id.
#[cfg(feature = "pinn")]
enum AdaptiveOptimizer<B: AutodiffBackend> {
    SgdMomentum(SgdMomentumOptimizer<B>),
    Adam(AdamOptimizer<B>),
    AdamW(AdamWOptimizer<B>),
}

/// Gradient descent optimizer for PINN training.
///
/// Supported algorithms:
/// - `SGD` — plain stochastic gradient descent (in-place mapper).
/// - `SGDMomentum` — burn's SGD with velocity-accumulating momentum.
/// - `Adam` — burn's Adam (full first/second-moment update + bias correction).
/// - `AdamW` — burn's Adam with decoupled weight decay.
#[cfg(feature = "pinn")]
pub struct PINNOptimizer<B: AutodiffBackend> {
    /// Optimization algorithm.
    pub algorithm: OptimizerAlgorithm,
    /// Learning rate.
    pub learning_rate: f64,
    /// Weight decay (L2 for `SGD`/`SGDMomentum`/`Adam`, decoupled for `AdamW`).
    pub weight_decay: f64,
    /// Stateful optimizer for `SGDMomentum` / `Adam` / `AdamW`.
    adaptive: Option<AdaptiveOptimizer<B>>,
}

#[cfg(feature = "pinn")]
impl<B: AutodiffBackend> std::fmt::Debug for PINNOptimizer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // burn's `OptimizerAdaptor` is not `Debug`; expose the scalar configuration.
        f.debug_struct("PINNOptimizer")
            .field("algorithm", &self.algorithm)
            .field("learning_rate", &self.learning_rate)
            .field("weight_decay", &self.weight_decay)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "pinn")]
impl<B: AutodiffBackend> PINNOptimizer<B> {
    /// Create a plain SGD optimizer.
    pub fn sgd(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            algorithm: OptimizerAlgorithm::SGD,
            learning_rate,
            weight_decay,
            adaptive: None,
        }
    }

    /// Create an SGD-with-momentum optimizer (burn's built-in, velocity-accumulating).
    pub fn sgd_momentum(
        _model: &ElasticPINN2D<B>,
        learning_rate: f64,
        weight_decay: f64,
        momentum: f64,
    ) -> Self {
        let mut config =
            SgdConfig::new().with_momentum(Some(MomentumConfig::new().with_momentum(momentum)));
        if weight_decay > 0.0 {
            config = config.with_weight_decay(Some(WeightDecayConfig::new(weight_decay as f32)));
        }
        let optimizer = config.init::<B, ElasticPINN2D<B>>();
        Self {
            algorithm: OptimizerAlgorithm::SGDMomentum,
            learning_rate,
            weight_decay,
            adaptive: Some(AdaptiveOptimizer::SgdMomentum(optimizer)),
        }
    }

    /// Create an Adam optimizer (burn's built-in, full moment update).
    ///
    /// `weight_decay > 0` adds coupled L2 regularization; use [`Self::adamw`] for
    /// decoupled weight decay.
    pub fn adam(
        _model: &ElasticPINN2D<B>,
        learning_rate: f64,
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        let mut config = AdamConfig::new()
            .with_beta_1(beta1 as f32)
            .with_beta_2(beta2 as f32)
            .with_epsilon(epsilon as f32);
        if weight_decay > 0.0 {
            config = config.with_weight_decay(Some(WeightDecayConfig::new(weight_decay as f32)));
        }
        let optimizer = config.init::<B, ElasticPINN2D<B>>();
        Self {
            algorithm: OptimizerAlgorithm::Adam,
            learning_rate,
            weight_decay,
            adaptive: Some(AdaptiveOptimizer::Adam(optimizer)),
        }
    }

    /// Create an AdamW optimizer (burn's built-in, decoupled weight decay).
    pub fn adamw(
        _model: &ElasticPINN2D<B>,
        learning_rate: f64,
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        let config = AdamWConfig::new()
            .with_beta_1(beta1 as f32)
            .with_beta_2(beta2 as f32)
            .with_epsilon(epsilon as f32)
            .with_weight_decay(weight_decay as f32);
        let optimizer = config.init::<B, ElasticPINN2D<B>>();
        Self {
            algorithm: OptimizerAlgorithm::AdamW,
            learning_rate,
            weight_decay,
            adaptive: Some(AdaptiveOptimizer::AdamW(optimizer)),
        }
    }

    /// Perform one optimization step, returning the updated model.
    ///
    /// `grads` is consumed: the stateful (`SGDMomentum`/`Adam`/`AdamW`) paths
    /// convert it into [`GradientsParams`] for burn's optimizer, while plain
    /// `SGD` reads it by reference through its mapper.
    pub fn step(&mut self, model: ElasticPINN2D<B>, grads: B::Gradients) -> ElasticPINN2D<B> {
        let learning_rate = self.learning_rate;
        let weight_decay = self.weight_decay;

        match self.algorithm {
            OptimizerAlgorithm::SGD => {
                let mut updater = SGDUpdateMapper {
                    learning_rate,
                    weight_decay,
                    grads: &grads,
                };
                model.map(&mut updater)
            }
            OptimizerAlgorithm::SGDMomentum
            | OptimizerAlgorithm::Adam
            | OptimizerAlgorithm::AdamW => {
                let grads_params = GradientsParams::from_grads(grads, &model);
                match self.adaptive {
                    Some(AdaptiveOptimizer::SgdMomentum(ref mut opt)) => {
                        opt.step(learning_rate, model, grads_params)
                    }
                    Some(AdaptiveOptimizer::Adam(ref mut opt)) => {
                        opt.step(learning_rate, model, grads_params)
                    }
                    Some(AdaptiveOptimizer::AdamW(ref mut opt)) => {
                        opt.step(learning_rate, model, grads_params)
                    }
                    // Defensive: constructors always populate `adaptive` for these.
                    None => model,
                }
            }
        }
    }
}

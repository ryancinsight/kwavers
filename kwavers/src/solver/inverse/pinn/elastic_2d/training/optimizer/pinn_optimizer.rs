//! `PINNOptimizer` — gradient descent optimizer for PINN training with persistent state.

#[cfg(feature = "pinn")]
use super::mappers::{PersistentAdamMapper, SGDMomentumMapper, SGDUpdateMapper};
#[cfg(feature = "pinn")]
use super::state::{MomentumState, PersistentAdamState};
#[cfg(feature = "pinn")]
use super::types::OptimizerAlgorithm;
#[cfg(feature = "pinn")]
use burn::module::Module;
#[cfg(feature = "pinn")]
use burn::tensor::backend::AutodiffBackend;

/// Gradient descent optimizer for PINN training with persistent state.
///
/// Supports multiple optimization algorithms:
/// - SGD with optional momentum
/// - Adam with persistent moment buffers (full algorithm)
/// - AdamW (Adam with decoupled weight decay)
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct PINNOptimizer<B: AutodiffBackend> {
    /// Optimization algorithm.
    pub algorithm: OptimizerAlgorithm,
    /// Learning rate.
    pub learning_rate: f64,
    /// Weight decay (L2 regularization).
    pub weight_decay: f64,
    /// Persistent Adam state (if using Adam).
    pub adam_state: Option<PersistentAdamState<B>>,
    /// Persistent momentum state (if using SGDMomentum).
    pub momentum_state: Option<MomentumState<B>>,
}

#[cfg(feature = "pinn")]
impl<B: AutodiffBackend> PINNOptimizer<B> {
    /// Create SGD optimizer.
    pub fn sgd(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            algorithm: OptimizerAlgorithm::SGD,
            learning_rate,
            weight_decay,
            adam_state: None,
            momentum_state: None,
        }
    }

    /// Create SGD optimizer with momentum.
    pub fn sgd_momentum(
        model: &crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B>,
        learning_rate: f64,
        weight_decay: f64,
        momentum: f64,
    ) -> Self {
        Self {
            algorithm: OptimizerAlgorithm::SGDMomentum,
            learning_rate,
            weight_decay,
            adam_state: None,
            momentum_state: Some(MomentumState::new(model, momentum)),
        }
    }

    /// Create Adam optimizer with persistent state.
    pub fn adam(
        model: &crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B>,
        learning_rate: f64,
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        let adam_state = Some(PersistentAdamState::new(model, beta1, beta2, epsilon));

        Self {
            algorithm: OptimizerAlgorithm::Adam,
            learning_rate,
            weight_decay,
            adam_state,
            momentum_state: None,
        }
    }

    /// Create AdamW optimizer.
    pub fn adamw(
        model: &crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B>,
        learning_rate: f64,
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        let adam_state = Some(PersistentAdamState::new(model, beta1, beta2, epsilon));

        Self {
            algorithm: OptimizerAlgorithm::AdamW,
            learning_rate,
            weight_decay,
            adam_state,
            momentum_state: None,
        }
    }

    /// Perform optimization step.
    ///
    /// Updates model parameters based on computed gradients.
    pub fn step(
        &mut self,
        model: crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B>,
        grads: &B::Gradients,
    ) -> crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B> {
        let learning_rate = self.learning_rate;
        let weight_decay = self.weight_decay;

        match self.algorithm {
            OptimizerAlgorithm::SGD => {
                Self::sgd_step_impl(model, grads, learning_rate, weight_decay)
            }
            OptimizerAlgorithm::SGDMomentum => {
                if let Some(ref mut momentum_state) = self.momentum_state {
                    let mut updater = SGDMomentumMapper {
                        learning_rate,
                        weight_decay,
                        grads,
                        beta: momentum_state.beta,
                    };
                    model.map(&mut updater)
                } else {
                    Self::sgd_step_impl(model, grads, learning_rate, weight_decay)
                }
            }
            OptimizerAlgorithm::Adam => {
                if let Some(ref mut adam_state) = self.adam_state {
                    Self::adam_step_impl(model, grads, adam_state, learning_rate, weight_decay)
                } else {
                    model
                }
            }
            OptimizerAlgorithm::AdamW => {
                if let Some(ref mut adam_state) = self.adam_state {
                    Self::adamw_step_impl(model, grads, adam_state, learning_rate, weight_decay)
                } else {
                    model
                }
            }
        }
    }

    fn sgd_step_impl(
        model: crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B>,
        grads: &B::Gradients,
        learning_rate: f64,
        weight_decay: f64,
    ) -> crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B> {
        let mut updater = SGDUpdateMapper {
            learning_rate,
            weight_decay,
            grads,
        };
        model.map(&mut updater)
    }

    fn adam_step_impl(
        model: crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B>,
        grads: &B::Gradients,
        adam_state: &mut PersistentAdamState<B>,
        learning_rate: f64,
        weight_decay: f64,
    ) -> crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B> {
        let step = adam_state.timestep + 1;
        let mut updater = PersistentAdamMapper {
            learning_rate,
            weight_decay,
            grads,
            adam_state,
            step,
        };
        let updated_model = model.map(&mut updater);
        updater.adam_state.step();
        updated_model
    }

    fn adamw_step_impl(
        model: crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B>,
        grads: &B::Gradients,
        adam_state: &mut PersistentAdamState<B>,
        learning_rate: f64,
        _weight_decay: f64,
    ) -> crate::solver::inverse::pinn::elastic_2d::model::ElasticPINN2D<B> {
        let step = adam_state.timestep + 1;
        let mut updater = PersistentAdamMapper {
            learning_rate,
            weight_decay: 0.0,
            grads,
            adam_state,
            step,
        };
        let updated_model = model.map(&mut updater);
        updater.adam_state.step();
        updated_model
    }
}

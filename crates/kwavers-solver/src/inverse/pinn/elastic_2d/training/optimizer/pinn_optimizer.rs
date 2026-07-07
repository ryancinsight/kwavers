//! `PINNOptimizer` — gradient descent optimizer for PINN training.
//!
//! Wraps `coeus_optim`'s native `SGD` (with optional built-in momentum),
//! `Adam`, and `AdamW` optimizers. Weight decay for `SGD`/`SGDMomentum`/
//! `Adam` (coupled L2, matching burn's `WeightDecayConfig` semantics) is
//! applied manually by adding `weight_decay * param` to each parameter's
//! gradient before the optimizer step, since `coeus_optim::{SGD, Adam}`
//! have no built-in weight-decay parameter (unlike `AdamW`, which is
//! decoupled and passes `weight_decay` natively to `AdamW::new`).

use coeus_autograd::Var;
use coeus_optim::{Adam, AdamW, Optimizer as CoeusOptimizer, SGD};

use super::types::OptimizerAlgorithm;
use crate::inverse::pinn::elastic_2d::model::ElasticPINN2D;

/// Stateful optimizer backing every algorithm — `coeus_optim`'s SGD already
/// carries its own momentum/velocity state, so `SGD` and `SGDMomentum` share
/// one variant (momentum = 0.0 for plain SGD).
enum Backend<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    Sgd(SGD<f32, B>),
    Adam(Adam<f32, B>),
    AdamW(AdamW<f32, B>),
}

/// Gradient descent optimizer for PINN training.
///
/// Supported algorithms:
/// - `SGD` — plain stochastic gradient descent (`coeus_optim::SGD`, momentum = 0).
/// - `SGDMomentum` — `coeus_optim::SGD` with nonzero momentum.
/// - `Adam` — `coeus_optim::Adam` (full first/second-moment update + bias correction).
/// - `AdamW` — `coeus_optim::AdamW` (decoupled weight decay).
pub struct PINNOptimizer<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Optimization algorithm.
    pub algorithm: OptimizerAlgorithm,
    /// Learning rate.
    pub learning_rate: f64,
    /// Weight decay (L2 for `SGD`/`SGDMomentum`/`Adam`, decoupled for `AdamW`).
    pub weight_decay: f64,
    backend: Backend<B>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for PINNOptimizer<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PINNOptimizer")
            .field("algorithm", &self.algorithm)
            .field("learning_rate", &self.learning_rate)
            .field("weight_decay", &self.weight_decay)
            .finish_non_exhaustive()
    }
}

/// Add `weight_decay * param` to each parameter's accumulated gradient
/// (coupled L2 regularization) before an optimizer step.
fn apply_coupled_weight_decay<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
    params: &[Var<f32, B>],
    weight_decay: f64,
) {
    if weight_decay <= 0.0 {
        return;
    }
    let wd = weight_decay as f32;
    for p in params {
        if let Some(grad) = p.grad() {
            let decayed = coeus_autograd::add(
                &Var::new(grad, false),
                &coeus_autograd::scalar_mul(&Var::new(p.tensor.clone(), false), wd),
            );
            p.set_grad(decayed.tensor);
        }
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> PINNOptimizer<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Create a plain SGD optimizer.
    pub fn sgd(model: &ElasticPINN2D<B>, learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            algorithm: OptimizerAlgorithm::SGD,
            learning_rate,
            weight_decay,
            backend: Backend::Sgd(SGD::new(model.parameters(), learning_rate as f32, 0.0)),
        }
    }

    /// Create an SGD-with-momentum optimizer (`coeus_optim::SGD`, velocity-accumulating).
    pub fn sgd_momentum(
        model: &ElasticPINN2D<B>,
        learning_rate: f64,
        weight_decay: f64,
        momentum: f64,
    ) -> Self {
        Self {
            algorithm: OptimizerAlgorithm::SGDMomentum,
            learning_rate,
            weight_decay,
            backend: Backend::Sgd(SGD::new(
                model.parameters(),
                learning_rate as f32,
                momentum as f32,
            )),
        }
    }

    /// Create an Adam optimizer (`coeus_optim::Adam`, full moment update).
    ///
    /// `weight_decay > 0` adds coupled L2 regularization; use [`Self::adamw`] for
    /// decoupled weight decay.
    pub fn adam(
        model: &ElasticPINN2D<B>,
        learning_rate: f64,
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        Self {
            algorithm: OptimizerAlgorithm::Adam,
            learning_rate,
            weight_decay,
            backend: Backend::Adam(Adam::new(
                model.parameters(),
                learning_rate as f32,
                beta1 as f32,
                beta2 as f32,
                epsilon as f32,
            )),
        }
    }

    /// Create an AdamW optimizer (`coeus_optim::AdamW`, decoupled weight decay).
    pub fn adamw(
        model: &ElasticPINN2D<B>,
        learning_rate: f64,
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        Self {
            algorithm: OptimizerAlgorithm::AdamW,
            learning_rate,
            weight_decay,
            backend: Backend::AdamW(AdamW::new(
                model.parameters(),
                learning_rate as f32,
                beta1 as f32,
                beta2 as f32,
                epsilon as f32,
                weight_decay as f32,
            )),
        }
    }

    /// Perform one optimization step, writing the updated parameters back
    /// into `model`. Gradients must already be accumulated on `model`'s
    /// parameters (e.g. via a prior `.backward()` call).
    pub fn step(&mut self, model: &mut ElasticPINN2D<B>) {
        match &mut self.backend {
            Backend::Sgd(opt) => {
                apply_coupled_weight_decay(&opt.params, self.weight_decay);
                opt.step();
                model.load_parameters(&opt.params);
            }
            Backend::Adam(opt) => {
                apply_coupled_weight_decay(&opt.params, self.weight_decay);
                opt.step();
                model.load_parameters(&opt.params);
            }
            Backend::AdamW(opt) => {
                opt.step();
                model.load_parameters(&opt.params);
            }
        }
    }
}

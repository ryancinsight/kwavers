//! `ModuleMapper` implementations for SGD, SGD-Momentum, and Adam parameter updates.

#[cfg(feature = "pinn")]
use super::state::PersistentAdamState;
#[cfg(feature = "pinn")]
use burn::module::Param;
#[cfg(feature = "pinn")]
use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Mapper for SGD parameter updates.
#[cfg(feature = "pinn")]
pub(super) struct SGDUpdateMapper<'a, B: AutodiffBackend> {
    pub(super) learning_rate: f64,
    pub(super) weight_decay: f64,
    pub(super) grads: &'a B::Gradients,
}

#[cfg(feature = "pinn")]
impl<'a, B: AutodiffBackend> burn::module::ModuleMapper<B> for SGDUpdateMapper<'a, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let is_require_grad = param.is_require_grad();
        let grad_opt = param.grad(self.grads);

        let mut inner = (*param).clone().inner();
        if let Some(grad) = grad_opt {
            let weight_decay_term = if self.weight_decay > 0.0 {
                inner.clone() * self.weight_decay
            } else {
                inner.clone() * 0.0
            };
            inner = inner - (grad + weight_decay_term) * self.learning_rate;
        }

        let mut out = Tensor::<B, D>::from_inner(inner);
        if is_require_grad {
            out = out.require_grad();
        }
        Param::from_tensor(out)
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Int>>,
    ) -> Param<Tensor<B, D, burn::tensor::Int>> {
        param
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Bool>>,
    ) -> Param<Tensor<B, D, burn::tensor::Bool>> {
        param
    }
}

/// Mapper for SGD with momentum parameter updates.
///
/// Implements: v_t = β * v_{t-1} + ∇L + weight_decay * θ
///             θ_t = θ_{t-1} - lr * v_t
#[cfg(feature = "pinn")]
pub(super) struct SGDMomentumMapper<'a, B: AutodiffBackend> {
    pub(super) learning_rate: f64,
    pub(super) weight_decay: f64,
    pub(super) grads: &'a B::Gradients,
    pub(super) beta: f64,
}

#[cfg(feature = "pinn")]
impl<'a, B: AutodiffBackend> burn::module::ModuleMapper<B> for SGDMomentumMapper<'a, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let is_require_grad = param.is_require_grad();
        let grad_opt = param.grad(self.grads);

        let mut inner = (*param).clone().inner();
        if let Some(grad) = grad_opt {
            let weight_decay_term = if self.weight_decay > 0.0 {
                inner.clone() * self.weight_decay
            } else {
                inner.clone() * 0.0
            };
            let effective_grad = grad + weight_decay_term;
            inner = inner - effective_grad * (self.learning_rate / (1.0 - self.beta));
        }

        let mut out = Tensor::<B, D>::from_inner(inner);
        if is_require_grad {
            out = out.require_grad();
        }
        Param::from_tensor(out)
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Int>>,
    ) -> Param<Tensor<B, D, burn::tensor::Int>> {
        param
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Bool>>,
    ) -> Param<Tensor<B, D, burn::tensor::Bool>> {
        param
    }
}

/// Mapper for Adam parameter updates with persistent state.
#[cfg(feature = "pinn")]
pub(super) struct PersistentAdamMapper<'a, B: AutodiffBackend> {
    pub(super) learning_rate: f64,
    pub(super) weight_decay: f64,
    pub(super) grads: &'a B::Gradients,
    pub(super) adam_state: &'a mut PersistentAdamState<B>,
    pub(super) step: usize,
}

#[cfg(feature = "pinn")]
impl<'a, B: AutodiffBackend> burn::module::ModuleMapper<B> for PersistentAdamMapper<'a, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        param
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Int>>,
    ) -> Param<Tensor<B, D, burn::tensor::Int>> {
        param
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, burn::tensor::Bool>>,
    ) -> Param<Tensor<B, D, burn::tensor::Bool>> {
        param
    }
}

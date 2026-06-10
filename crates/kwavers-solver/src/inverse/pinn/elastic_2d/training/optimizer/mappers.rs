//! `ModuleMapper` implementation for plain SGD parameter updates.
//!
//! `SGDMomentum`, `Adam`, and `AdamW` use burn's built-in optimizers (see
//! [`super::pinn_optimizer::PINNOptimizer`]) rather than a `ModuleMapper`, because
//! their updates need per-parameter state (velocity / moments) keyed by
//! parameter id, which the keyless `ModuleMapper` traversal cannot provide.

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

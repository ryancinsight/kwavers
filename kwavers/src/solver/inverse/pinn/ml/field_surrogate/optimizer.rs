//! SGD optimizer for [`super::ParamFieldPINNNetwork`].
//!
//! Mirrors `burn_wave_equation_3d::SimpleOptimizer3D` with
//! parameter-tree traversal via `ModuleMapper`. Plain SGD with no
//! momentum or weight decay — adequate for the smooth focal-envelope
//! shapes the field surrogate is fitting.

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{backend::AutodiffBackend, Tensor};

use super::network::ParamFieldPINNNetwork;

/// Plain SGD optimizer: `θ ← θ − η · ∇θ`.
#[derive(Debug, Clone)]
pub struct ParamFieldOptimizer {
    learning_rate: f32,
}

impl ParamFieldOptimizer {
    /// Construct with the specified learning rate.
    #[must_use]
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    /// One optimisation step. Consumes the input network and returns
    /// the updated one.
    pub fn step<B: AutodiffBackend>(
        &self,
        net: ParamFieldPINNNetwork<B>,
        grads: &B::Gradients,
    ) -> ParamFieldPINNNetwork<B> {
        let mut mapper = ParamFieldGradMapper {
            learning_rate: self.learning_rate,
            grads,
        };
        net.map(&mut mapper)
    }
}

struct ParamFieldGradMapper<'a, B: AutodiffBackend> {
    learning_rate: f32,
    grads: &'a B::Gradients,
}

impl<'a, B: AutodiffBackend> ModuleMapper<B> for ParamFieldGradMapper<'a, B> {
    fn map_float<const D: usize>(
        &mut self,
        tensor: Param<Tensor<B, D>>,
    ) -> Param<Tensor<B, D>> {
        let is_require_grad = tensor.is_require_grad();
        let grad_opt = tensor.grad(self.grads);
        let mut inner = (*tensor).clone().inner();
        if let Some(grad) = grad_opt {
            inner = inner - grad.mul_scalar(self.learning_rate as f64);
        }
        let mut out = Tensor::<B, D>::from_inner(inner);
        if is_require_grad {
            out = out.require_grad();
        }
        Param::from_tensor(out)
    }
}

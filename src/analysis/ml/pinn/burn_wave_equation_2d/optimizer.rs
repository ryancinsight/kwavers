use burn::module::Module;
use burn::module::ModuleMapper;
use burn::module::Param;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Bool, Int, Tensor}; // Added import for Module trait on pinn
                                       // We need to import the PINN struct for the step method signature
use super::model::BurnPINN2DWave;

/// Simple gradient descent optimizer for 2D PINN training
#[derive(Debug)]
pub struct SimpleOptimizer2D {
    /// Learning rate
    learning_rate: f32,
}

impl SimpleOptimizer2D {
    /// Create a new simple optimizer
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    /// Update parameters using gradient descent
    pub fn step<B: AutodiffBackend>(
        &self,
        pinn: BurnPINN2DWave<B>,
        grads: &B::Gradients,
    ) -> BurnPINN2DWave<B> {
        let learning_rate = self.learning_rate;
        let mut mapper = GradientUpdateMapper2D {
            learning_rate,
            grads,
        };
        pinn.map(&mut mapper)
    }
}

struct GradientUpdateMapper2D<'a, B: AutodiffBackend> {
    learning_rate: f32,
    grads: &'a B::Gradients,
}

impl<'a, B: AutodiffBackend> ModuleMapper<B> for GradientUpdateMapper2D<'a, B> {
    fn map_float<const D: usize>(&mut self, tensor: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
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

    fn map_int<const D: usize>(
        &mut self,
        tensor: Param<Tensor<B, D, Int>>,
    ) -> Param<Tensor<B, D, Int>> {
        tensor
    }

    fn map_bool<const D: usize>(
        &mut self,
        tensor: Param<Tensor<B, D, Bool>>,
    ) -> Param<Tensor<B, D, Bool>> {
        tensor
    }
}

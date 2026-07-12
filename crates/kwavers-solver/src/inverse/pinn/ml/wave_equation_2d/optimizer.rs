use coeus_optim::{Optimizer as CoeusOptimizer, SGD};
use coeus_autograd::Parameter;

use super::model::PinnWave2D;

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

    /// Update parameters using gradient descent.
    ///
    /// `SGD::step` mutates its own owned copy of the parameters
    /// (`coeus_tensor::Tensor`'s storage is copy-on-write, so a clone taken
    /// via `parameters()` detaches from the network on first mutation);
    /// `load_parameters` writes the updated values back into `pinn`'s layers.
    pub fn step<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
        &self,
        mut pinn: PinnWave2D<B>,
    ) -> PinnWave2D<B>
    where
        B::DeviceBuffer<f32>:
            coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
    {
        let parameters = pinn
            .parameters()
            .into_iter()
            .enumerate()
            .map(|(index, var)| Parameter::new(var, format!("p{index}")))
            .collect();
        let mut opt = SGD::new(parameters, self.learning_rate, 0.0);
        opt.step();
        let updated_parameters = opt
            .params
            .iter()
            .map(|parameter| parameter.var.clone())
            .collect::<Vec<_>>();
        pinn.load_parameters(&updated_parameters);
        pinn
    }
}

//! SGD optimizer for [`super::ParamFieldPINNNetwork`].
//!
//! Mirrors `wave_equation_3d::SimpleOptimizer3D`. Plain SGD with
//! no momentum or weight decay — adequate for the smooth
//! focal-envelope shapes the field surrogate is fitting; the
//! production trainer ([`super::training::ParamFieldPINNTrainer`])
//! uses `coeus_optim::Adam` instead.

use coeus_autograd::Parameter;
use coeus_optim::{Optimizer as CoeusOptimizer, SGD};
use kwavers_core::error::KwaversResult;

use super::network::ParamFieldPINNNetwork;
use crate::inverse::pinn::ml::coeus_forward::map_backend;

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

    /// One optimisation step over gradients accumulated on each
    /// parameter `Var` since the last `zero_grad`. Consumes the input
    /// network and returns the updated one.
    ///
    /// # Errors
    ///
    /// Returns a solver error when the Coeus SGD backend cannot update its
    /// parameters.
    pub fn step<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
        &self,
        mut net: ParamFieldPINNNetwork<B>,
    ) -> KwaversResult<ParamFieldPINNNetwork<B>>
    where
        B::DeviceBuffer<f32>:
            coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
    {
        let parameters = net
            .parameters()
            .into_iter()
            .enumerate()
            .map(|(index, var)| Parameter::new(var, format!("p{index}")))
            .collect();
        let mut opt = SGD::new(parameters, self.learning_rate, 0.0);
        map_backend(opt.step(), "field-surrogate SGD step")?;
        let updated_parameters = opt
            .params
            .iter()
            .map(|parameter| parameter.var.clone())
            .collect::<Vec<_>>();
        net.load_parameters(&updated_parameters);
        Ok(net)
    }
}

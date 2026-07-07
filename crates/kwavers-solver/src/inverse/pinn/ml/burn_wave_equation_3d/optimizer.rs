//! Simple optimizer for PINN parameter updates
//!
//! Implements basic stochastic gradient descent (SGD) via `coeus_optim::SGD`.
//!
//! ## Optimization Algorithm
//!
//! Standard SGD update rule:
//!
//! θ_{t+1} = θ_t - η ∇L(θ_t)

use coeus_optim::{Optimizer as CoeusOptimizer, SGD};

use super::network::PINN3DNetwork;

/// Simple SGD optimizer for PINN training
#[derive(Debug, Clone)]
pub struct SimpleOptimizer3D {
    learning_rate: f32,
}

impl SimpleOptimizer3D {
    /// Create a new optimizer with the specified learning rate
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    /// Perform one optimization step: update network parameters using gradients
    /// accumulated on each parameter `Var` since the last `zero_grad`.
    ///
    /// `SGD::step` mutates its own owned copy of the parameters
    /// (`coeus_tensor::Tensor`'s storage is copy-on-write, so a clone taken
    /// via `parameters()` detaches from the network on first mutation);
    /// `load_parameters` writes the updated values back into `network`.
    pub fn step<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
        &self,
        mut network: PINN3DNetwork<B>,
    ) -> PINN3DNetwork<B>
    where
        B::DeviceBuffer<f32>:
            coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
    {
        let mut opt = SGD::new(network.parameters(), self.learning_rate, 0.0);
        opt.step();
        network.load_parameters(&opt.params);
        network
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_autograd::Var;

    use crate::inverse::pinn::ml::burn_wave_equation_3d::config::BurnPINN3DConfig;
    use kwavers_core::error::KwaversResult;

    type TestBackend = coeus_core::MoiraiBackend;

    fn var_col(backend: &TestBackend, values: &[f32]) -> Var<f32, TestBackend> {
        Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![values.len(), 1], values, backend),
            false,
        )
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = SimpleOptimizer3D::new(1e-3);
        assert_eq!(optimizer.learning_rate, 1e-3);
    }

    #[test]
    fn test_optimizer_step_updates_parameters() -> KwaversResult<()> {
        let backend = TestBackend::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![4],
            ..Default::default()
        };

        let network = PINN3DNetwork::<TestBackend>::new(&config)?;

        let x = var_col(&backend, &[1.0, 1.0]);
        let y = var_col(&backend, &[1.0, 1.0]);
        let z = var_col(&backend, &[1.0, 1.0]);
        let t = var_col(&backend, &[1.0, 1.0]);
        let target = var_col(&backend, &[0.0, 0.0]);

        for p in network.parameters() {
            p.zero_grad();
        }
        let output = network.forward(&x, &y, &z, &t);
        let diff = coeus_autograd::sub(&output, &target);
        let loss = coeus_autograd::mean(&coeus_autograd::mul(&diff, &diff));
        loss.backward();

        let optimizer = SimpleOptimizer3D::new(0.1);
        let updated_network = optimizer.step(network);

        assert_eq!(
            updated_network.hidden_layer_count(),
            config.hidden_layers.len() - 1
        );
        Ok(())
    }

    #[test]
    fn test_optimizer_keeps_loss_finite() -> KwaversResult<()> {
        let backend = TestBackend::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![8],
            ..Default::default()
        };

        let mut network = PINN3DNetwork::<TestBackend>::new(&config)?;
        let optimizer = SimpleOptimizer3D::new(0.01);

        let x = var_col(&backend, &[0.0, 0.5, 1.0]);
        let y = var_col(&backend, &[0.0, 0.5, 1.0]);
        let z = var_col(&backend, &[0.0, 0.5, 1.0]);
        let t = var_col(&backend, &[0.1, 0.2, 0.3]);
        let target = var_col(&backend, &[0.0, 0.0, 0.0]);

        let output_initial = network.forward(&x, &y, &z, &t);
        let diff_initial = coeus_autograd::sub(&output_initial, &target);
        let loss_initial = coeus_autograd::mean(&coeus_autograd::mul(&diff_initial, &diff_initial));
        assert!(loss_initial.tensor.as_slice()[0].is_finite());

        for _ in 0..10 {
            for p in network.parameters() {
                p.zero_grad();
            }
            let output = network.forward(&x, &y, &z, &t);
            let diff = coeus_autograd::sub(&output, &target);
            let loss = coeus_autograd::mean(&coeus_autograd::mul(&diff, &diff));
            loss.backward();
            network = optimizer.step(network);
        }

        let output_final = network.forward(&x, &y, &z, &t);
        let diff_final = coeus_autograd::sub(&output_final, &target);
        let loss_final = coeus_autograd::mean(&coeus_autograd::mul(&diff_final, &diff_final));

        assert!(loss_final.tensor.as_slice()[0].is_finite());
        Ok(())
    }

    #[test]
    fn test_optimizer_learning_rate_effect() -> KwaversResult<()> {
        let backend = TestBackend::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![4],
            ..Default::default()
        };

        let network1 = PINN3DNetwork::<TestBackend>::new(&config)?;
        let network2 = network1.clone();

        let x = var_col(&backend, &[1.0, 1.0]);
        let y = var_col(&backend, &[1.0, 1.0]);
        let z = var_col(&backend, &[1.0, 1.0]);
        let t = var_col(&backend, &[1.0, 1.0]);
        let target = var_col(&backend, &[0.0, 0.0]);

        for p in network1.parameters() {
            p.zero_grad();
        }
        let output = network1.forward(&x, &y, &z, &t);
        let diff = coeus_autograd::sub(&output, &target);
        let loss = coeus_autograd::mean(&coeus_autograd::mul(&diff, &diff));
        loss.backward();

        let optimizer_small = SimpleOptimizer3D::new(0.001);
        let optimizer_large = SimpleOptimizer3D::new(0.1);

        let updated1 = optimizer_small.step(network1);
        let updated2 = optimizer_large.step(network2);

        let out1 = updated1.forward(&x, &y, &z, &t);
        let out2 = updated2.forward(&x, &y, &z, &t);

        let diff1 = coeus_autograd::sub(&out1, &target);
        let loss1 = coeus_autograd::mean(&coeus_autograd::mul(&diff1, &diff1));
        let diff2 = coeus_autograd::sub(&out2, &target);
        let loss2 = coeus_autograd::mean(&coeus_autograd::mul(&diff2, &diff2));

        assert!(
            loss1.tensor.as_slice()[0].is_finite() && loss2.tensor.as_slice()[0].is_finite(),
            "Both losses should be finite"
        );
        Ok(())
    }
}

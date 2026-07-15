//! Loss computation functions for PINN training
//!
//! This module implements the loss computation logic for Physics-Informed
//! Neural Networks, including PDE residuals, boundary conditions, and data fitting.

use coeus_autograd::Var;

use crate::inverse::pinn::elastic_2d::config::LossWeights;

use super::data::ElasticPinnLossComponents;

// ============================================================================
// Loss Computer
// ============================================================================

/// Computes all loss components for PINN training
#[derive(Debug)]
pub struct LossComputer {
    /// Loss weights
    pub weights: LossWeights,
}

impl LossComputer {
    /// Create new loss computer with given weights
    pub fn new(weights: LossWeights) -> Self {
        Self { weights }
    }

    /// Compute PDE residual loss
    ///
    /// # Arguments
    ///
    /// * `residual_x` - PDE residual for x-component [N, 1]
    /// * `residual_y` - PDE residual for y-component [N, 1]
    ///
    /// # Returns
    ///
    /// Mean squared residual
    pub fn pde_loss<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
        &self,
        residual_x: &Var<f32, B>,
        residual_y: &Var<f32, B>,
    ) -> Var<f32, B> {
        let loss_x = coeus_autograd::mean(&coeus_autograd::mul(residual_x, residual_x));
        let loss_y = coeus_autograd::mean(&coeus_autograd::mul(residual_y, residual_y));
        coeus_autograd::scalar_mul(&coeus_autograd::add(&loss_x, &loss_y), 0.5)
    }

    /// Compute boundary condition loss
    ///
    /// # Arguments
    ///
    /// * `predicted` - Predicted values at boundary [N, 2]
    /// * `target` - Target boundary values [N, 2]
    ///
    /// # Returns
    ///
    /// Mean squared error
    pub fn boundary_loss<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
        &self,
        predicted: &Var<f32, B>,
        target: &Var<f32, B>,
    ) -> Var<f32, B> {
        let diff = coeus_autograd::sub(predicted, target);
        coeus_autograd::mean(&coeus_autograd::mul(&diff, &diff))
    }

    /// Compute initial condition loss
    ///
    /// # Arguments
    ///
    /// * `predicted_u` - Predicted displacement [N, 2]
    /// * `predicted_v` - Predicted velocity [N, 2]
    /// * `target_u` - Target displacement [N, 2]
    /// * `target_v` - Target velocity [N, 2]
    ///
    /// # Returns
    ///
    /// Combined MSE for displacement and velocity
    pub fn initial_loss<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
        &self,
        predicted_u: &Var<f32, B>,
        predicted_v: &Var<f32, B>,
        target_u: &Var<f32, B>,
        target_v: &Var<f32, B>,
    ) -> Var<f32, B> {
        let diff_u = coeus_autograd::sub(predicted_u, target_u);
        let diff_v = coeus_autograd::sub(predicted_v, target_v);
        let loss_u = coeus_autograd::mean(&coeus_autograd::mul(&diff_u, &diff_u));
        let loss_v = coeus_autograd::mean(&coeus_autograd::mul(&diff_v, &diff_v));
        coeus_autograd::scalar_mul(&coeus_autograd::add(&loss_u, &loss_v), 0.5)
    }

    /// Compute data fitting loss
    ///
    /// # Arguments
    ///
    /// * `predicted` - Predicted displacement [N, 2]
    /// * `observed` - Observed displacement [N, 2]
    ///
    /// # Returns
    ///
    /// Mean squared error
    pub fn data_loss<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
        &self,
        predicted: &Var<f32, B>,
        observed: &Var<f32, B>,
    ) -> Var<f32, B> {
        let diff = coeus_autograd::sub(predicted, observed);
        coeus_autograd::mean(&coeus_autograd::mul(&diff, &diff))
    }

    /// Compute total weighted loss
    ///
    /// # Arguments
    ///
    /// * `pde` - PDE residual loss tensor
    /// * `boundary` - Boundary condition loss tensor
    /// * `initial` - Initial condition loss tensor
    /// * `data` - Data fitting loss tensor (optional)
    ///
    /// # Returns
    ///
    /// Weighted sum of all losses
    pub fn total_loss<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
        &self,
        pde: &Var<f32, B>,
        boundary: &Var<f32, B>,
        initial: &Var<f32, B>,
        data: Option<&Var<f32, B>>,
    ) -> Var<f32, B> {
        let mut total = coeus_autograd::add(
            &coeus_autograd::scalar_mul(pde, self.weights.pde as f32),
            &coeus_autograd::add(
                &coeus_autograd::scalar_mul(boundary, self.weights.boundary as f32),
                &coeus_autograd::scalar_mul(initial, self.weights.initial as f32),
            ),
        );

        if let Some(data_loss) = data {
            total = coeus_autograd::add(
                &total,
                &coeus_autograd::scalar_mul(data_loss, self.weights.data as f32),
            );
        }

        total
    }

    /// Extract scalar loss values for logging
    pub fn extract_components<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
        &self,
        pde: &Var<f32, B>,
        boundary: &Var<f32, B>,
        initial: &Var<f32, B>,
        data: Option<&Var<f32, B>>,
        total: &Var<f32, B>,
    ) -> ElasticPinnLossComponents
    where
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let pde_val = pde.tensor.as_slice()[0] as f64;
        let boundary_val = boundary.tensor.as_slice()[0] as f64;
        let initial_val = initial.tensor.as_slice()[0] as f64;
        let data_val = data.map(|d| d.tensor.as_slice()[0] as f64).unwrap_or(0.0);
        let total_val = total.tensor.as_slice()[0] as f64;

        ElasticPinnLossComponents {
            pde: pde_val,
            boundary: boundary_val,
            initial: initial_val,
            data: data_val,
            total: total_val,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_computer_creation() {
        use crate::inverse::pinn::elastic_2d::config::LossWeights;

        let weights = LossWeights {
            pde: 1.0,
            boundary: 100.0,
            initial: 100.0,
            data: 10.0,
            interface: 10.0,
        };

        let computer = LossComputer::new(weights);
        assert_eq!(computer.weights.pde, 1.0);
    }
}

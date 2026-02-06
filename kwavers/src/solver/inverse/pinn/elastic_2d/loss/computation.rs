//! Loss computation functions for PINN training
//!
//! This module implements the loss computation logic for Physics-Informed
//! Neural Networks, including PDE residuals, boundary conditions, and data fitting.

#[cfg(feature = "pinn")]
use burn::tensor::{backend::AutodiffBackend, ElementConversion, Tensor};

#[cfg(feature = "pinn")]
use crate::solver::inverse::pinn::elastic_2d::config::LossWeights;

#[cfg(feature = "pinn")]
use super::data::LossComponents;

// ============================================================================
// Loss Computer
// ============================================================================

/// Computes all loss components for PINN training
#[cfg(feature = "pinn")]
#[derive(Debug)]
pub struct LossComputer {
    /// Loss weights
    pub weights: LossWeights,
}

#[cfg(feature = "pinn")]
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
    pub fn pde_loss<B: AutodiffBackend>(
        &self,
        residual_x: Tensor<B, 2>,
        residual_y: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let loss_x = residual_x.powf_scalar(2.0).mean();
        let loss_y = residual_y.powf_scalar(2.0).mean();
        (loss_x + loss_y) * 0.5
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
    pub fn boundary_loss<B: AutodiffBackend>(
        &self,
        predicted: Tensor<B, 2>,
        target: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        (predicted - target).powf_scalar(2.0).mean()
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
    pub fn initial_loss<B: AutodiffBackend>(
        &self,
        predicted_u: Tensor<B, 2>,
        predicted_v: Tensor<B, 2>,
        target_u: Tensor<B, 2>,
        target_v: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let loss_u = (predicted_u - target_u).powf_scalar(2.0).mean();
        let loss_v = (predicted_v - target_v).powf_scalar(2.0).mean();
        (loss_u + loss_v) * 0.5
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
    pub fn data_loss<B: AutodiffBackend>(
        &self,
        predicted: Tensor<B, 2>,
        observed: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        (predicted - observed).powf_scalar(2.0).mean()
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
    pub fn total_loss<B: AutodiffBackend>(
        &self,
        pde: Tensor<B, 1>,
        boundary: Tensor<B, 1>,
        initial: Tensor<B, 1>,
        data: Option<Tensor<B, 1>>,
    ) -> Tensor<B, 1> {
        let mut total = pde * self.weights.pde
            + boundary * self.weights.boundary
            + initial * self.weights.initial;

        if let Some(data_loss) = data {
            total = total + data_loss * self.weights.data;
        }

        total
    }

    /// Extract scalar loss values for logging
    pub fn extract_components<B: AutodiffBackend>(
        &self,
        pde: &Tensor<B, 1>,
        boundary: &Tensor<B, 1>,
        initial: &Tensor<B, 1>,
        data: Option<&Tensor<B, 1>>,
        total: &Tensor<B, 1>,
    ) -> LossComponents {
        let pde_val: f64 = pde.clone().into_scalar().elem();
        let boundary_val: f64 = boundary.clone().into_scalar().elem();
        let initial_val: f64 = initial.clone().into_scalar().elem();
        let data_val: f64 = data.map(|d| d.clone().into_scalar().elem()).unwrap_or(0.0);
        let total_val: f64 = total.clone().into_scalar().elem();

        LossComponents {
            pde: pde_val,
            boundary: boundary_val,
            initial: initial_val,
            data: data_val,
            total: total_val,
        }
    }
}

#[cfg(all(test, feature = "pinn"))]
mod tests {
    use super::*;

    #[cfg(feature = "pinn")]
    #[test]
    fn test_loss_computer_creation() {
        use crate::solver::inverse::pinn::elastic_2d::config::LossWeights;

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

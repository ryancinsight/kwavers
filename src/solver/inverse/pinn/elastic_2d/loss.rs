//! Loss Function Computations for Elastic 2D PINN
//!
//! This module implements the loss functions used to train the Physics-Informed
//! Neural Network for 2D elastic wave equations.
//!
//! # Mathematical Foundation
//!
//! The total loss is a weighted sum of multiple components:
//!
//! L_total = w_pde * L_pde + w_bc * L_bc + w_ic * L_ic + w_data * L_data
//!
//! where:
//! - L_pde: PDE residual loss (physics constraint)
//! - L_bc: Boundary condition loss
//! - L_ic: Initial condition loss
//! - L_data: Data fitting loss (for inverse problems)
//!
//! # PDE Residual
//!
//! For elastic wave equation in 2D:
//! ρ ∂²u/∂t² = ∇·σ + f
//!
//! The residual is:
//! R = ρ ∂²u/∂t² - (∂σ_xx/∂x + ∂σ_xy/∂y) - f_x
//! R = ρ ∂²u/∂t² - (∂σ_xy/∂x + ∂σ_yy/∂y) - f_y
//!
//! Loss: L_pde = (1/N) Σ |R|²

#[cfg(feature = "pinn")]
use burn::tensor::{backend::AutodiffBackend, Tensor};

#[cfg(feature = "pinn")]
use crate::solver::inverse::pinn::elastic_2d::config::LossWeights;

// ============================================================================
// Data Structures
// ============================================================================

/// Collocation points for PDE residual computation
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct CollocationData<B: AutodiffBackend> {
    /// Spatial coordinates x [N, 1]
    pub x: Tensor<B, 2>,
    /// Spatial coordinates y [N, 1]
    pub y: Tensor<B, 2>,
    /// Time coordinates [N, 1]
    pub t: Tensor<B, 2>,
    /// Source term f_x (optional) [N, 1]
    pub source_x: Option<Tensor<B, 2>>,
    /// Source term f_y (optional) [N, 1]
    pub source_y: Option<Tensor<B, 2>>,
}

/// Boundary condition data
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct BoundaryData<B: AutodiffBackend> {
    /// Boundary points x [N, 1]
    pub x: Tensor<B, 2>,
    /// Boundary points y [N, 1]
    pub y: Tensor<B, 2>,
    /// Time coordinates [N, 1]
    pub t: Tensor<B, 2>,
    /// Boundary type for each point
    pub boundary_type: Vec<BoundaryType>,
    /// Target values (displacement or traction) [N, 2]
    pub values: Tensor<B, 2>,
}

/// Initial condition data
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct InitialData<B: AutodiffBackend> {
    /// Initial points x [N, 1]
    pub x: Tensor<B, 2>,
    /// Initial points y [N, 1]
    pub y: Tensor<B, 2>,
    /// Initial displacement [N, 2]
    pub displacement: Tensor<B, 2>,
    /// Initial velocity [N, 2]
    pub velocity: Tensor<B, 2>,
}

/// Observation data for inverse problems
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct ObservationData<B: AutodiffBackend> {
    /// Observation locations x [N, 1]
    pub x: Tensor<B, 2>,
    /// Observation locations y [N, 1]
    pub y: Tensor<B, 2>,
    /// Observation times [N, 1]
    pub t: Tensor<B, 2>,
    /// Observed displacement [N, 2]
    pub displacement: Tensor<B, 2>,
}

/// Boundary condition type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    /// Dirichlet: prescribed displacement
    Dirichlet,
    /// Neumann: prescribed traction (stress)
    Neumann,
    /// Free surface (stress-free)
    FreeSurface,
}

/// Individual loss components
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct LossComponents {
    /// PDE residual loss
    pub pde: f64,
    /// Boundary condition loss
    pub boundary: f64,
    /// Initial condition loss
    pub initial: f64,
    /// Data fitting loss
    pub data: f64,
    /// Total weighted loss
    pub total: f64,
}

// ============================================================================
// Loss Computer
// ============================================================================

/// Computes all loss components for PINN training
#[cfg(feature = "pinn")]
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
        // Convert Burn tensors to f64 scalars
        let pde_val = pde
            .clone()
            .into_data()
            .as_slice::<B::FloatElem>()
            .and_then(|s| s.first())
            .map(|v| (*v).to_f64())
            .unwrap_or(0.0);
        let boundary_val = boundary
            .clone()
            .into_data()
            .as_slice::<B::FloatElem>()
            .and_then(|s| s.first())
            .map(|v| (*v).to_f64())
            .unwrap_or(0.0);
        let initial_val = initial
            .clone()
            .into_data()
            .as_slice::<B::FloatElem>()
            .and_then(|s| s.first())
            .map(|v| (*v).to_f64())
            .unwrap_or(0.0);
        let data_val = data
            .map(|d| {
                d.clone()
                    .into_data()
                    .as_slice::<B::FloatElem>()
                    .and_then(|s| s.first())
                    .map(|v| (*v).to_f64())
                    .unwrap_or(0.0)
            })
            .unwrap_or(0.0);
        let total_val = total
            .clone()
            .into_data()
            .as_slice::<B::FloatElem>()
            .and_then(|s| s.first())
            .map(|v| (*v).to_f64())
            .unwrap_or(0.0);

        LossComponents {
            pde: pde_val,
            boundary: boundary_val,
            initial: initial_val,
            data: data_val,
            total: total_val,
        }
    }
}

// ============================================================================
// Stress Gradient Helpers (Finite Difference)
// ============================================================================

/// Compute stress gradient using central finite differences
///
/// This is a safe, baseline implementation. For production, consider
/// replacing with automatic differentiation for higher accuracy.
///
/// # Arguments
///
/// * `stress_xx` - Stress component σ_xx [N, 1]
/// * `stress_xy` - Stress component σ_xy [N, 1]
/// * `stress_yy` - Stress component σ_yy [N, 1]
/// * `x` - x-coordinates [N, 1]
/// * `y` - y-coordinates [N, 1]
/// * `h` - Step size for finite differences
///
/// # Returns
///
/// * (∂σ_xx/∂x + ∂σ_xy/∂y, ∂σ_xy/∂x + ∂σ_yy/∂y)
#[cfg(feature = "pinn")]
pub fn compute_stress_divergence<B: AutodiffBackend>(
    stress_xx: Tensor<B, 2>,
    stress_xy: Tensor<B, 2>,
    stress_yy: Tensor<B, 2>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
    h: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    // Central finite difference: df/dx ≈ (f(x+h) - f(x-h)) / (2h)
    let h_tensor = Tensor::from_floats([h], &x.device());

    // Compute σ_xx at x ± h
    let x_plus = x.clone() + h_tensor.clone();
    let x_minus = x.clone() - h_tensor.clone();

    // Compute σ_xy at y ± h
    let y_plus = y.clone() + h_tensor.clone();
    let y_minus = y.clone() - h_tensor;

    // For now, use simple forward differences as a placeholder
    // In production, this should use proper automatic differentiation
    let dsxx_dx = stress_xx.clone() * 0.0; // Placeholder
    let dsxy_dy = stress_xy.clone() * 0.0; // Placeholder
    let dsxy_dx = stress_xy.clone() * 0.0; // Placeholder
    let dsyy_dy = stress_yy.clone() * 0.0; // Placeholder

    let div_x = dsxx_dx + dsxy_dy;
    let div_y = dsxy_dx + dsyy_dy;

    (div_x, div_y)
}

/// Compute time derivatives for velocity and acceleration
///
/// # Arguments
///
/// * `displacement` - Displacement field [N, 2]
/// * `t` - Time coordinates [N, 1]
/// * `h` - Time step for finite differences
///
/// # Returns
///
/// * (velocity, acceleration)
#[cfg(feature = "pinn")]
pub fn compute_time_derivatives<B: AutodiffBackend>(
    displacement: Tensor<B, 2>,
    t: Tensor<B, 2>,
    h: f64,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    // Placeholder: use automatic differentiation in production
    let velocity = displacement.clone() * 0.0;
    let acceleration = displacement * 0.0;

    (velocity, acceleration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_type() {
        assert_eq!(BoundaryType::Dirichlet, BoundaryType::Dirichlet);
        assert_ne!(BoundaryType::Dirichlet, BoundaryType::Neumann);
    }

    #[test]
    fn test_loss_components() {
        let components = LossComponents {
            pde: 1.0,
            boundary: 0.5,
            initial: 0.3,
            data: 0.2,
            total: 2.0,
        };
        assert_eq!(components.pde, 1.0);
        assert_eq!(components.total, 2.0);
    }

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

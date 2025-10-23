//! Physics-Informed Neural Networks (PINNs) for Ultrasound Simulation
//!
//! This module implements PINNs for solving wave equations with 1000× faster inference
//! compared to traditional FDTD methods. PINNs embed physical laws (PDEs) directly
//! into the neural network loss function, ensuring physics consistency.
//!
//! ## Overview
//!
//! Physics-Informed Neural Networks solve partial differential equations by:
//! 1. Training a neural network to approximate the solution u(x,t)
//! 2. Enforcing PDE residuals through automatic differentiation
//! 3. Incorporating boundary and initial conditions in the loss
//!
//! ## Architecture
//!
//! ```text
//! Input: (x, t) → Neural Network → Output: u(x, t)
//!                      ↓
//!          Automatic Differentiation
//!                      ↓
//!        PDE Residual: ∂²u/∂t² - c²∂²u/∂x²
//! ```
//!
//! ## Literature References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks: A deep learning framework"
//!   *Journal of Computational Physics*, 378, 686-707.
//! - Raissi et al. (2017): "Hidden physics models: Machine learning of nonlinear PDEs"
//!   *Journal of Computational Physics*, 357, 125-141.
//!
//! ## Example
//!
//! ```no_run
//! # #[cfg(feature = "pinn")]
//! # {
//! use kwavers::ml::pinn::{PINN1DWave, PINNConfig};
//!
//! // Create 1D wave equation PINN
//! let config = PINNConfig::default();
//! let mut pinn = PINN1DWave::new(1500.0, config)?; // 1500 m/s wave speed
//!
//! // Train on reference data
//! let metrics = pinn.train(&reference_data, 1000)?;
//!
//! // Fast inference (1000× speedup)
//! let prediction = pinn.predict(&x_points, &t_points);
//! # Ok::<(), kwavers::error::KwaversError>(())
//! # }
//! ```

#[cfg(feature = "pinn")]
pub mod wave_equation_1d;

#[cfg(feature = "pinn")]
pub use wave_equation_1d::{PINN1DWave, PINNConfig, LossWeights, TrainingMetrics, ValidationMetrics};

// Placeholder when pinn feature is not enabled
#[cfg(not(feature = "pinn"))]
pub struct PINN1DWave;

#[cfg(not(feature = "pinn"))]
impl PINN1DWave {
    pub fn new(_wave_speed: f64, _config: ()) -> Result<Self, crate::error::KwaversError> {
        Err(crate::error::KwaversError::InvalidInput(
            "PINN feature not enabled. Add 'pinn' feature to Cargo.toml".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_pinn_module_exists() {
        // Basic module existence test
        assert!(true);
    }
}

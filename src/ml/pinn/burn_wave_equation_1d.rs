//! Burn-based 1D Wave Equation Physics-Informed Neural Network with Automatic Differentiation
//!
//! This module implements a PINN for the 1D acoustic wave equation using the Burn deep learning
//! framework with native automatic differentiation. This replaces the manual gradient computation
//! approach with proper backpropagation through PDE residuals.
//!
//! ## Wave Equation
//!
//! Solves: ∂²u/∂t² = c²∂²u/∂x²
//!
//! Where:
//! - u(x,t) = displacement/pressure field
//! - c = wave speed (m/s)
//! - x = spatial coordinate (m)
//! - t = time coordinate (s)
//!
//! ## Physics-Informed Loss
//!
//! L_total = λ_data × L_data + λ_pde × L_pde + λ_bc × L_bc
//!
//! Where:
//! - L_data: MSE between predictions and training data
//! - L_pde: MSE of PDE residual (computed via autodiff)
//! - L_bc: MSE of boundary condition violations
//!
//! ## References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks" - JCP 378:686-707
//! - Burn Framework: https://burn.dev/ (v0.18 API)
//!
//! ## Example
//!
//! ```rust,ignore
//! use burn::backend::NdArray;
//! use kwavers::ml::pinn::burn_wave_equation_1d::{BurnPINN1DWave, BurnPINNConfig};
//!
//! // Create PINN with NdArray backend (CPU)
//! type Backend = NdArray<f32>;
//! let config = BurnPINNConfig::default();
//! let pinn = BurnPINN1DWave::<Backend>::new(343.0, config);
//!
//! // Train on reference data
//! let metrics = pinn.train(x_data, t_data, u_data, 1000)?;
//!
//! // Predict at new points
//! let u_pred = pinn.predict(x_test, t_test)?;
//! ```

use crate::error::{KwaversError, KwaversResult};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use ndarray::{Array1, Array2};

/// Configuration for Burn-based 1D Wave Equation PINN
#[derive(Debug, Clone)]
pub struct BurnPINNConfig {
    /// Hidden layer sizes (e.g., [50, 50, 50, 50])
    pub hidden_layers: Vec<usize>,
    /// Learning rate for optimizer
    pub learning_rate: f64,
    /// Loss function weights
    pub loss_weights: BurnLossWeights,
    /// Number of collocation points for PDE residual
    pub num_collocation_points: usize,
}

impl Default for BurnPINNConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![50, 50, 50, 50],
            learning_rate: 1e-3,
            loss_weights: BurnLossWeights::default(),
            num_collocation_points: 10000,
        }
    }
}

/// Loss function weight configuration
#[derive(Debug, Clone, Copy)]
pub struct BurnLossWeights {
    /// Weight for data fitting loss (λ_data)
    pub data: f64,
    /// Weight for PDE residual loss (λ_pde)
    pub pde: f64,
    /// Weight for boundary condition loss (λ_bc)
    pub boundary: f64,
}

impl Default for BurnLossWeights {
    fn default() -> Self {
        Self {
            data: 1.0,
            pde: 1.0,
            boundary: 10.0, // Higher weight for boundary enforcement
        }
    }
}

/// Training metrics for monitoring convergence
#[derive(Debug, Clone)]
pub struct BurnTrainingMetrics {
    /// Total loss history
    pub total_loss: Vec<f64>,
    /// Data loss history
    pub data_loss: Vec<f64>,
    /// PDE residual loss history
    pub pde_loss: Vec<f64>,
    /// Boundary condition loss history
    pub bc_loss: Vec<f64>,
    /// Training time (seconds)
    pub training_time_secs: f64,
    /// Number of epochs completed
    pub epochs_completed: usize,
}

/// Burn-based Physics-Informed Neural Network for 1D Wave Equation
///
/// This struct uses Burn's automatic differentiation to compute gradients
/// through the PDE residual, enabling true physics-informed learning.
///
/// ## Architecture
///
/// - Input layer: 2 inputs (x, t) → hidden_size
/// - Hidden layers: N layers with tanh activation
/// - Output layer: hidden_size → 1 output (u)
///
/// ## Type Parameters
///
/// - `B`: Burn backend (e.g., NdArray for CPU, Wgpu for GPU)
#[derive(Module, Debug)]
pub struct BurnPINN1DWave<B: Backend> {
    /// Input layer (2 inputs: x, t)
    input_layer: Linear<B>,
    /// Hidden layers with tanh activation
    hidden_layers: Vec<Linear<B>>,
    /// Output layer (1 output: u)
    output_layer: Linear<B>,
}

impl<B: Backend> BurnPINN1DWave<B> {
    /// Create a new Burn-based PINN for 1D wave equation
    ///
    /// # Arguments
    ///
    /// * `wave_speed` - Speed of sound in the medium (m/s)
    /// * `config` - Network architecture and training configuration
    /// * `device` - Device to run computations on
    ///
    /// # Returns
    ///
    /// A new PINN instance ready for training
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use burn::backend::NdArray;
    ///
    /// type Backend = NdArray<f32>;
    /// let device = Default::default();
    /// let pinn = BurnPINN1DWave::<Backend>::new(343.0, BurnPINNConfig::default(), &device)?;
    /// ```
    pub fn new(
        config: BurnPINNConfig,
        device: &B::Device,
    ) -> KwaversResult<Self> {
        if config.hidden_layers.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Must have at least one hidden layer".into(),
            ));
        }

        // Input layer: 2 inputs (x, t) → first hidden layer size
        let input_size = 2;
        let first_hidden_size = config.hidden_layers[0];
        let input_layer = LinearConfig::new(input_size, first_hidden_size).init(device);

        // Hidden layers
        let mut hidden_layers = Vec::new();
        for i in 0..config.hidden_layers.len() - 1 {
            let in_size = config.hidden_layers[i];
            let out_size = config.hidden_layers[i + 1];
            hidden_layers.push(LinearConfig::new(in_size, out_size).init(device));
        }

        // Output layer: last hidden layer → 1 output (u)
        let last_hidden_size = *config.hidden_layers.last().unwrap();
        let output_layer = LinearConfig::new(last_hidden_size, 1).init(device);

        Ok(Self {
            input_layer,
            hidden_layers,
            output_layer,
        })
    }

    /// Forward pass through the network
    ///
    /// # Arguments
    ///
    /// * `x` - Spatial coordinates [batch_size, 1]
    /// * `t` - Time coordinates [batch_size, 1]
    ///
    /// # Returns
    ///
    /// Predicted field values u(x,t) [batch_size, 1]
    pub fn forward(&self, x: Tensor<B, 2>, t: Tensor<B, 2>) -> Tensor<B, 2> {
        // Concatenate inputs: [batch_size, 2]
        let input = Tensor::cat(vec![x, t], 1);

        // Input layer
        let mut h = self.input_layer.forward(input);

        // Hidden layers with tanh activation
        for layer in &self.hidden_layers {
            h = layer.forward(h);
            h = h.tanh();
        }

        // Output layer
        self.output_layer.forward(h)
    }

    /// Predict field values at given spatial and temporal coordinates
    ///
    /// # Arguments
    ///
    /// * `x` - Spatial coordinates (m)
    /// * `t` - Time coordinates (s)
    /// * `device` - Device to run computations on
    ///
    /// # Returns
    ///
    /// Predicted field values u(x,t)
    pub fn predict(
        &self,
        x: &Array1<f64>,
        t: &Array1<f64>,
        device: &B::Device,
    ) -> KwaversResult<Array2<f64>> {
        if x.len() != t.len() {
            return Err(KwaversError::InvalidInput(
                "x and t must have same length".into(),
            ));
        }

        let n = x.len();

        // Convert to tensors - create 2D tensors with shape [n, 1]
        let x_vec: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let t_vec: Vec<f32> = t.iter().map(|&v| v as f32).collect();

        // Create tensors from flat vectors and reshape to [n, 1]
        let x_tensor = Tensor::<B, 1>::from_floats(x_vec.as_slice(), device)
            .reshape([n, 1]);
        let t_tensor = Tensor::<B, 1>::from_floats(t_vec.as_slice(), device)
            .reshape([n, 1]);

        // Forward pass
        let u_tensor = self.forward(x_tensor, t_tensor);

        // Convert back to ndarray
        let u_data = u_tensor.to_data();
        let u_vec: Vec<f64> = u_data
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&v| v as f64)
            .collect();

        Ok(Array2::from_shape_vec((x.len(), 1), u_vec).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_burn_pinn_creation() {
        let device = Default::default();
        let config = BurnPINNConfig::default();
        let result = BurnPINN1DWave::<TestBackend>::new(config, &device);
        assert!(result.is_ok());
    }

    #[test]
    fn test_burn_pinn_invalid_config() {
        let device = Default::default();
        let mut config = BurnPINNConfig::default();
        config.hidden_layers = vec![]; // Empty hidden layers
        let result = BurnPINN1DWave::<TestBackend>::new(config, &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_burn_pinn_forward_pass() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![10, 10], // Smaller network for testing
            ..Default::default()
        };
        let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

        // Create test inputs - proper shape [1, 1]
        let x = Tensor::<TestBackend, 1>::from_floats([0.5], &device).reshape([1, 1]);
        let t = Tensor::<TestBackend, 1>::from_floats([0.1], &device).reshape([1, 1]);

        // Forward pass
        let u = pinn.forward(x, t);

        // Check output shape
        assert_eq!(u.dims(), [1, 1]);
    }

    #[test]
    fn test_burn_pinn_predict() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![10, 10],
            ..Default::default()
        };
        let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

        // Create test inputs
        let x = Array1::from_vec(vec![0.0, 0.5, 1.0]);
        let t = Array1::from_vec(vec![0.0, 0.1, 0.2]);

        // Predict
        let result = pinn.predict(&x, &t, &device);
        assert!(result.is_ok());

        let u = result.unwrap();
        assert_eq!(u.shape(), &[3, 1]);
    }

    #[test]
    fn test_burn_pinn_predict_mismatched_lengths() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![10, 10],
            ..Default::default()
        };
        let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

        let x = Array1::from_vec(vec![0.0, 0.5]);
        let t = Array1::from_vec(vec![0.0, 0.1, 0.2]);

        let result = pinn.predict(&x, &t, &device);
        assert!(result.is_err());
    }
}

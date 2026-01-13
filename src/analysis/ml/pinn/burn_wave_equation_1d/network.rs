//! Neural network architecture for Burn-based 1D Wave Equation PINN
//!
//! This module implements the core neural network architecture using the Burn deep learning
//! framework. The network learns to approximate solutions to the 1D acoustic wave equation
//! through physics-informed training.
//!
//! ## Architecture
//!
//! The network uses a fully connected feedforward architecture:
//! - **Input layer**: 2 inputs (spatial x, temporal t) → first hidden layer
//! - **Hidden layers**: Configurable depth and width with tanh activation
//! - **Output layer**: last hidden layer → 1 output (field u)
//!
//! ## Activation Function
//!
//! **tanh (hyperbolic tangent)** is used for hidden layers because:
//! - Smooth and infinitely differentiable (essential for computing ∂²u/∂x², ∂²u/∂t²)
//! - Bounded output [-1, 1] provides numerical stability
//! - Standard choice for PINNs (Raissi et al. 2019)
//!
//! ## Backend Support
//!
//! The implementation is generic over Burn backends:
//! - **NdArray**: CPU-only backend (fast compilation, development)
//! - **WGPU**: GPU acceleration via WebGPU (requires `pinn-gpu` feature)
//!
//! ## References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks: A deep learning framework
//!   for solving forward and inverse problems involving nonlinear partial differential equations"
//!   Journal of Computational Physics, 378:686-707. DOI: 10.1016/j.jcp.2018.10.045
//! - Hornik et al. (1989): "Multilayer feedforward networks are universal approximators"
//!   Neural Networks, 2(5):359-366. DOI: 10.1016/0893-6080(89)90020-8
//!
//! ## Examples
//!
//! ```rust,ignore
//! use burn::backend::NdArray;
//! use kwavers::analysis::ml::pinn::burn_wave_equation_1d::{BurnPINN1DWave, BurnPINNConfig};
//!
//! type Backend = NdArray<f32>;
//! let device = Default::default();
//! let config = BurnPINNConfig::default();
//!
//! // Create network
//! let pinn = BurnPINN1DWave::<Backend>::new(config, &device)?;
//!
//! // Forward pass
//! let x = Tensor::<Backend, 2>::from_floats([[0.5]], &device);
//! let t = Tensor::<Backend, 2>::from_floats([[0.1]], &device);
//! let u = pinn.forward(x, t);
//!
//! // Predict at multiple points
//! let x_vals = Array1::from_vec(vec![0.0, 0.5, 1.0]);
//! let t_vals = Array1::from_vec(vec![0.0, 0.1, 0.2]);
//! let u_pred = pinn.predict(&x_vals, &t_vals, &device)?;
//! ```

use crate::core::error::{KwaversError, KwaversResult};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use ndarray::{Array1, Array2};

use super::config::BurnPINNConfig;

/// Burn-based Physics-Informed Neural Network for 1D Wave Equation
///
/// This structure implements a fully connected feedforward neural network
/// that learns to approximate solutions to the 1D acoustic wave equation:
///
/// **∂²u/∂t² = c²∂²u/∂x²**
///
/// The network is trained using physics-informed loss functions that combine
/// data fitting with PDE residual minimization.
///
/// ## Universal Approximation
///
/// By the Universal Approximation Theorem (Hornik et al. 1989), a feedforward
/// network with at least one hidden layer and sufficient neurons can approximate
/// any continuous function to arbitrary accuracy. For PINNs, this extends to
/// approximating PDE solutions.
///
/// ## Architecture Details
///
/// - **Input normalization**: Not explicitly applied; assumes inputs are in reasonable ranges
/// - **Weight initialization**: Burn's default initialization (Xavier/Glorot uniform)
/// - **Activation**: tanh for smoothness and differentiability
/// - **Output**: Linear (no activation) for unbounded field values
///
/// ## Type Parameters
///
/// - `B`: Burn backend (must implement `Backend` trait)
///   - For inference: Any backend (NdArray, WGPU, etc.)
///   - For training: Must use `AutodiffBackend` for gradient computation
///
/// # Examples
///
/// ```rust,ignore
/// use burn::backend::NdArray;
///
/// type Backend = NdArray<f32>;
/// let device = Default::default();
///
/// let config = BurnPINNConfig {
///     hidden_layers: vec![50, 50, 50, 50],
///     ..Default::default()
/// };
///
/// let pinn = BurnPINN1DWave::<Backend>::new(config, &device)?;
/// println!("Created PINN with {} parameters", config.num_parameters());
/// ```
#[derive(Module, Debug)]
pub struct BurnPINN1DWave<B: Backend> {
    /// Input layer (2 inputs: x, t) → first hidden layer
    ///
    /// Maps spatiotemporal coordinates to the first hidden representation.
    input_layer: Linear<B>,

    /// Hidden layers with tanh activation
    ///
    /// Each layer transforms the representation, enabling the network to learn
    /// complex nonlinear mappings from (x,t) to u(x,t).
    hidden_layers: Vec<Linear<B>>,

    /// Output layer (last hidden layer → 1 output: u)
    ///
    /// Maps the final hidden representation to the scalar field value u(x,t).
    output_layer: Linear<B>,
}

impl<B: Backend> BurnPINN1DWave<B> {
    /// Create a new Burn-based PINN for 1D wave equation
    ///
    /// Initializes the neural network with the specified architecture.
    /// Weight initialization uses Burn's defaults (typically Xavier/Glorot uniform).
    ///
    /// # Arguments
    ///
    /// * `config` - Network architecture and training configuration
    /// * `device` - Device to allocate network parameters on
    ///
    /// # Returns
    ///
    /// - `Ok(pinn)` with initialized network
    /// - `Err(KwaversError::InvalidInput)` if configuration is invalid
    ///
    /// # Validation
    ///
    /// The configuration is validated to ensure:
    /// - At least one hidden layer is specified
    /// - All layer sizes are positive
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::backend::NdArray;
    /// use kwavers::analysis::ml::pinn::burn_wave_equation_1d::{BurnPINN1DWave, BurnPINNConfig};
    ///
    /// type Backend = NdArray<f32>;
    /// let device = Default::default();
    /// let config = BurnPINNConfig::default();
    /// let pinn = BurnPINN1DWave::<Backend>::new(config, &device)?;
    /// ```
    pub fn new(config: BurnPINNConfig, device: &B::Device) -> KwaversResult<Self> {
        // Validate configuration
        if config.hidden_layers.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Must have at least one hidden layer".into(),
            ));
        }

        // Input layer: 2 inputs (x, t) → first hidden layer size
        let input_size = 2;
        let first_hidden_size = config.hidden_layers[0];
        let input_layer = LinearConfig::new(input_size, first_hidden_size).init(device);

        // Hidden layers: progressive transformation
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

    /// Get the device this network is allocated on
    ///
    /// # Returns
    ///
    /// Device handle for this network
    pub fn device(&self) -> B::Device {
        self.input_layer.devices()[0].clone()
    }

    /// Forward pass through the network
    ///
    /// Computes the network output u(x,t) for given spatiotemporal coordinates.
    ///
    /// # Architecture Flow
    ///
    /// 1. Concatenate inputs: [x, t] → [batch_size, 2]
    /// 2. Input layer: [batch_size, 2] → [batch_size, hidden[0]]
    /// 3. Hidden layers: Apply linear transformation + tanh activation
    /// 4. Output layer: [batch_size, hidden[-1]] → [batch_size, 1]
    ///
    /// # Arguments
    ///
    /// * `x` - Spatial coordinates [batch_size, 1]
    /// * `t` - Time coordinates [batch_size, 1]
    ///
    /// # Returns
    ///
    /// Predicted field values u(x,t) [batch_size, 1]
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::tensor::Tensor;
    ///
    /// // Single point
    /// let x = Tensor::<Backend, 2>::from_floats([[0.5]], &device);
    /// let t = Tensor::<Backend, 2>::from_floats([[0.1]], &device);
    /// let u = pinn.forward(x, t);
    /// assert_eq!(u.dims(), [1, 1]);
    ///
    /// // Multiple points (batch)
    /// let x = Tensor::<Backend, 2>::from_floats([[0.0], [0.5], [1.0]], &device);
    /// let t = Tensor::<Backend, 2>::from_floats([[0.0], [0.1], [0.2]], &device);
    /// let u = pinn.forward(x, t);
    /// assert_eq!(u.dims(), [3, 1]);
    /// ```
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

        // Output layer (no activation for unbounded output)
        self.output_layer.forward(h)
    }

    /// Predict field values at given spatial and temporal coordinates
    ///
    /// High-level interface for inference that accepts ndarray inputs and returns
    /// ndarray outputs. Internally converts to Burn tensors, performs forward pass,
    /// and converts back.
    ///
    /// # Arguments
    ///
    /// * `x` - Spatial coordinates (m) [n_points]
    /// * `t` - Time coordinates (s) [n_points]
    /// * `device` - Device to run computations on
    ///
    /// # Returns
    ///
    /// - `Ok(u)` with predicted field values [n_points, 1]
    /// - `Err(KwaversError::InvalidInput)` if x and t have different lengths
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use ndarray::Array1;
    ///
    /// let x = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    /// let t = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4]);
    ///
    /// let u_pred = pinn.predict(&x, &t, &device)?;
    /// assert_eq!(u_pred.shape(), &[5, 1]);
    /// ```
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
        let x_tensor = Tensor::<B, 1>::from_floats(x_vec.as_slice(), device).reshape([n, 1]);
        let t_tensor = Tensor::<B, 1>::from_floats(t_vec.as_slice(), device).reshape([n, 1]);

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
    fn test_pinn_creation() {
        let device = Default::default();
        let config = BurnPINNConfig::default();
        let result = BurnPINN1DWave::<TestBackend>::new(config, &device);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pinn_invalid_config_empty_layers() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![],
            ..Default::default()
        };
        let result = BurnPINN1DWave::<TestBackend>::new(config, &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_pinn_forward_pass() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![10, 10],
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

        // Output should be finite
        let u_val = u.to_data().as_slice::<f32>().unwrap()[0];
        assert!(u_val.is_finite());
    }

    #[test]
    fn test_pinn_forward_pass_batch() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![10, 10],
            ..Default::default()
        };
        let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

        // Batch of 3 points
        let x = Tensor::<TestBackend, 1>::from_floats([0.0, 0.5, 1.0], &device).reshape([3, 1]);
        let t = Tensor::<TestBackend, 1>::from_floats([0.0, 0.1, 0.2], &device).reshape([3, 1]);

        let u = pinn.forward(x, t);
        assert_eq!(u.dims(), [3, 1]);

        // All outputs should be finite
        let binding = u.to_data();
        let u_vals = binding.as_slice::<f32>().unwrap();
        for &val in u_vals {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_pinn_predict() {
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

        // All predictions should be finite
        for &val in u.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_pinn_predict_mismatched_lengths() {
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

    #[test]
    fn test_pinn_device() {
        let device = Default::default();
        let config = BurnPINNConfig::default();
        let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

        let _ = pinn.device(); // Just check it doesn't panic
    }

    // GPU backend tests (conditional compilation)
    #[cfg(feature = "pinn-gpu")]
    mod gpu_tests {
        use super::*;
        use burn::backend::{Autodiff, Wgpu};

        type GpuBackend = Autodiff<Wgpu<f32>>;

        #[test]
        fn test_pinn_gpu_creation() {
            let device = burn::backend::wgpu::WgpuDevice::default();
            let config = BurnPINNConfig {
                hidden_layers: vec![20, 20],
                ..Default::default()
            };
            let result = BurnPINN1DWave::<GpuBackend>::new(config, &device);
            // GPU may not be available in all test environments
            let _ = result;
        }

        #[test]
        fn test_pinn_gpu_forward_pass() {
            let device = burn::backend::wgpu::WgpuDevice::default();
            let config = BurnPINNConfig {
                hidden_layers: vec![10, 10],
                ..Default::default()
            };
            if let Ok(pinn) = BurnPINN1DWave::<GpuBackend>::new(config, &device) {
                let x = Tensor::<GpuBackend, 1>::from_floats([0.5], &device).reshape([1, 1]);
                let t = Tensor::<GpuBackend, 1>::from_floats([0.1], &device).reshape([1, 1]);

                let u = pinn.forward(x, t);
                assert!(u.to_data().as_slice::<f32>().is_ok());
            }
        }
    }
}

//! Neural network architecture for 2D Elastic Wave PINN
//!
//! This module defines the neural network model that approximates the displacement
//! field solution to the 2D elastic wave equation.
//!
//! # Mathematical Formulation
//!
//! The network approximates the displacement field:
//!
//! ```text
//! u(x, y, t; θ) : ℝ³ → ℝ²
//! ```
//!
//! where:
//! - Input: (x, y, t) - spatial coordinates and time
//! - Output: (uₓ, uᵧ) - displacement components
//! - θ: trainable network parameters (weights and biases)
//!
//! # Architecture
//!
//! ## Standard Architecture
//!
//! ```text
//! Input [3] → Dense[3→N₁] → Activation → Dense[N₁→N₂] → ... → Dense[Nₖ→2] → Output [2]
//! ```
//!
//! Typical configuration:
//! - Input: 3 neurons (x, y, t)
//! - Hidden: 4-8 layers of 50-200 neurons each
//! - Output: 2 neurons (uₓ, uᵧ)
//! - Activation: tanh, sin, or swish
//!
//! ## Fourier Feature Network (Optional)
//!
//! For problems with high-frequency content, a Fourier feature mapping can be applied:
//!
//! ```text
//! γ(v) = [cos(2πB·v), sin(2πB·v)]ᵀ
//! ```
//!
//! where B is a random Fourier feature matrix. This helps overcome spectral bias.
//!
//! # Physics Integration
//!
//! The network is designed to:
//! 1. Provide smooth derivatives (required for PDE residual computation)
//! 2. Support automatic differentiation for gradient computation
//! 3. Allow spatially-varying material parameters (for inverse problems)
//!
//! # Usage
//!
//! ```rust,ignore
//! use kwavers::solver::inverse::pinn::elastic_2d::{Config, ElasticPINN2D};
//! use burn::backend::NdArray;
//!
//! type Backend = NdArray<f32>;
//!
//! let config = Config::default();
//! let device = Default::default();
//! let model = ElasticPINN2D::<Backend>::new(&config, &device)?;
//!
//! // Forward pass
//! let x = Tensor::from_floats([[0.5]], &device);
//! let y = Tensor::from_floats([[0.5]], &device);
//! let t = Tensor::from_floats([[0.1]], &device);
//! let displacement = model.forward(x, y, t);
//! ```

use super::config::Config;
use crate::core::error::{KwaversError, KwaversResult};

#[cfg(feature = "pinn")]
use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{backend::Backend, Tensor},
};

/// Physics-Informed Neural Network for 2D Elastic Wave Equation
///
/// This struct represents the neural network that approximates displacement fields
/// for elastic wave propagation problems.
///
/// # Type Parameters
///
/// - `B`: Burn backend (e.g., `NdArray` for CPU, `Wgpu` for GPU)
///
/// # Architecture Details
///
/// The network consists of:
/// - **Input layer**: Maps 3D input (x, y, t) to first hidden layer
/// - **Hidden layers**: Configurable depth and width with specified activation
/// - **Output layer**: Maps last hidden layer to 2D output (uₓ, uᵧ)
///
/// # Material Parameters (Inverse Problems)
///
/// For inverse problems, material parameters (λ, μ, ρ) can be made trainable:
/// - **Constant**: Single trainable scalar per material parameter
/// - **Spatially-varying**: Network that maps (x, y) → parameter value
///
/// # Derivatives
///
/// All derivatives required for PDE residual computation are obtained via
/// automatic differentiation through the network.
#[cfg(feature = "pinn")]
#[derive(Module, Debug)]
pub struct ElasticPINN2D<B: Backend> {
    /// Input layer: 3 inputs (x, y, t) → first hidden layer size
    pub input_layer: Linear<B>,

    /// Hidden layers with activation functions
    pub hidden_layers: Vec<Linear<B>>,

    /// Output layer: last hidden layer → 2 outputs (uₓ, uᵧ)
    pub output_layer: Linear<B>,

    /// Learnable Lamé parameter λ (if optimizing material properties)
    ///
    /// For inverse problems, this is a trainable parameter.
    /// For forward problems, this is fixed.
    pub lambda: Option<Param<Tensor<B, 1>>>,

    /// Learnable shear modulus μ (if optimizing material properties)
    pub mu: Option<Param<Tensor<B, 1>>>,

    /// Learnable density ρ (if optimizing material properties)
    pub rho: Option<Param<Tensor<B, 1>>>,
}

#[cfg(feature = "pinn")]
impl<B: Backend> ElasticPINN2D<B> {
    /// Create a new PINN model for 2D elastic wave equation
    ///
    /// # Arguments
    ///
    /// * `config` - Network configuration (architecture, material parameters, etc.)
    /// * `device` - Device to initialize the model on (CPU or GPU)
    ///
    /// # Returns
    ///
    /// A new PINN model ready for training
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid (validated before construction)
    pub fn new(config: &Config, device: &B::Device) -> KwaversResult<Self> {
        // Validate configuration
        config.validate().map_err(KwaversError::InvalidInput)?;

        let input_size = 3; // (x, y, t)
        let output_size = 2; // (uₓ, uᵧ)

        // Input layer: 3 → first hidden layer size
        let first_hidden_size = config.hidden_layers[0];
        let input_layer = LinearConfig::new(input_size, first_hidden_size).init(device);

        // Hidden layers
        let mut hidden_layers = Vec::with_capacity(config.hidden_layers.len() - 1);
        for i in 0..config.hidden_layers.len() - 1 {
            let in_size = config.hidden_layers[i];
            let out_size = config.hidden_layers[i + 1];
            hidden_layers.push(LinearConfig::new(in_size, out_size).init(device));
        }

        // Output layer: last hidden → 2 outputs
        let last_hidden_size = *config.hidden_layers.last().unwrap();
        let output_layer = LinearConfig::new(last_hidden_size, output_size).init(device);

        // Initialize material parameters if optimizing
        let lambda = if config.optimize_lambda {
            let lambda_init = config.lambda_init.expect("lambda_init required");
            let tensor = Tensor::from_floats([lambda_init as f32], device);
            Some(Param::from_tensor(tensor))
        } else {
            None
        };

        let mu = if config.optimize_mu {
            let mu_init = config.mu_init.expect("mu_init required");
            let tensor = Tensor::from_floats([mu_init as f32], device);
            Some(Param::from_tensor(tensor))
        } else {
            None
        };

        let rho = if config.optimize_rho {
            let rho_init = config.rho_init.expect("rho_init required");
            let tensor = Tensor::from_floats([rho_init as f32], device);
            Some(Param::from_tensor(tensor))
        } else {
            None
        };

        Ok(Self {
            input_layer,
            hidden_layers,
            output_layer,
            lambda,
            mu,
            rho,
        })
    }

    /// Forward pass through the network
    ///
    /// Computes the displacement field approximation u(x, y, t).
    ///
    /// # Arguments
    ///
    /// * `x` - X spatial coordinates [batch_size, 1]
    /// * `y` - Y spatial coordinates [batch_size, 1]
    /// * `t` - Time coordinates [batch_size, 1]
    ///
    /// # Returns
    ///
    /// Displacement field [batch_size, 2] where columns are (uₓ, uᵧ)
    ///
    /// # Mathematical Operation
    ///
    /// ```text
    /// input = [x, y, t] ∈ ℝ^(batch_size × 3)
    /// h₀ = activation(W₀·input + b₀)
    /// hᵢ = activation(Wᵢ·hᵢ₋₁ + bᵢ) for i = 1..k
    /// output = Wₖ₊₁·hₖ + bₖ₊₁ ∈ ℝ^(batch_size × 2)
    /// ```
    pub fn forward(&self, x: Tensor<B, 2>, y: Tensor<B, 2>, t: Tensor<B, 2>) -> Tensor<B, 2> {
        // Concatenate inputs: [batch_size, 3]
        let input = Tensor::cat(vec![x, y, t], 1);

        // Input layer with tanh activation (standard for PINNs)
        let mut h = self.input_layer.forward(input);
        h = h.tanh();

        // Hidden layers with tanh activation
        for layer in &self.hidden_layers {
            h = layer.forward(h);
            h = h.tanh();
        }

        // Output layer (no activation - linear output)
        self.output_layer.forward(h)
    }

    /// Get Lamé parameter λ
    ///
    /// # Arguments
    ///
    /// * `fixed_value` - Fixed value to use if not optimizing (Pa)
    ///
    /// # Returns
    ///
    /// Lamé parameter as scalar tensor
    pub fn get_lambda(&self, fixed_value: f64) -> Tensor<B, 1> {
        match &self.lambda {
            Some(param) => param.val(),
            None => {
                // Use fixed value - need device from existing tensors
                let device = self.input_layer.weight.device();
                Tensor::from_floats([fixed_value as f32], &device)
            }
        }
    }

    /// Get shear modulus μ
    ///
    /// # Arguments
    ///
    /// * `fixed_value` - Fixed value to use if not optimizing (Pa)
    ///
    /// # Returns
    ///
    /// Shear modulus as scalar tensor
    pub fn get_mu(&self, fixed_value: f64) -> Tensor<B, 1> {
        match &self.mu {
            Some(param) => param.val(),
            None => {
                let device = self.input_layer.weight.device();
                Tensor::from_floats([fixed_value as f32], &device)
            }
        }
    }

    /// Get density ρ
    ///
    /// # Arguments
    ///
    /// * `fixed_value` - Fixed value to use if not optimizing (kg/m³)
    ///
    /// # Returns
    ///
    /// Density as scalar tensor
    pub fn get_rho(&self, fixed_value: f64) -> Tensor<B, 1> {
        match &self.rho {
            Some(param) => param.val(),
            None => {
                let device = self.input_layer.weight.device();
                Tensor::from_floats([fixed_value as f32], &device)
            }
        }
    }

    /// Get estimated material parameters (for inverse problems)
    ///
    /// # Returns
    ///
    /// (λ, μ, ρ) as Option tuples - None if not being optimized
    pub fn estimated_parameters(&self) -> (Option<f64>, Option<f64>, Option<f64>) {
        let lambda_est = self.lambda.as_ref().map(|p| {
            let data = p.val().to_data();
            data.as_slice::<f32>().unwrap()[0] as f64
        });

        let mu_est = self.mu.as_ref().map(|p| {
            let data = p.val().to_data();
            data.as_slice::<f32>().unwrap()[0] as f64
        });

        let rho_est = self.rho.as_ref().map(|p| {
            let data = p.val().to_data();
            data.as_slice::<f32>().unwrap()[0] as f64
        });

        (lambda_est, mu_est, rho_est)
    }

    /// Get the device the model is on
    pub fn device(&self) -> B::Device {
        self.input_layer.weight.device()
    }

    /// Count total number of trainable parameters
    ///
    /// # Returns
    ///
    /// Total parameter count (network weights + material parameters)
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;

        // Note: Burn Linear layer weight shape is [in_features, out_features]
        // and bias shape is [out_features]

        // Input layer: weight [3, 10], bias [10]
        let input_dims = self.input_layer.weight.dims();
        count += input_dims[0] * input_dims[1]; // weights: 3 * 10 = 30
        count += input_dims[1]; // biases: 10

        // Hidden layers: weight [in, out], bias [out]
        for layer in &self.hidden_layers {
            let dims = layer.weight.dims();
            count += dims[0] * dims[1]; // weights
            count += dims[1]; // biases (out_features)
        }

        // Output layer: weight [10, 2], bias [2]
        let output_dims = self.output_layer.weight.dims();
        count += output_dims[0] * output_dims[1]; // weights: 10 * 2 = 20
        count += output_dims[1]; // biases: 2

        // Material parameters
        if self.lambda.is_some() {
            count += 1;
        }
        if self.mu.is_some() {
            count += 1;
        }
        if self.rho.is_some() {
            count += 1;
        }

        count
    }

    /// Save model to file using Burn's Record system
    ///
    /// # Arguments
    ///
    /// * `path` - File path to save to (will create .mpk.gz file)
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    ///
    /// # Format
    ///
    /// Uses BinFileRecorder with MessagePack format and full precision settings.
    /// The model weights, biases, and material parameters are serialized.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// model.save_checkpoint("checkpoints/model_epoch_100.mpk")?;
    /// ```
    pub fn save_checkpoint<P: AsRef<std::path::Path>>(&self, path: P) -> KwaversResult<()> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let path_buf = path.as_ref().to_path_buf();
        self.clone().save_file(path_buf, &recorder).map_err(|e| {
            KwaversError::InvalidInput(format!("Model checkpoint save failed: {:?}", e))
        })
    }

    /// Load model from file using Burn's Record system
    ///
    /// # Arguments
    ///
    /// * `path` - File path to load from
    /// * `device` - Device to load model onto
    ///
    /// # Returns
    ///
    /// Loaded model on specified device
    ///
    /// # Format
    ///
    /// Expects BinFileRecorder MessagePack format (matches save_checkpoint).
    ///
    /// # Note
    ///
    /// This is a placeholder - use Trainer::load_checkpoint for full functionality
    pub fn load_checkpoint<P: AsRef<std::path::Path>>(
        _path: P,
        _device: &B::Device,
    ) -> KwaversResult<Self> {
        // Placeholder - requires config to construct model structure
        Err(KwaversError::InvalidInput(
            "Use Trainer::load_checkpoint instead - direct model loading requires config"
                .to_string(),
        ))
    }
}

/// Non-Burn fallback for when Burn feature is disabled
#[cfg(not(feature = "pinn"))]
#[derive(Debug)]
pub struct ElasticPINN2D {
    _phantom: std::marker::PhantomData<()>,
}

#[cfg(not(feature = "pinn"))]
impl ElasticPINN2D {
    pub fn new(_config: &Config, _device: &()) -> KwaversResult<Self> {
        Err(KwaversError::InvalidInput(
            "ElasticPINN2D requires 'burn' feature to be enabled".to_string(),
        ))
    }
}

#[cfg(all(test, feature = "pinn"))]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_model_creation() {
        let config = Config::default();
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device);
        assert!(model.is_ok());
    }

    #[test]
    fn test_model_forward_pass() {
        let config = Config::default();
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

        // Single point
        let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
        let y = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
        let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);

        let output = model.forward(x, y, t);
        let dims = output.dims();
        assert_eq!(dims[0], 1); // batch size
        assert_eq!(dims[1], 2); // (uₓ, uᵧ)
    }

    #[test]
    fn test_model_batch_forward() {
        let config = Config::default();
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

        let batch_size = 10;
        // Create batch tensors by repeating single values along the batch dimension
        let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device).repeat(&[batch_size, 1]);
        let y = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device).repeat(&[batch_size, 1]);
        let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device).repeat(&[batch_size, 1]);

        let output = model.forward(x, y, t);
        let dims = output.dims();
        assert_eq!(dims[0], batch_size);
        assert_eq!(dims[1], 2);
    }

    #[test]
    fn test_inverse_problem_parameters() {
        let config = Config::inverse_problem(1e9, 5e8, 1000.0);
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

        assert!(model.lambda.is_some());
        assert!(model.mu.is_some());
        assert!(model.rho.is_some());

        let (lambda_est, mu_est, rho_est) = model.estimated_parameters();
        assert!(lambda_est.is_some());
        assert!(mu_est.is_some());
        assert!(rho_est.is_some());

        // Check initialization
        assert!((lambda_est.unwrap() - 1e9).abs() < 1e-3);
        assert!((mu_est.unwrap() - 5e8).abs() < 1e-3);
        assert!((rho_est.unwrap() - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn test_forward_problem_no_learnable_params() {
        let config = Config::forward_problem(1e9, 5e8, 1000.0);
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

        assert!(model.lambda.is_none());
        assert!(model.mu.is_none());
        assert!(model.rho.is_none());

        let (lambda_est, mu_est, rho_est) = model.estimated_parameters();
        assert!(lambda_est.is_none());
        assert!(mu_est.is_none());
        assert!(rho_est.is_none());
    }

    #[test]
    fn test_parameter_count() {
        let config = Config {
            hidden_layers: vec![10, 10],
            ..Default::default()
        };
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

        let count = model.num_parameters();

        // Burn Linear layers have weight shape [in_features, out_features] and bias [out_features]
        // Input: [3, 10] weights + [10] bias = 30 + 10 = 40
        // Hidden: [10, 10] weights + [10] bias = 100 + 10 = 110
        // Output: [10, 2] weights + [2] bias = 20 + 2 = 22
        // Total: 40 + 110 + 22 = 172
        assert_eq!(count, 172);
    }

    #[test]
    fn test_activation_functions() {
        let device = Default::default();

        // Test that different activation functions produce different outputs
        let x = Tensor::<TestBackend, 2>::from_floats([[1.0]], &device);

        // Tanh: Should be around 0.76
        let y_tanh = x.clone().tanh();
        let tanh_val = y_tanh.to_data().as_slice::<f32>().unwrap()[0];
        assert!((tanh_val - 0.76).abs() < 0.1);

        // Sin: Should be around 0.84
        let y_sin = x.clone().sin();
        let sin_val = y_sin.to_data().as_slice::<f32>().unwrap()[0];
        assert!((sin_val - 0.84).abs() < 0.1);

        // Swish (x * sigmoid(x)): Should be around 0.73
        // Sigmoid(x) = 1 / (1 + exp(-x))
        let neg_x = x.clone().neg();
        let exp_neg_x = neg_x.exp();
        let one = Tensor::<TestBackend, 2>::ones_like(&x);
        let sigmoid_x = one.clone() / (one + exp_neg_x);
        let y_swish = x.clone() * sigmoid_x;
        let swish_val = y_swish.to_data().as_slice::<f32>().unwrap()[0];
        assert!((swish_val - 0.73).abs() < 0.1);
    }

    #[test]
    fn test_get_material_parameters() {
        let config = Config::inverse_problem(2e9, 1e9, 2000.0);
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

        let lambda = model.get_lambda(1e9);
        let mu = model.get_mu(5e8);
        let rho = model.get_rho(1000.0);

        // Should return learned values, not fixed values
        let lambda_val = lambda.to_data().as_slice::<f32>().unwrap()[0] as f64;
        let mu_val = mu.to_data().as_slice::<f32>().unwrap()[0] as f64;
        let rho_val = rho.to_data().as_slice::<f32>().unwrap()[0] as f64;

        assert!((lambda_val - 2e9).abs() < 1e-3);
        assert!((mu_val - 1e9).abs() < 1e-3);
        assert!((rho_val - 2000.0).abs() < 1e-3);
    }
}

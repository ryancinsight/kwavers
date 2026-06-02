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
//! ## References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks: A deep learning framework
//!   for solving forward and inverse problems involving nonlinear partial differential equations"
//!   Journal of Computational Physics, 378:686-707. DOI: 10.1016/j.jcp.2018.10.045
//! - Hornik et al. (1989): "Multilayer feedforward networks are universal approximators"
//!   Neural Networks, 2(5):359-366. DOI: 10.1016/0893-6080(89)90020-8

use super::super::config::BurnPINNConfig;
use kwavers_core::error::{KwaversError, KwaversResult};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use ndarray::{Array1, Array2};

/// Burn-based Physics-Informed Neural Network for 1D Wave Equation.
///
/// Implements a fully connected feedforward neural network that approximates
/// solutions to the 1D acoustic wave equation: **∂²u/∂t² = c²∂²u/∂x²**
///
/// By the Universal Approximation Theorem (Hornik et al. 1989), this architecture
/// can approximate any continuous function to arbitrary accuracy.
#[derive(Module, Debug)]
pub struct BurnPINN1DWave<B: Backend> {
    /// Input layer (2 inputs: x, t) → first hidden layer.
    input_layer: Linear<B>,

    /// Hidden layers with tanh activation.
    hidden_layers: Vec<Linear<B>>,

    /// Output layer (last hidden layer → 1 output: u).
    output_layer: Linear<B>,
}

impl<B: Backend> BurnPINN1DWave<B> {
    /// Create a new Burn-based PINN for 1D wave equation.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn new(config: BurnPINNConfig, device: &B::Device) -> KwaversResult<Self> {
        if config.hidden_layers.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Must have at least one hidden layer".into(),
            ));
        }

        let input_size = 2;
        let first_hidden_size = config.hidden_layers[0];
        let input_layer = LinearConfig::new(input_size, first_hidden_size).init(device);

        let mut hidden_layers = Vec::new();
        for i in 0..config.hidden_layers.len() - 1 {
            let in_size = config.hidden_layers[i];
            let out_size = config.hidden_layers[i + 1];
            hidden_layers.push(LinearConfig::new(in_size, out_size).init(device));
        }

        let last_hidden_size = *config.hidden_layers.last().unwrap();
        let output_layer = LinearConfig::new(last_hidden_size, 1).init(device);

        Ok(Self {
            input_layer,
            hidden_layers,
            output_layer,
        })
    }

    /// Get the device this network is allocated on.
    pub fn device(&self) -> B::Device {
        self.input_layer.devices()[0].clone()
    }

    /// Forward pass through the network.
    ///
    /// Computes u(x,t) for given spatiotemporal coordinates.
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
        let input = Tensor::cat(vec![x, t], 1);
        let mut h = self.input_layer.forward(input);
        for layer in &self.hidden_layers {
            h = layer.forward(h);
            h = h.tanh();
        }
        self.output_layer.forward(h)
    }

    /// Predict field values at given spatial and temporal coordinates.
    ///
    /// High-level inference interface accepting ndarray inputs and returning ndarray outputs.
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
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
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
        let x_vec: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let t_vec: Vec<f32> = t.iter().map(|&v| v as f32).collect();

        let x_tensor = Tensor::<B, 1>::from_floats(x_vec.as_slice(), device).reshape([n, 1]);
        let t_tensor = Tensor::<B, 1>::from_floats(t_vec.as_slice(), device).reshape([n, 1]);

        let u_tensor = self.forward(x_tensor, t_tensor);
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

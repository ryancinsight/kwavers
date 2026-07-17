//! Neural network architecture for Coeus-backed 1D Wave Equation PINN
//!
//! This module implements the core neural network architecture using the Coeus autodiff
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

use super::super::config::PinnConfig;
use coeus_autograd::Var;
use coeus_nn::{Linear, Module};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array1, Array2};

/// Coeus-backed Physics-Informed Neural Network for 1D Wave Equation.
///
/// Implements a fully connected feedforward neural network that approximates
/// solutions to the 1D acoustic wave equation: **∂²u/∂t² = c²∂²u/∂x²**
///
/// By the Universal Approximation Theorem (Hornik et al. 1989), this architecture
/// can approximate any continuous function to arbitrary accuracy.
#[derive(Clone)]
pub struct PinnWave1D<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Input layer (2 inputs: x, t) → first hidden layer.
    input_layer: Linear<f32, B>,

    /// Hidden layers with tanh activation.
    hidden_layers: Vec<Linear<f32, B>>,

    /// Output layer (last hidden layer → 1 output: u).
    output_layer: Linear<f32, B>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for PinnWave1D<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnWave1D")
            .field("hidden_layers", &(self.hidden_layers.len()))
            .finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> PinnWave1D<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Create a new PINN for the 1D wave equation.
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn new(config: PinnConfig) -> KwaversResult<Self> {
        if config.hidden_layers.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Must have at least one hidden layer".into(),
            ));
        }

        let input_size = 2;
        let first_hidden_size = config.hidden_layers[0];
        let input_layer = Linear::new(input_size, first_hidden_size, true);

        let mut hidden_layers = Vec::new();
        for i in 0..(config.hidden_layers.len()) - 1 {
            let in_size = config.hidden_layers[i];
            let out_size = config.hidden_layers[i + 1];
            hidden_layers.push(Linear::new(in_size, out_size, true));
        }

        let last_hidden_size = *config.hidden_layers.last().unwrap();
        let output_layer = Linear::new(last_hidden_size, 1, true);

        Ok(Self {
            input_layer,
            hidden_layers,
            output_layer,
        })
    }

    /// Collect all trainable parameters, in the order `load_parameters` expects them back.
    pub fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut params = self.input_layer.parameters();
        for layer in &self.hidden_layers {
            params.extend(layer.parameters());
        }
        params.extend(self.output_layer.parameters());
        params
    }

    /// Write optimizer-updated parameter values back into this network's layers.
    ///
    /// # Panics
    /// - Panics if `(params.len())` does not match `self.parameters().len()`.
    pub fn load_parameters(&mut self, params: &[Var<f32, B>]) {
        let mut offset = 0;
        let n_in = self.input_layer.parameters().len();
        self.input_layer
            .load_parameters(&params[offset..offset + n_in]);
        offset += n_in;
        for layer in &mut self.hidden_layers {
            let n = layer.parameters().len();
            layer.load_parameters(&params[offset..offset + n]);
            offset += n;
        }
        let n_out = self.output_layer.parameters().len();
        self.output_layer
            .load_parameters(&params[offset..offset + n_out]);
        assert_eq!(
            offset + n_out,
            (params.len()),
            "load_parameters: parameter count mismatch"
        );
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
    pub fn forward(&self, x: &Var<f32, B>, t: &Var<f32, B>) -> Var<f32, B> {
        let input = coeus_autograd::cat(&[x, t], 1);
        let mut h = self.input_layer.forward(&input);
        for layer in &self.hidden_layers {
            h = layer.forward(&h);
            h = coeus_autograd::tanh(&h);
        }
        self.output_layer.forward(&h)
    }

    /// Predict field values at given spatial and temporal coordinates.
    ///
    /// High-level inference interface accepting ndarray inputs and returning ndarray outputs.
    ///
    /// # Arguments
    ///
    /// * `x` - Spatial coordinates (m) [n_points]
    /// * `t` - Time coordinates (s) [n_points]
    ///
    /// # Returns
    ///
    /// - `Ok(u)` with predicted field values [n_points, 1]
    /// - `Err(KwaversError::InvalidInput)` if x and t have different lengths
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn predict(&self, x: &Array1<f64>, t: &Array1<f64>) -> KwaversResult<Array2<f64>> {
        if (x.len()) != (t.len()) {
            return Err(KwaversError::InvalidInput(
                "x and t must have same length".into(),
            ));
        }

        let n = x.len();
        let x_vec: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let t_vec: Vec<f32> = t.iter().map(|&v| v as f32).collect();

        let backend = B::default();
        let x_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n, 1], &x_vec, &backend),
            false,
        );
        let t_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n, 1], &t_vec, &backend),
            false,
        );

        let u_var = self.forward(&x_var, &t_var);
        let u_vec: Vec<f64> = u_var.tensor.as_slice().iter().map(|&v| v as f64).collect();

        Ok(Array2::from_shape_vec([n, 1], u_vec).unwrap())
    }
}

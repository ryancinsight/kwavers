//! `BurnPINN2DWave` — 2D physics-informed neural network model.

use super::super::config::{BurnLossWeights2D, BurnPINN2DConfig};
use super::wave_speed::WaveSpeedFn;
use kwavers_core::error::{KwaversError, KwaversResult};
use burn::module::{Ignored, Module};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Physics-informed neural network for 2D wave equation.
///
/// Supports spatially varying wave speeds c(x,y) for complex media.
#[derive(Module, Debug)]
pub struct BurnPINN2DWave<B: Backend> {
    /// Input layer (3 inputs: x, y, t).
    pub input_layer: Linear<B>,
    /// Hidden layers with tanh activation.
    pub hidden_layers: Vec<Linear<B>>,
    /// Output layer (1 output: u).
    pub output_layer: Linear<B>,
    /// Wave speed function c(x,y) for heterogeneous media (optional).
    pub wave_speed_fn: Option<WaveSpeedFn<B>>,
    /// Configuration used to create the model.
    pub config: Ignored<BurnPINN2DConfig>,
}

impl<B: Backend> BurnPINN2DWave<B> {
    /// Create a new homogeneous PINN model.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn new(config: BurnPINN2DConfig, device: &B::Device) -> KwaversResult<Self> {
        if config.hidden_layers.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Must have at least one hidden layer".into(),
            ));
        }

        let input_size = 3;
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
            wave_speed_fn: None,
            config: Ignored(config),
        })
    }

    /// Create a heterogeneous PINN model with spatially varying wave speed.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new_heterogeneous<F>(
        config: BurnPINN2DConfig,
        wave_speed_fn: F,
        device: &B::Device,
    ) -> KwaversResult<Self>
    where
        F: Fn(f32, f32) -> f32 + Send + Sync + 'static,
    {
        let mut pinn = Self::new(config, device)?;
        pinn.wave_speed_fn = Some(WaveSpeedFn::new(Arc::new(wave_speed_fn)));
        Ok(pinn)
    }

    /// Forward pass through the network.
    pub fn forward(&self, x: Tensor<B, 2>, y: Tensor<B, 2>, t: Tensor<B, 2>) -> Tensor<B, 2> {
        let input = Tensor::cat(vec![x, y, t], 1);
        let mut h = self.input_layer.forward(input);
        for layer in &self.hidden_layers {
            h = layer.forward(h);
            h = h.tanh();
        }
        self.output_layer.forward(h)
    }

    /// Get wave speed at a specific location, using a default value if no function is provided.
    pub fn get_wave_speed_with_default(&self, x: f32, y: f32, default_c: f32) -> f32 {
        self.wave_speed_fn
            .as_ref()
            .map(|f| f.get(x, y))
            .unwrap_or(default_c)
    }

    /// Get the device this model is on.
    pub fn device(&self) -> B::Device {
        self.input_layer
            .devices()
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// Get the number of parameters in the model.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn num_parameters(&self) -> usize {
        let input_params = self.config.0.hidden_layers[0] * 3;
        let mut hidden_params = 0;
        for i in 0..self.config.0.hidden_layers.len() - 1 {
            hidden_params += self.config.0.hidden_layers[i] * self.config.0.hidden_layers[i + 1];
        }
        let output_params = *self.config.0.hidden_layers.last().unwrap();
        let bias_count: usize = self.config.0.hidden_layers.iter().sum::<usize>() + 1;
        input_params + hidden_params + output_params + bias_count
    }

    /// Get all model parameters (weights and biases).
    pub fn parameters(&self) -> Vec<Tensor<B, 1>> {
        let mut params = Vec::new();

        params.push(self.input_layer.weight.val().flatten(0, 1));
        if let Some(bias) = &self.input_layer.bias {
            params.push(bias.val().flatten(0, 0));
        }

        for layer in &self.hidden_layers {
            params.push(layer.weight.val().flatten(0, 1));
            if let Some(bias) = &layer.bias {
                params.push(bias.val().flatten(0, 0));
            }
        }

        params.push(self.output_layer.weight.val().flatten(0, 1));
        if let Some(bias) = &self.output_layer.bias {
            params.push(bias.val().flatten(0, 0));
        }

        params
    }

    /// Predict field values at given spatial and temporal coordinates.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn predict(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        t: &Array1<f64>,
        device: &B::Device,
    ) -> KwaversResult<Array2<f64>> {
        if x.len() != y.len() || x.len() != t.len() {
            return Err(KwaversError::InvalidInput(
                "x, y, and t must have same length".into(),
            ));
        }

        let n = x.len();

        let x_vec: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let y_vec: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let t_vec: Vec<f32> = t.iter().map(|&v| v as f32).collect();

        let x_tensor = Tensor::<B, 1>::from_floats(x_vec.as_slice(), device).reshape([n, 1]);
        let y_tensor = Tensor::<B, 1>::from_floats(y_vec.as_slice(), device).reshape([n, 1]);
        let t_tensor = Tensor::<B, 1>::from_floats(t_vec.as_slice(), device).reshape([n, 1]);

        let u_tensor = self.forward(x_tensor, y_tensor, t_tensor);

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

impl<B: AutodiffBackend> BurnPINN2DWave<B> {
    /// Compute PDE residual using finite differences within autodiff framework.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn compute_pde_residual(
        &self,
        x: Tensor<B, 2>,
        y: Tensor<B, 2>,
        t: Tensor<B, 2>,
        wave_speed: f64,
    ) -> Tensor<B, 2> {
        let base_eps = (f32::EPSILON).sqrt();
        let scale_factor = 1e-2_f32;
        let eps = base_eps * scale_factor;

        let u = self.forward(x.clone(), y.clone(), t.clone());

        let u_xx = {
            let u_x_plus = self.forward(x.clone().add_scalar(eps), y.clone(), t.clone());
            let u_x_minus = self.forward(x.clone().sub_scalar(eps), y.clone(), t.clone());
            u_x_plus
                .add(u_x_minus)
                .sub(u.clone().mul_scalar(2.0))
                .div_scalar(eps * eps)
        };

        let u_yy = {
            let u_y_plus = self.forward(x.clone(), y.clone().add_scalar(eps), t.clone());
            let u_y_minus = self.forward(x.clone(), y.clone().sub_scalar(eps), t.clone());
            u_y_plus
                .add(u_y_minus)
                .sub(u.clone().mul_scalar(2.0))
                .div_scalar(eps * eps)
        };

        let u_tt = {
            let u_t_plus = self.forward(x.clone(), y.clone(), t.clone().add_scalar(eps));
            let u_t_minus = self.forward(x.clone(), y.clone(), t.clone().sub_scalar(eps));
            u_t_plus
                .add(u_t_minus)
                .sub(u.clone().mul_scalar(2.0))
                .div_scalar(eps * eps)
        };

        let laplacian = u_xx.add(u_yy);

        let batch_size = x.shape().dims[0];
        let c_values: Vec<f32> = (0..batch_size)
            .map(|i| {
                let x_val = x
                    .clone()
                    .slice([i..i + 1, 0..1])
                    .into_data()
                    .as_slice::<f32>()
                    .unwrap()[0];
                let y_val = y
                    .clone()
                    .slice([i..i + 1, 0..1])
                    .into_data()
                    .as_slice::<f32>()
                    .unwrap()[0];
                self.get_wave_speed_with_default(x_val, y_val, wave_speed as f32)
            })
            .collect();

        let c_tensor =
            Tensor::<B, 1>::from_floats(c_values.as_slice(), &x.device()).reshape([batch_size, 1]);
        let c_squared = c_tensor.powf_scalar(2.0);

        u_tt.sub(laplacian.mul(c_squared))
    }

    /// Compute physics-informed loss function.
    pub fn compute_physics_loss(
        &self,
        x_data: Tensor<B, 2>,
        y_data: Tensor<B, 2>,
        t_data: Tensor<B, 2>,
        u_data: Tensor<B, 2>,
        x_collocation: Tensor<B, 2>,
        y_collocation: Tensor<B, 2>,
        t_collocation: Tensor<B, 2>,
        x_boundary: Tensor<B, 2>,
        y_boundary: Tensor<B, 2>,
        t_boundary: Tensor<B, 2>,
        u_boundary: Tensor<B, 2>,
        x_initial: Tensor<B, 2>,
        y_initial: Tensor<B, 2>,
        t_initial: Tensor<B, 2>,
        u_initial: Tensor<B, 2>,
        wave_speed: f64,
        loss_weights: BurnLossWeights2D,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
    ) {
        let u_pred_data = self.forward(x_data, y_data, t_data);
        let data_loss = (u_pred_data - u_data).powf_scalar(2.0).mean();

        let residual =
            self.compute_pde_residual(x_collocation, y_collocation, t_collocation, wave_speed);
        let pde_loss = residual.powf_scalar(2.0).mean() * 1e-12_f32;

        let u_pred_boundary = self.forward(x_boundary, y_boundary, t_boundary);
        let bc_loss = (u_pred_boundary - u_boundary).powf_scalar(2.0).mean();

        let u_pred_initial = self.forward(x_initial, y_initial, t_initial);
        let ic_loss = (u_pred_initial - u_initial).powf_scalar(2.0).mean();

        let total_loss = data_loss.clone() * loss_weights.data as f32
            + pde_loss.clone() * loss_weights.pde as f32
            + bc_loss.clone() * loss_weights.boundary as f32
            + ic_loss.clone() * loss_weights.initial as f32;

        (total_loss, data_loss, pde_loss, bc_loss, ic_loss)
    }
}

//! `PinnWave2D` — 2D physics-informed neural network model.

use super::super::config::{LossWeights2D, PinnConfig2D};
use super::wave_speed::WaveSpeedFn;
use coeus_autograd::Var;
use coeus_nn::{Linear, Module};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{
    Array1,
    Array2,
};
use std::sync::Arc;

/// Decomposed physics-informed loss components returned by
/// [`PinnWave2D::compute_physics_loss`]:
/// `(total, data, pde, boundary, initial)` scalar losses.
type PhysicsLossComponents<B> = (
    Var<f32, B>,
    Var<f32, B>,
    Var<f32, B>,
    Var<f32, B>,
    Var<f32, B>,
);

/// Physics-informed neural network for 2D wave equation.
///
/// Supports spatially varying wave speeds c(x,y) for complex media.
#[derive(Clone)]
pub struct PinnWave2D<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Input layer (3 inputs: x, y, t).
    pub input_layer: Linear<f32, B>,
    /// Hidden layers with tanh activation.
    pub hidden_layers: Vec<Linear<f32, B>>,
    /// Output layer (1 output: u).
    pub output_layer: Linear<f32, B>,
    /// Wave speed function c(x,y) for heterogeneous media (optional).
    pub wave_speed_fn: Option<WaveSpeedFn<B>>,
    /// Configuration used to create the model.
    pub config: PinnConfig2D,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for PinnWave2D<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnWave2D")
            .field("hidden_layers", &(self.hidden_layers.len()))
            .field("wave_speed_fn", &self.wave_speed_fn)
            .finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> PinnWave2D<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Create a new homogeneous PINN model.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn new(config: PinnConfig2D) -> KwaversResult<Self> {
        if config.hidden_layers.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Must have at least one hidden layer".into(),
            ));
        }

        let input_size = 3;
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
            wave_speed_fn: None,
            config,
        })
    }

    /// Create a heterogeneous PINN model with spatially varying wave speed.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new_heterogeneous<F>(config: PinnConfig2D, wave_speed_fn: F) -> KwaversResult<Self>
    where
        F: Fn(f32, f32) -> f32 + Send + Sync + 'static,
    {
        let mut pinn = Self::new(config)?;
        pinn.wave_speed_fn = Some(WaveSpeedFn::new(Arc::new(wave_speed_fn)));
        Ok(pinn)
    }

    /// Forward pass through the network.
    pub fn forward(&self, x: &Var<f32, B>, y: &Var<f32, B>, t: &Var<f32, B>) -> Var<f32, B> {
        let input = coeus_autograd::cat(&[x, y, t], 1);
        let mut h = self.input_layer.forward(&input);
        for layer in &self.hidden_layers {
            h = layer.forward(&h);
            h = coeus_autograd::tanh(&h);
        }
        self.output_layer.forward(&h)
    }

    /// Get wave speed at a specific location, using a default value if no function is provided.
    pub fn get_wave_speed_with_default(&self, x: f32, y: f32, default_c: f32) -> f32 {
        self.wave_speed_fn
            .as_ref()
            .map(|f| f.get(x, y))
            .unwrap_or(default_c)
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

    /// Get the number of parameters in the model.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn num_parameters(&self) -> usize {
        let input_params = self.config.hidden_layers[0] * 3;
        let mut hidden_params = 0;
        for i in 0..(self.config.hidden_layers.len()) - 1 {
            hidden_params += self.config.hidden_layers[i] * self.config.hidden_layers[i + 1];
        }
        let output_params = *self.config.hidden_layers.last().unwrap();
        let bias_count: usize = self.config.hidden_layers.iter().sum::<usize>() + 1;
        input_params + hidden_params + output_params + bias_count
    }

    /// Predict field values at given spatial and temporal coordinates.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn predict(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        t: &Array1<f64>,
    ) -> KwaversResult<Array2<f64>> {
        if (x.len()) != (y.len()) || (x.len()) != (t.len()) {
            return Err(KwaversError::InvalidInput(
                "x, y, and t must have same length".into(),
            ));
        }

        let n = (x.len());
        let backend = B::default();

        let x_vec: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let y_vec: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let t_vec: Vec<f32> = t.iter().map(|&v| v as f32).collect();

        let x_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n, 1], &x_vec, &backend),
            false,
        );
        let y_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n, 1], &y_vec, &backend),
            false,
        );
        let t_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n, 1], &t_vec, &backend),
            false,
        );

        let u_var = self.forward(&x_var, &y_var, &t_var);
        let u_vec: Vec<f64> = u_var.tensor.as_slice().iter().map(|&v| v as f64).collect();

        Ok(Array2::from_shape_vec([n, 1], u_vec).unwrap())
    }

    /// Compute PDE residual using finite differences.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn compute_pde_residual(
        &self,
        x: &Var<f32, B>,
        y: &Var<f32, B>,
        t: &Var<f32, B>,
        wave_speed: f64,
    ) -> Var<f32, B> {
        let base_eps = (f32::EPSILON).sqrt();
        let scale_factor = 1e-2_f32;
        let eps = base_eps * scale_factor;

        let x_plus = coeus_autograd::scalar_add(x, eps);
        let x_minus = coeus_autograd::scalar_add(x, -eps);
        let y_plus = coeus_autograd::scalar_add(y, eps);
        let y_minus = coeus_autograd::scalar_add(y, -eps);
        let t_plus = coeus_autograd::scalar_add(t, eps);
        let t_minus = coeus_autograd::scalar_add(t, -eps);

        let u = self.forward(x, y, t);
        let two_u = coeus_autograd::scalar_mul(&u, 2.0);

        let u_xx = {
            let u_x_plus = self.forward(&x_plus, y, t);
            let u_x_minus = self.forward(&x_minus, y, t);
            let sum = coeus_autograd::add(&u_x_plus, &u_x_minus);
            let diff = coeus_autograd::sub(&sum, &two_u);
            coeus_autograd::scalar_mul(&diff, 1.0 / (eps * eps))
        };

        let u_yy = {
            let u_y_plus = self.forward(x, &y_plus, t);
            let u_y_minus = self.forward(x, &y_minus, t);
            let sum = coeus_autograd::add(&u_y_plus, &u_y_minus);
            let diff = coeus_autograd::sub(&sum, &two_u);
            coeus_autograd::scalar_mul(&diff, 1.0 / (eps * eps))
        };

        let u_tt = {
            let u_t_plus = self.forward(x, y, &t_plus);
            let u_t_minus = self.forward(x, y, &t_minus);
            let sum = coeus_autograd::add(&u_t_plus, &u_t_minus);
            let diff = coeus_autograd::sub(&sum, &two_u);
            coeus_autograd::scalar_mul(&diff, 1.0 / (eps * eps))
        };

        let laplacian = coeus_autograd::add(&u_xx, &u_yy);

        let x_slice = x.tensor.as_slice();
        let y_slice = y.tensor.as_slice();
        let batch_size = (x_slice.len());
        let c_values: Vec<f32> = (0..batch_size)
            .map(|i| self.get_wave_speed_with_default(x_slice[i], y_slice[i], wave_speed as f32))
            .collect();

        let backend = B::default();
        let c_tensor =
            coeus_tensor::Tensor::from_slice_on(vec![batch_size, 1], &c_values, &backend);
        let c_var = Var::new(c_tensor, false);
        let c_squared = coeus_autograd::mul(&c_var, &c_var);

        coeus_autograd::sub(&u_tt, &coeus_autograd::mul(&laplacian, &c_squared))
    }

    /// Compute physics-informed loss function.
    // Independent collocation/boundary/initial tensors plus scalar weights with
    // no cohesive sub-grouping; bundling would not clarify the call site.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // (total, data, pde, bc, ic) mirrors the previous training API 1:1
    pub fn compute_physics_loss(
        &self,
        x_data: &Var<f32, B>,
        y_data: &Var<f32, B>,
        t_data: &Var<f32, B>,
        u_data: &Var<f32, B>,
        x_collocation: &Var<f32, B>,
        y_collocation: &Var<f32, B>,
        t_collocation: &Var<f32, B>,
        x_boundary: &Var<f32, B>,
        y_boundary: &Var<f32, B>,
        t_boundary: &Var<f32, B>,
        u_boundary: &Var<f32, B>,
        x_initial: &Var<f32, B>,
        y_initial: &Var<f32, B>,
        t_initial: &Var<f32, B>,
        u_initial: &Var<f32, B>,
        wave_speed: f64,
        loss_weights: LossWeights2D,
    ) -> PhysicsLossComponents<B> {
        let u_pred_data = self.forward(x_data, y_data, t_data);
        let data_diff = coeus_autograd::sub(&u_pred_data, u_data);
        let data_loss = coeus_autograd::mean(&coeus_autograd::mul(&data_diff, &data_diff));

        let residual =
            self.compute_pde_residual(x_collocation, y_collocation, t_collocation, wave_speed);
        let pde_loss_raw = coeus_autograd::mean(&coeus_autograd::mul(&residual, &residual));
        let pde_loss = coeus_autograd::scalar_mul(&pde_loss_raw, 1e-12);

        let u_pred_boundary = self.forward(x_boundary, y_boundary, t_boundary);
        let bc_diff = coeus_autograd::sub(&u_pred_boundary, u_boundary);
        let bc_loss = coeus_autograd::mean(&coeus_autograd::mul(&bc_diff, &bc_diff));

        let u_pred_initial = self.forward(x_initial, y_initial, t_initial);
        let ic_diff = coeus_autograd::sub(&u_pred_initial, u_initial);
        let ic_loss = coeus_autograd::mean(&coeus_autograd::mul(&ic_diff, &ic_diff));

        let total_loss = coeus_autograd::add(
            &coeus_autograd::add(
                &coeus_autograd::scalar_mul(&data_loss, loss_weights.data as f32),
                &coeus_autograd::scalar_mul(&pde_loss, loss_weights.pde as f32),
            ),
            &coeus_autograd::add(
                &coeus_autograd::scalar_mul(&bc_loss, loss_weights.boundary as f32),
                &coeus_autograd::scalar_mul(&ic_loss, loss_weights.initial as f32),
            ),
        );

        (total_loss, data_loss, pde_loss, bc_loss, ic_loss)
    }
}

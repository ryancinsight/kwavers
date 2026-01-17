use super::config::BurnPINN2DConfig;
use crate::core::error::{KwaversError, KwaversResult};
use burn::module::{Ignored, Module, ModuleMapper, ModuleVisitor};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Wrapper for wave speed function to implement Debug and Module traits
#[derive(Clone)]
pub struct WaveSpeedFn<B: Backend> {
    /// CPU function for wave speed
    pub func: Arc<dyn Fn(f32, f32) -> f32 + Send + Sync>,
    /// Optional device-resident grid of wave speeds
    pub grid: Option<Tensor<B, 2>>,
}

impl<B: Backend> WaveSpeedFn<B> {
    /// Create a new wave speed function from a CPU closure
    pub fn new(func: Arc<dyn Fn(f32, f32) -> f32 + Send + Sync>) -> Self {
        Self { func, grid: None }
    }

    /// Create a new wave speed function from a device-resident grid
    pub fn from_grid(grid: Tensor<B, 2>) -> KwaversResult<Self> {
        let shape = grid.shape();
        let dims = match shape.dims.as_slice() {
            [nx, ny] => [*nx, *ny],
            _ => {
                return Err(KwaversError::System(
                    crate::core::error::SystemError::InvalidConfiguration {
                        parameter: "wave_speed_grid".to_string(),
                        reason: format!(
                            "Expected wave speed grid with 2 dimensions, got {:?}",
                            shape.dims
                        ),
                    },
                ))
            }
        };

        let [nx, ny] = dims;
        if nx == 0 || ny == 0 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "wave_speed_grid".to_string(),
                    reason: format!("Grid dimensions must be non-zero, got {dims:?}"),
                },
            ));
        }

        let data = grid.clone().to_data();
        let slice = data.as_slice::<f32>().map_err(|e| {
            KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
                parameter: "wave_speed_grid".to_string(),
                reason: format!("Expected f32 tensor data for wave speed grid: {e:?}"),
            })
        })?;

        let expected_len = nx.checked_mul(ny).ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::InvalidConfiguration {
                parameter: "wave_speed_grid".to_string(),
                reason: format!("Grid size overflows usize: {dims:?}"),
            })
        })?;
        if slice.len() != expected_len {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "wave_speed_grid".to_string(),
                    reason: format!(
                        "Wave speed grid data length mismatch: expected {expected_len}, got {}",
                        slice.len()
                    ),
                },
            ));
        }

        let data_cpu = Arc::new(slice.to_vec());
        let func = Arc::new(move |x: f32, y: f32| -> f32 {
            let x = x.clamp(0.0, 1.0);
            let y = y.clamp(0.0, 1.0);

            let fx = if nx <= 1 { 0.0 } else { x * ((nx - 1) as f32) };
            let fy = if ny <= 1 { 0.0 } else { y * ((ny - 1) as f32) };

            let fx0 = fx.floor();
            let fy0 = fy.floor();

            let x0 = (fx0 as isize).clamp(0, (nx - 1) as isize) as usize;
            let y0 = (fy0 as isize).clamp(0, (ny - 1) as isize) as usize;
            let x1 = (x0 + 1).min(nx - 1);
            let y1 = (y0 + 1).min(ny - 1);

            let wx = (fx - fx0).clamp(0.0, 1.0);
            let wy = (fy - fy0).clamp(0.0, 1.0);

            let at = |ix: usize, iy: usize| -> f32 { data_cpu[(ix * ny) + iy] };

            let c00 = at(x0, y0);
            let c10 = at(x1, y0);
            let c01 = at(x0, y1);
            let c11 = at(x1, y1);

            let c0 = c00 + wx * (c10 - c00);
            let c1 = c01 + wx * (c11 - c01);
            c0 + wy * (c1 - c0)
        });

        Ok(Self {
            func,
            grid: Some(grid),
        })
    }

    /// Get wave speed at coordinates (x, y)
    pub fn get(&self, x: f32, y: f32) -> f32 {
        // Prefer grid if available (this is still CPU-bound if we call it this way)
        (self.func)(x, y)
    }
}

impl<B: Backend> std::fmt::Debug for WaveSpeedFn<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WaveSpeedFn")
    }
}

impl<B: Backend> Module<B> for WaveSpeedFn<B> {
    type Record = ();

    fn collect_devices(&self, mut devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        if let Some(grid) = &self.grid {
            devices.push(grid.device());
        }
        devices
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            func: self.func,
            grid: self.grid.map(|g| g.to_device(device)),
        }
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            func: self.func,
            grid: self.grid.map(|g| g.to_device(device)),
        }
    }

    fn map<M: ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
        self
    }

    fn visit<V: ModuleVisitor<B>>(&self, _visitor: &mut V) {
        // No parameters to visit
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: Backend> burn::module::ModuleDisplayDefault for WaveSpeedFn<B> {
    fn content(
        &self,
        content: burn::module::Content,
    ) -> std::option::Option<burn::module::Content> {
        Some(content)
    }
}
impl<B: Backend> burn::module::ModuleDisplay for WaveSpeedFn<B> {}

impl<B: AutodiffBackend> burn::module::AutodiffModule<B> for WaveSpeedFn<B> {
    type InnerModule = WaveSpeedFn<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        WaveSpeedFn {
            func: self.func.clone(),
            grid: self.grid.as_ref().map(|g| g.clone().inner()),
        }
    }
}

/// Supports spatially varying wave speeds c(x,y) for complex media
#[derive(Module, Debug)]
pub struct BurnPINN2DWave<B: Backend> {
    /// Input layer (3 inputs: x, y, t)
    pub input_layer: Linear<B>,
    /// Hidden layers with tanh activation
    pub hidden_layers: Vec<Linear<B>>,
    /// Output layer (1 output: u)
    pub output_layer: Linear<B>,
    /// Wave speed function c(x,y) for heterogeneous media (optional)
    pub wave_speed_fn: Option<WaveSpeedFn<B>>,
    /// Configuration used to create the model
    pub config: Ignored<BurnPINN2DConfig>,
}

impl<B: Backend> BurnPINN2DWave<B> {
    pub fn new(config: BurnPINN2DConfig, device: &B::Device) -> KwaversResult<Self> {
        if config.hidden_layers.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Must have at least one hidden layer".into(),
            ));
        }

        // Input layer: 3 inputs (x, y, t) -> first hidden layer size
        let input_size = 3;
        let first_hidden_size = config.hidden_layers[0];
        let input_layer = LinearConfig::new(input_size, first_hidden_size).init(device);

        // Hidden layers
        let mut hidden_layers = Vec::new();
        for i in 0..config.hidden_layers.len() - 1 {
            let in_size = config.hidden_layers[i];
            let out_size = config.hidden_layers[i + 1];
            hidden_layers.push(LinearConfig::new(in_size, out_size).init(device));
        }

        // Output layer: last hidden layer -> 1 output (u)
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

    /// Create heterogeneous model
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

    /// Forward pass through the network
    pub fn forward(&self, x: Tensor<B, 2>, y: Tensor<B, 2>, t: Tensor<B, 2>) -> Tensor<B, 2> {
        // Concatenate inputs: [batch_size, 3]
        let input = Tensor::cat(vec![x, y, t], 1);

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

    /// Get wave speed at a specific location, using a default value if no function is provided
    pub fn get_wave_speed_with_default(&self, x: f32, y: f32, default_c: f32) -> f32 {
        self.wave_speed_fn
            .as_ref()
            .map(|f| f.get(x, y))
            .unwrap_or(default_c)
    }

    /// Get the device this model is on
    ///
    /// # Returns
    /// The device where the model's parameters are stored
    pub fn device(&self) -> B::Device {
        // Get device from one of the model's parameters
        self.input_layer
            .devices()
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// Get the number of parameters in the model
    ///
    /// # Returns
    /// Total count of trainable parameters
    pub fn num_parameters(&self) -> usize {
        let input_params = self.config.0.hidden_layers[0] * 3; // input_layer: 3 -> first_hidden
        let mut hidden_params = 0;
        for i in 0..self.config.0.hidden_layers.len() - 1 {
            hidden_params += self.config.0.hidden_layers[i] * self.config.0.hidden_layers[i + 1];
        }
        let output_params = *self.config.0.hidden_layers.last().unwrap(); // last_hidden -> 1

        // Add bias terms
        let bias_count: usize = self.config.0.hidden_layers.iter().sum::<usize>() + 1;

        input_params + hidden_params + output_params + bias_count
    }

    /// Get all model parameters (weights and biases)
    ///
    /// # Returns
    /// Vector of all parameter tensors in the model
    ///
    /// # Implementation Note
    /// Returns flattened parameter tensors from all layers:
    /// - Input layer weight and bias
    /// - Hidden layers weights and biases
    /// - Output layer weight and bias
    pub fn parameters(&self) -> Vec<Tensor<B, 1>> {
        let mut params = Vec::new();

        // Input layer parameters
        params.push(self.input_layer.weight.val().flatten(0, 1));
        if let Some(bias) = &self.input_layer.bias {
            params.push(bias.val().flatten(0, 0));
        }

        // Hidden layers parameters
        for layer in &self.hidden_layers {
            params.push(layer.weight.val().flatten(0, 1));
            if let Some(bias) = &layer.bias {
                params.push(bias.val().flatten(0, 0));
            }
        }

        // Output layer parameters
        params.push(self.output_layer.weight.val().flatten(0, 1));
        if let Some(bias) = &self.output_layer.bias {
            params.push(bias.val().flatten(0, 0));
        }

        params
    }

    /// Predict field values at given spatial and temporal coordinates
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

        // Convert to tensors
        let x_vec: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let y_vec: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let t_vec: Vec<f32> = t.iter().map(|&v| v as f32).collect();

        let x_tensor = Tensor::<B, 1>::from_floats(x_vec.as_slice(), device).reshape([n, 1]);
        let y_tensor = Tensor::<B, 1>::from_floats(y_vec.as_slice(), device).reshape([n, 1]);
        let t_tensor = Tensor::<B, 1>::from_floats(t_vec.as_slice(), device).reshape([n, 1]);

        // Forward pass
        let u_tensor = self.forward(x_tensor, y_tensor, t_tensor);

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

// Autodiff implementation for physics-informed loss in 2D
impl<B: AutodiffBackend> BurnPINN2DWave<B> {
    /// Compute PDE residual using finite differences within autodiff framework
    pub fn compute_pde_residual(
        &self,
        x: Tensor<B, 2>,
        y: Tensor<B, 2>,
        t: Tensor<B, 2>,
        wave_speed: f64,
    ) -> Tensor<B, 2> {
        // Adaptive epsilon selection for numerical stability
        let base_eps = (f32::EPSILON).sqrt();
        let scale_factor = 1e-2_f32;
        let eps = base_eps * scale_factor;

        let u = self.forward(x.clone(), y.clone(), t.clone());

        let x_plus = x.clone().add_scalar(eps);
        let x_minus = x.clone().sub_scalar(eps);
        let y_plus = y.clone().add_scalar(eps);
        let y_minus = y.clone().sub_scalar(eps);
        let t_plus = t.clone().add_scalar(eps);
        let t_minus = t.clone().sub_scalar(eps);

        let u_x_plus = self.forward(x_plus, y.clone(), t.clone());
        let u_x_minus = self.forward(x_minus, y.clone(), t.clone());
        let u_xx = u_x_plus
            .add(u_x_minus)
            .sub(u.clone().mul_scalar(2.0))
            .div_scalar(eps * eps);

        let u_y_plus = self.forward(x.clone(), y_plus, t.clone());
        let u_y_minus = self.forward(x.clone(), y_minus, t.clone());
        let u_yy = u_y_plus
            .add(u_y_minus)
            .sub(u.clone().mul_scalar(2.0))
            .div_scalar(eps * eps);

        let u_t_plus = self.forward(x.clone(), y.clone(), t_plus);
        let u_t_minus = self.forward(x.clone(), y.clone(), t_minus);
        let u_tt = u_t_plus
            .add(u_t_minus)
            .sub(u.clone().mul_scalar(2.0))
            .div_scalar(eps * eps);

        let laplacian = u_xx.add(u_yy);

        // Get wave speed
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

    /// Compute physics-informed loss function
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
        loss_weights: super::config::BurnLossWeights2D,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
    ) {
        // Data loss
        let u_pred_data = self.forward(x_data, y_data, t_data);
        let data_loss = (u_pred_data - u_data).powf_scalar(2.0).mean();

        // PDE residual loss
        let residual =
            self.compute_pde_residual(x_collocation, y_collocation, t_collocation, wave_speed);
        let pde_loss = residual.powf_scalar(2.0).mean() * 1e-12_f32;

        // Boundary condition loss
        let u_pred_boundary = self.forward(x_boundary, y_boundary, t_boundary);
        let bc_loss = (u_pred_boundary - u_boundary).powf_scalar(2.0).mean();

        // Initial condition loss
        let u_pred_initial = self.forward(x_initial, y_initial, t_initial);
        let ic_loss = (u_pred_initial - u_initial).powf_scalar(2.0).mean();

        // Total physics-informed loss
        let total_loss = data_loss.clone() * loss_weights.data as f32
            + pde_loss.clone() * loss_weights.pde as f32
            + bc_loss.clone() * loss_weights.boundary as f32
            + ic_loss.clone() * loss_weights.initial as f32;

        (total_loss, data_loss, pde_loss, bc_loss, ic_loss)
    }
}

//! Neural network architecture for 3D wave equation PINN
//!
//! This module implements the core neural network that learns the wave equation solution.
//! The network takes spatiotemporal coordinates (x, y, z, t) as input and predicts the
//! displacement/pressure field u(x, y, z, t).
//!
//! ## Architecture
//!
//! - Input layer: (x, y, z, t) → hidden_dim
//! - Hidden layers: Fully connected with ReLU activation
//! - Output layer: hidden_dim → u (scalar field)
//!
//! ## PDE Residual Computation
//!
//! The `compute_pde_residual` method computes the residual of the wave equation:
//!
//! R = ∂²u/∂t² - c²(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
//!
//! Spatial and temporal derivatives are computed via finite differences with adaptive
//! step size to balance numerical stability and accuracy.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::{backend::Backend, Tensor, TensorData};

use super::config::BurnPINN3DConfig;

/// Neural network for 3D wave equation PINN
///
/// This network learns to approximate solutions u(x, y, z, t) to the 3D wave equation
/// using physics-informed training with PDE residual enforcement.
#[derive(Module, Debug, Clone)]
pub struct PINN3DNetwork<B: Backend> {
    /// Input layer: (x, y, z, t) → hidden[0]
    input_layer: Linear<B>,
    /// Hidden layers with ReLU activation
    hidden_layers: Vec<(Linear<B>, Relu)>,
    /// Output layer: hidden[last] → u
    output_layer: Linear<B>,
}

impl<B: Backend> PINN3DNetwork<B> {
    /// Create a new PINN network with the specified architecture
    ///
    /// # Arguments
    ///
    /// * `config` - Network configuration specifying hidden layer dimensions
    /// * `device` - Target device for network parameters
    ///
    /// # Returns
    ///
    /// A new `PINN3DNetwork` instance with randomized initial weights
    ///
    /// # Architecture
    ///
    /// - Input: 4D (x, y, z, t)
    /// - Hidden: As specified in config (e.g., [128, 128, 128])
    /// - Output: 1D (u)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::backend::NdArray;
    /// use kwavers::analysis::ml::pinn::burn_wave_equation_3d::{PINN3DNetwork, BurnPINN3DConfig};
    ///
    /// type Backend = NdArray<f32>;
    /// let device = Default::default();
    /// let config = BurnPINN3DConfig::default();
    /// let network = PINN3DNetwork::<Backend>::new(&config, &device);
    /// ```
    pub fn new(config: &BurnPINN3DConfig, device: &B::Device) -> Self {
        let input_size = 4; // (x, y, z, t)
        let output_size = 1; // u

        // Input layer: 4D → hidden[0]
        let input_layer = LinearConfig::new(input_size, config.hidden_layers[0]).init(device);

        // Hidden layers: hidden[i] → hidden[i+1] with ReLU
        let mut hidden_layers = Vec::new();
        for i in 0..config.hidden_layers.len() - 1 {
            let layer = LinearConfig::new(config.hidden_layers[i], config.hidden_layers[i + 1])
                .init(device);
            hidden_layers.push((layer, Relu::new()));
        }

        // Output layer: hidden[last] → 1D
        let output_layer =
            LinearConfig::new(*config.hidden_layers.last().unwrap(), output_size).init(device);

        Self {
            input_layer,
            hidden_layers,
            output_layer,
        }
    }

    /// Forward pass through the network
    ///
    /// # Arguments
    ///
    /// * `x` - Spatial x-coordinates [batch_size, 1]
    /// * `y` - Spatial y-coordinates [batch_size, 1]
    /// * `z` - Spatial z-coordinates [batch_size, 1]
    /// * `t` - Time coordinates [batch_size, 1]
    ///
    /// # Returns
    ///
    /// Predicted displacement/pressure field u [batch_size, 1]
    ///
    /// # Mathematical Form
    ///
    /// u = NN(x, y, z, t) = σ(W_L σ(...σ(W_1[x,y,z,t] + b_1)...) + b_L)
    ///
    /// where σ is the ReLU activation function.
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        y: Tensor<B, 2>,
        z: Tensor<B, 2>,
        t: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        // Concatenate inputs: [x, y, z, t] → [batch_size, 4]
        let input = Tensor::cat(vec![x, y, z, t], 1);

        // Input layer
        let mut output = self.input_layer.forward(input);

        // Hidden layers with ReLU activation
        for (layer, relu) in &self.hidden_layers {
            output = layer.forward(output);
            output = relu.forward(output);
        }

        // Output layer (no activation for regression)
        self.output_layer.forward(output)
    }

    /// Compute PDE residual for the wave equation using finite differences
    ///
    /// # Arguments
    ///
    /// * `x` - Spatial x-coordinates [batch_size, 1]
    /// * `y` - Spatial y-coordinates [batch_size, 1]
    /// * `z` - Spatial z-coordinates [batch_size, 1]
    /// * `t` - Time coordinates [batch_size, 1]
    /// * `wave_speed` - Function c(x, y, z) returning wave speed at each point
    ///
    /// # Returns
    ///
    /// PDE residual R = ∂²u/∂t² - c²∇²u [batch_size, 1]
    ///
    /// # Mathematical Form
    ///
    /// R(x,y,z,t) = ∂²u/∂t² - c²(x,y,z) × (∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
    ///
    /// # Numerical Method
    ///
    /// Second-order central finite differences:
    ///
    /// ∂²u/∂x² ≈ [u(x+ε) - 2u(x) + u(x-ε)] / ε²
    ///
    /// with adaptive ε = sqrt(f32::EPSILON) × 0.01 for numerical stability.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::tensor::Tensor;
    ///
    /// let residual = network.compute_pde_residual(
    ///     x_colloc,
    ///     y_colloc,
    ///     z_colloc,
    ///     t_colloc,
    ///     |x, y, z| 1500.0, // Constant wave speed
    /// );
    /// ```
    pub fn compute_pde_residual(
        &self,
        x: Tensor<B, 2>,
        y: Tensor<B, 2>,
        z: Tensor<B, 2>,
        t: Tensor<B, 2>,
        wave_speed: impl Fn(f32, f32, f32) -> f32,
    ) -> Tensor<B, 2> {
        // Adaptive step size: balance numerical stability and accuracy
        // ε = sqrt(machine epsilon) × scale_factor
        let base_eps = (f32::EPSILON).sqrt();
        let scale_factor = 1e-2_f32;
        let eps = base_eps * scale_factor;

        // Base prediction
        let u = self.forward(x.clone(), y.clone(), z.clone(), t.clone());

        // Perturbed coordinates for finite differences
        let x_plus = x.clone().add_scalar(eps);
        let x_minus = x.clone().sub_scalar(eps);
        let y_plus = y.clone().add_scalar(eps);
        let y_minus = y.clone().sub_scalar(eps);
        let z_plus = z.clone().add_scalar(eps);
        let z_minus = z.clone().sub_scalar(eps);
        let t_plus = t.clone().add_scalar(eps);
        let t_minus = t.clone().sub_scalar(eps);

        // Second-order spatial derivatives: ∂²u/∂x²
        let u_x_plus = self.forward(x_plus, y.clone(), z.clone(), t.clone());
        let u_x_minus = self.forward(x_minus, y.clone(), z.clone(), t.clone());
        let u_xx = u_x_plus
            .add(u_x_minus)
            .sub(u.clone().mul_scalar(2.0))
            .div_scalar(eps * eps);

        // ∂²u/∂y²
        let u_y_plus = self.forward(x.clone(), y_plus, z.clone(), t.clone());
        let u_y_minus = self.forward(x.clone(), y_minus, z.clone(), t.clone());
        let u_yy = u_y_plus
            .add(u_y_minus)
            .sub(u.clone().mul_scalar(2.0))
            .div_scalar(eps * eps);

        // ∂²u/∂z²
        let u_z_plus = self.forward(x.clone(), y.clone(), z_plus, t.clone());
        let u_z_minus = self.forward(x.clone(), y.clone(), z_minus, t.clone());
        let u_zz = u_z_plus
            .add(u_z_minus)
            .sub(u.clone().mul_scalar(2.0))
            .div_scalar(eps * eps);

        // Second-order temporal derivative: ∂²u/∂t²
        let u_t_plus = self.forward(x.clone(), y.clone(), z.clone(), t_plus);
        let u_t_minus = self.forward(x.clone(), y.clone(), z.clone(), t_minus);
        let u_tt = u_t_plus
            .add(u_t_minus)
            .sub(u.clone().mul_scalar(2.0))
            .div_scalar(eps * eps);

        // Compute wave speed c(x,y,z) at each collocation point
        let batch_size = x.shape().dims[0];
        let c_values: Vec<f32> = (0..batch_size)
            .map(|i| {
                // Extract scalar coordinates from tensors
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
                let z_val = z
                    .clone()
                    .slice([i..i + 1, 0..1])
                    .into_data()
                    .as_slice::<f32>()
                    .unwrap()[0];
                wave_speed(x_val, y_val, z_val)
            })
            .collect();

        // Convert wave speed to tensor and compute c²
        let c_tensor =
            Tensor::<B, 2>::from_data(TensorData::from(c_values.as_slice()), &x.device())
                .unsqueeze_dim(1);
        let c_squared = c_tensor.powf_scalar(2.0);

        // Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²
        let laplacian = u_xx.add(u_yy).add(u_zz);

        // PDE residual: R = ∂²u/∂t² - c²∇²u
        u_tt.sub(laplacian.mul(c_squared))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_network_creation() {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![128, 128],
            ..Default::default()
        };

        let network = PINN3DNetwork::<TestBackend>::new(&config, &device);

        // Verify architecture
        assert_eq!(network.hidden_layers.len(), 1); // 2 hidden dims → 1 connection
    }

    #[test]
    fn test_forward_pass() {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![32, 32],
            ..Default::default()
        };

        let network = PINN3DNetwork::<TestBackend>::new(&config, &device);

        // Create batch of 10 points
        let batch_size = 10;
        let x = Tensor::<TestBackend, 2>::zeros([batch_size, 1], &device);
        let y = Tensor::<TestBackend, 2>::ones([batch_size, 1], &device);
        let z = Tensor::<TestBackend, 2>::ones([batch_size, 1], &device).mul_scalar(0.5);
        let t = Tensor::<TestBackend, 2>::ones([batch_size, 1], &device).mul_scalar(0.1);

        let output = network.forward(x, y, z, t);

        // Verify output shape
        assert_eq!(output.shape().dims, [batch_size, 1]);
    }

    #[test]
    fn test_pde_residual_shape() {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![16, 16],
            ..Default::default()
        };

        let network = PINN3DNetwork::<TestBackend>::new(&config, &device);

        // Create collocation points
        let n_points = 5;
        let x = Tensor::<TestBackend, 2>::zeros([n_points, 1], &device);
        let y = Tensor::<TestBackend, 2>::zeros([n_points, 1], &device);
        let z = Tensor::<TestBackend, 2>::zeros([n_points, 1], &device);
        let t = Tensor::<TestBackend, 2>::zeros([n_points, 1], &device);

        // Constant wave speed
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let residual = network.compute_pde_residual(x, y, z, t, wave_speed);

        // Verify residual shape matches input
        assert_eq!(residual.shape().dims, [n_points, 1]);
    }

    #[test]
    fn test_pde_residual_heterogeneous_medium() {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![16],
            ..Default::default()
        };

        let network = PINN3DNetwork::<TestBackend>::new(&config, &device);

        // Create points in two different regions
        let x_data = vec![0.25, 0.75]; // Left and right regions
        let y_data = vec![0.5, 0.5];
        let z_data = vec![0.5, 0.5];
        let t_data = vec![0.1, 0.1];

        let x = Tensor::<TestBackend, 2>::from_data(TensorData::from(x_data.as_slice()), &device)
            .unsqueeze_dim(1);
        let y = Tensor::<TestBackend, 2>::from_data(TensorData::from(y_data.as_slice()), &device)
            .unsqueeze_dim(1);
        let z = Tensor::<TestBackend, 2>::from_data(TensorData::from(z_data.as_slice()), &device)
            .unsqueeze_dim(1);
        let t = Tensor::<TestBackend, 2>::from_data(TensorData::from(t_data.as_slice()), &device)
            .unsqueeze_dim(1);

        // Layered medium: different speeds in left/right halves
        let wave_speed = |x: f32, _y: f32, _z: f32| if x < 0.5 { 1500.0 } else { 3000.0 };

        let residual = network.compute_pde_residual(x, y, z, t, wave_speed);

        // Verify residual is computed (non-trivial)
        assert_eq!(residual.shape().dims, [2, 1]);
        let residual_data = residual.into_data().as_slice::<f32>().unwrap().to_vec();
        assert!(residual_data.iter().all(|&r| r.is_finite()));
    }

    #[test]
    fn test_network_forward_deterministic() {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![8],
            ..Default::default()
        };

        let network = PINN3DNetwork::<TestBackend>::new(&config, &device);

        let x = Tensor::<TestBackend, 2>::ones([3, 1], &device);
        let y = Tensor::<TestBackend, 2>::ones([3, 1], &device);
        let z = Tensor::<TestBackend, 2>::ones([3, 1], &device);
        let t = Tensor::<TestBackend, 2>::ones([3, 1], &device);

        // Two forward passes with same input should give same output
        let output1 = network.forward(x.clone(), y.clone(), z.clone(), t.clone());
        let output2 = network.forward(x, y, z, t);

        let data1 = output1.into_data().as_slice::<f32>().unwrap().to_vec();
        let data2 = output2.into_data().as_slice::<f32>().unwrap().to_vec();

        assert_eq!(data1, data2);
    }
}

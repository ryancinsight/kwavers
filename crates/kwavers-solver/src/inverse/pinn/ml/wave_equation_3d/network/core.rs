//! Neural network architecture for 3D wave equation PINN
//!
//! This module implements the core neural network that learns the wave equation solution.
//! The network takes spatiotemporal coordinates (x, y, z, t) as input and predicts the
//! displacement/pressure field u(x, y, z, t).
//!
//! ## Architecture
//!
//! - Input layer: (x, y, z, t) → hidden_dim
//! - Hidden layers: Fully connected with tanh activation
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

use coeus_autograd::Var;
use coeus_nn::{Linear, Module};

use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::config::PinnConfig3D;

/// Neural network for 3D wave equation PINN
///
/// This network learns to approximate solutions u(x, y, z, t) to the 3D wave equation
/// using physics-informed training with PDE residual enforcement.
pub struct PINN3DNetwork<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Input layer: (x, y, z, t) → hidden[0]
    input_layer: Linear<f32, B>,
    /// Hidden layers
    hidden_layers: Vec<Linear<f32, B>>,
    /// Output layer: hidden[last] → u
    output_layer: Linear<f32, B>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> Clone for PINN3DNetwork<B> {
    fn clone(&self) -> Self {
        Self {
            input_layer: self.input_layer.clone(),
            hidden_layers: self.hidden_layers.clone(),
            output_layer: self.output_layer.clone(),
        }
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for PINN3DNetwork<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PINN3DNetwork")
            .field("hidden_layer_count", &(self.hidden_layers.len()))
            .finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> PINN3DNetwork<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Create a new PINN network with the specified architecture
    ///
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    pub fn new(config: &PinnConfig3D) -> KwaversResult<Self> {
        config.validate()?;
        let input_size = 4; // (x, y, z, t)
        let output_size = 1; // u

        let first_hidden = *config
            .hidden_layers
            .first()
            .ok_or_else(|| KwaversError::InvalidInput("Hidden layers must be non-empty".into()))?;
        let input_layer = Linear::new(input_size, first_hidden, true);

        let mut hidden_layers = Vec::new();
        for window in config.hidden_layers.windows(2) {
            let [in_features, out_features] = window else {
                continue;
            };
            hidden_layers.push(Linear::new(*in_features, *out_features, true));
        }

        let last_hidden = *config
            .hidden_layers
            .last()
            .ok_or_else(|| KwaversError::InvalidInput("Hidden layers must be non-empty".into()))?;
        let output_layer = Linear::new(last_hidden, output_size, true);

        Ok(Self {
            input_layer,
            hidden_layers,
            output_layer,
        })
    }

    pub fn hidden_layer_count(&self) -> usize {
        self.hidden_layers.len()
    }

    /// Flatten all layer parameters (weights and biases) in forward order.
    pub fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut params = self.input_layer.parameters();
        for layer in &self.hidden_layers {
            params.extend(layer.parameters());
        }
        params.extend(self.output_layer.parameters());
        params
    }

    /// Write updated parameter values back into the network's layers.
    ///
    /// `coeus_tensor::Tensor` storage is copy-on-write, so a clone taken via
    /// `parameters()` detaches from the network on first mutation; an
    /// optimizer that mutates its own owned copy needs this round-trip to
    /// propagate updates back.
    pub fn load_parameters(&mut self, params: &[Var<f32, B>]) {
        let mut offset = 0;
        let n = self.input_layer.parameters().len();
        self.input_layer
            .load_parameters(&params[offset..offset + n]);
        offset += n;
        for layer in &mut self.hidden_layers {
            let n = layer.parameters().len();
            layer.load_parameters(&params[offset..offset + n]);
            offset += n;
        }
        let n = self.output_layer.parameters().len();
        self.output_layer
            .load_parameters(&params[offset..offset + n]);
    }

    /// Forward pass through the network
    ///
    /// u = NN(x, y, z, t) = tanh(W_L tanh(...tanh(W_1[x,y,z,t] + b_1)...) + b_L)
    pub fn forward(
        &self,
        x: &Var<f32, B>,
        y: &Var<f32, B>,
        z: &Var<f32, B>,
        t: &Var<f32, B>,
    ) -> Var<f32, B> {
        let input = coeus_autograd::cat(&[x, y, z, t], 1);
        let mut output = coeus_autograd::tanh(&self.input_layer.forward(&input));

        for layer in &self.hidden_layers {
            output = coeus_autograd::tanh(&layer.forward(&output));
        }

        self.output_layer.forward(&output)
    }

    /// Compute PDE residual for the wave equation using finite differences
    ///
    /// R(x,y,z,t) = ∂²u/∂t² - c²(x,y,z) × (∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
    ///
    /// Second-order central finite differences with adaptive ε = sqrt(f32::EPSILON) × 0.01.
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    pub fn compute_pde_residual(
        &self,
        x: &Var<f32, B>,
        y: &Var<f32, B>,
        z: &Var<f32, B>,
        t: &Var<f32, B>,
        wave_speed: impl Fn(f32, f32, f32) -> KwaversResult<f32>,
    ) -> KwaversResult<Var<f32, B>>
    where
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let eps = 1e-3_f32;

        let u = self.forward(x, y, z, t);

        let x_plus = coeus_autograd::scalar_add(x, eps);
        let x_minus = coeus_autograd::scalar_add(x, -eps);
        let y_plus = coeus_autograd::scalar_add(y, eps);
        let y_minus = coeus_autograd::scalar_add(y, -eps);
        let z_plus = coeus_autograd::scalar_add(z, eps);
        let z_minus = coeus_autograd::scalar_add(z, -eps);
        let t_plus = coeus_autograd::scalar_add(t, eps);
        let t_minus = coeus_autograd::scalar_add(t, -eps);

        let two_u = coeus_autograd::scalar_mul(&u, 2.0);

        let u_x_plus = self.forward(&x_plus, y, z, t);
        let u_x_minus = self.forward(&x_minus, y, z, t);
        let u_xx = coeus_autograd::scalar_mul(
            &coeus_autograd::sub(&coeus_autograd::add(&u_x_plus, &u_x_minus), &two_u),
            1.0 / (eps * eps),
        );

        let u_y_plus = self.forward(x, &y_plus, z, t);
        let u_y_minus = self.forward(x, &y_minus, z, t);
        let u_yy = coeus_autograd::scalar_mul(
            &coeus_autograd::sub(&coeus_autograd::add(&u_y_plus, &u_y_minus), &two_u),
            1.0 / (eps * eps),
        );

        let u_z_plus = self.forward(x, y, &z_plus, t);
        let u_z_minus = self.forward(x, y, &z_minus, t);
        let u_zz = coeus_autograd::scalar_mul(
            &coeus_autograd::sub(&coeus_autograd::add(&u_z_plus, &u_z_minus), &two_u),
            1.0 / (eps * eps),
        );

        let u_t_plus = self.forward(x, y, z, &t_plus);
        let u_t_minus = self.forward(x, y, z, &t_minus);
        let u_tt = coeus_autograd::scalar_mul(
            &coeus_autograd::sub(&coeus_autograd::add(&u_t_plus, &u_t_minus), &two_u),
            1.0 / (eps * eps),
        );

        // Compute wave speed c(x,y,z) at each collocation point
        let x_vals = x.tensor.as_slice();
        let y_vals = y.tensor.as_slice();
        let z_vals = z.tensor.as_slice();
        let c_values: Vec<f32> = x_vals
            .iter()
            .zip(y_vals.iter())
            .zip(z_vals.iter())
            .map(|((&xv, &yv), &zv)| wave_speed(xv, yv, zv))
            .collect::<KwaversResult<Vec<f32>>>()?;

        for &c in &c_values {
            if !c.is_finite() || c <= 0.0 {
                return Err(KwaversError::InvalidInput(
                    "Wave speed values must be finite and positive".into(),
                ));
            }
        }
        let c_squared: Vec<f32> = c_values.iter().map(|&c| c * c).collect();

        let backend = B::default();
        let c_squared_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![(c_squared.len()), 1], &c_squared, &backend),
            false,
        );

        // Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²
        let laplacian = coeus_autograd::add(&coeus_autograd::add(&u_xx, &u_yy), &u_zz);

        // PDE residual: R = ∂²u/∂t² - c²∇²u
        Ok(coeus_autograd::sub(
            &u_tt,
            &coeus_autograd::mul(&laplacian, &c_squared_var),
        ))
    }
}

//! Burn-based GPU Acceleration Framework
//!
//! This module provides high-level GPU acceleration using the Burn deep learning framework.
//! It offers better integration with scientific computing workflows and automatic differentiation
//! capabilities compared to raw wgpu implementations.
//!
//! ## Features
//!
//! - **Unified GPU Interface**: Single interface for different GPU operations
//! - **Automatic Differentiation**: Native support for gradient computations
//! - **Memory Management**: Efficient tensor operations and memory pooling
//! - **Multi-Backend Support**: CPU, WGPU, CUDA (via Burn backends)
//! - **Scientific Computing**: Optimized for PDEs, wave propagation, and physics simulations

use crate::core::error::{KwaversError, KwaversResult};
use burn::prelude::*;
use burn::tensor::{backend::Backend, Tensor};
use ndarray::Array3;
use std::marker::PhantomData;

/// Burn-based GPU accelerator for scientific computing
#[derive(Debug)]
pub struct BurnGpuAccelerator<B: Backend> {
    /// Burn device for tensor operations
    device: B::Device,
    /// Backend type marker
    _backend: PhantomData<B>,
}

/// Operations supported by the GPU accelerator
#[derive(Debug, Clone)]
pub enum GpuOperation {
    /// Acoustic wave propagation (pressure update)
    AcousticPropagation {
        dt: f64,
        sound_speed: f64,
        density: f64,
    },
    /// Electromagnetic wave propagation
    ElectromagneticPropagation {
        dt: f64,
        permittivity: f64,
        permeability: f64,
    },
    /// PDE residual computation for PINN
    PdeResidual,
    /// Matrix operations (FFT, convolution, etc.)
    MatrixOperation,
    /// Custom user-defined operation
    Custom(String),
}

/// Configuration for GPU operations
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Preferred backend (WGPU, CUDA, etc.)
    pub backend: String,
    /// Memory allocation strategy
    pub memory_strategy: MemoryStrategy,
    /// Precision for computations
    pub precision: Precision,
}

/// Memory allocation strategies
#[derive(Debug, Clone)]
pub enum MemoryStrategy {
    /// Pre-allocate fixed memory pools
    Pooled,
    /// Dynamic allocation as needed
    Dynamic,
    /// Memory-mapped for large datasets
    Mapped,
}

/// Precision for GPU computations
#[derive(Debug, Clone, Copy)]
pub enum Precision {
    /// Single precision (f32)
    F32,
    /// Double precision (f64)
    F64,
    /// Mixed precision (f16/f32)
    Mixed,
}

impl<B: Backend> BurnGpuAccelerator<B> {
    /// Create new GPU accelerator
    pub fn new(config: &GpuConfig) -> KwaversResult<Self>
    where
        B::Device: Default,
    {
        if !config.enable_gpu {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: "GPU acceleration disabled".to_string(),
                },
            ));
        }

        let device = B::Device::default();

        Ok(Self {
            device,
            _backend: PhantomData,
        })
    }

    /// Get the Burn device
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Convert ndarray Array3 to Burn tensor
    pub fn array_to_tensor(&self, array: &Array3<f64>) -> Tensor<B, 3> {
        let shape = array.shape();
        let data: Vec<f32> = array.iter().map(|&x| x as f32).collect();

        Tensor::<B, 1>::from_data(TensorData::from(data.as_slice()), &self.device)
            .reshape([shape[0], shape[1], shape[2]])
    }

    /// Convert Burn tensor to ndarray Array3
    pub fn tensor_to_array(&self, tensor: Tensor<B, 3>) -> Array3<f64> {
        let shape = tensor.shape();
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().unwrap();

        Array3::from_shape_vec(
            (shape[0], shape[1], shape[2]),
            slice.iter().map(|&x| x as f64).collect(),
        )
        .unwrap()
    }

    /// Execute acoustic wave propagation on GPU
    pub fn propagate_acoustic_wave(
        &self,
        pressure: &Array3<f64>,
        velocity_x: &Array3<f64>,
        velocity_y: &Array3<f64>,
        velocity_z: &Array3<f64>,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
        dt: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Convert inputs to tensors
        let p_tensor = self.array_to_tensor(pressure);
        let vx_tensor = self.array_to_tensor(velocity_x);
        let vy_tensor = self.array_to_tensor(velocity_y);
        let vz_tensor = self.array_to_tensor(velocity_z);
        let rho_tensor = self.array_to_tensor(density);
        let c_tensor = self.array_to_tensor(sound_speed);

        // Compute divergence: ∇·v
        let divergence = self.compute_divergence(&vx_tensor, &vy_tensor, &vz_tensor, dx, dy, dz);

        // Update pressure: p^{n+1} = p^n - dt * ρ * c² * ∇·v
        let c_squared = c_tensor.clone().powf_scalar(2.0);
        let rho_c_squared = rho_tensor * c_squared;
        let pressure_update = (divergence * rho_c_squared).mul_scalar(dt as f32);
        let new_pressure = p_tensor - pressure_update;

        // Convert back to array
        Ok(self.tensor_to_array(new_pressure))
    }

    /// Compute divergence of velocity field
    fn compute_divergence(
        &self,
        vx: &Tensor<B, 3>,
        vy: &Tensor<B, 3>,
        vz: &Tensor<B, 3>,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> Tensor<B, 3> {
        // ∂vx/∂x
        let dvx_dx = self.compute_gradient_x(vx, dx as f32);

        // ∂vy/∂y
        let dvy_dy = self.compute_gradient_y(vy, dy as f32);

        // ∂vz/∂z
        let dvz_dz = self.compute_gradient_z(vz, dz as f32);

        dvx_dx + dvy_dy + dvz_dz
    }

    /// Compute gradient in x-direction using central differences
    fn compute_gradient_x(&self, field: &Tensor<B, 3>, dx: f32) -> Tensor<B, 3> {
        let shifted_right = field.clone().roll(&[1], &[0]);
        let shifted_left = field.clone().roll(&[-1], &[0]);
        (shifted_right - shifted_left).div_scalar(2.0 * dx)
    }

    /// Compute gradient in y-direction using central differences
    fn compute_gradient_y(&self, field: &Tensor<B, 3>, dy: f32) -> Tensor<B, 3> {
        let shifted_up = field.clone().roll(&[1], &[1]);
        let shifted_down = field.clone().roll(&[-1], &[1]);
        (shifted_up - shifted_down).div_scalar(2.0 * dy)
    }

    /// Compute gradient in z-direction using central differences
    fn compute_gradient_z(&self, field: &Tensor<B, 3>, dz: f32) -> Tensor<B, 3> {
        let shifted_front = field.clone().roll(&[1], &[2]);
        let shifted_back = field.clone().roll(&[-1], &[2]);
        (shifted_front - shifted_back).div_scalar(2.0 * dz)
    }

    /// Execute electromagnetic wave propagation on GPU
    pub fn propagate_electromagnetic_wave(
        &self,
        ex: &Array3<f64>,
        ey: &Array3<f64>,
        ez: &Array3<f64>,
        hx: &Array3<f64>,
        hy: &Array3<f64>,
        hz: &Array3<f64>,
        epsilon: &Array3<f64>,
        mu: &Array3<f64>,
        sigma: &Array3<f64>,
        dt: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> KwaversResult<(
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
    )> {
        // Convert inputs to tensors
        let ex_tensor = self.array_to_tensor(ex);
        let ey_tensor = self.array_to_tensor(ey);
        let ez_tensor = self.array_to_tensor(ez);
        let hx_tensor = self.array_to_tensor(hx);
        let hy_tensor = self.array_to_tensor(hy);
        let hz_tensor = self.array_to_tensor(hz);
        let eps_tensor = self.array_to_tensor(epsilon);
        let mu_tensor = self.array_to_tensor(mu);
        let sigma_tensor = self.array_to_tensor(sigma);

        // Update H field from E field
        let (hx_new, hy_new, hz_new) = self.update_magnetic_field(
            &hx_tensor, &hy_tensor, &hz_tensor, &ex_tensor, &ey_tensor, &ez_tensor, &mu_tensor,
            dt as f32, dx as f32, dy as f32, dz as f32,
        );

        // Update E field from H field
        let (ex_new, ey_new, ez_new) = self.update_electric_field(
            &ex_tensor,
            &ey_tensor,
            &ez_tensor,
            &hx_new,
            &hy_new,
            &hz_new,
            &eps_tensor,
            &sigma_tensor,
            dt as f32,
            dx as f32,
            dy as f32,
            dz as f32,
        );

        // Convert back to arrays
        Ok((
            self.tensor_to_array(ex_new),
            self.tensor_to_array(ey_new),
            self.tensor_to_array(ez_new),
            self.tensor_to_array(hx_new),
            self.tensor_to_array(hy_new),
            self.tensor_to_array(hz_new),
        ))
    }

    /// Update magnetic field (H) from electric field (E)
    fn update_magnetic_field(
        &self,
        hx: &Tensor<B, 3>,
        hy: &Tensor<B, 3>,
        hz: &Tensor<B, 3>,
        ex: &Tensor<B, 3>,
        ey: &Tensor<B, 3>,
        ez: &Tensor<B, 3>,
        mu: &Tensor<B, 3>,
        dt: f32,
        dx: f32,
        dy: f32,
        dz: f32,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        // Maxwell's equations: ∂H/∂t = -∇×E / μ
        // Using Yee's algorithm

        // ∂Ez/∂y - ∂Ey/∂z
        let dex_dy = self.compute_gradient_y(ez, dy);
        let dey_dz = self.compute_gradient_z(ey, dz);
        let curl_x = dex_dy - dey_dz;

        // ∂Ex/∂z - ∂Ez/∂x
        let dez_dx = self.compute_gradient_x(ex, dx);
        let dex_dz = self.compute_gradient_z(ez, dz);
        let curl_y = dex_dz - dez_dx;

        // ∂Ey/∂x - ∂Ex/∂y
        let dey_dx = self.compute_gradient_x(ey, dx);
        let dex_dy = self.compute_gradient_y(ex, dy);
        let curl_z = dey_dx - dex_dy;

        // Update H: H^{n+1} = H^n - (dt/μ) * ∇×E
        let dt_mu = mu.clone().recip().mul_scalar(dt);
        let hx_new = hx.clone() - dt_mu.clone() * curl_x;
        let hy_new = hy.clone() - dt_mu.clone() * curl_y;
        let hz_new = hz.clone() - dt_mu * curl_z;

        (hx_new, hy_new, hz_new)
    }

    /// Update electric field (E) from magnetic field (H)
    fn update_electric_field(
        &self,
        ex: &Tensor<B, 3>,
        ey: &Tensor<B, 3>,
        ez: &Tensor<B, 3>,
        hx: &Tensor<B, 3>,
        hy: &Tensor<B, 3>,
        hz: &Tensor<B, 3>,
        eps: &Tensor<B, 3>,
        sigma: &Tensor<B, 3>,
        dt: f32,
        dx: f32,
        dy: f32,
        dz: f32,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        // Maxwell's equations: ∂E/∂t = (∇×H - σE) / ε
        // Using Yee's algorithm

        // ∂Hz/∂y - ∂Hy/∂z
        let dhz_dy = self.compute_gradient_y(hz, dy);
        let dhy_dz = self.compute_gradient_z(hy, dz);
        let curl_x = dhz_dy - dhy_dz;

        // ∂Hx/∂z - ∂Hz/∂x
        let dhx_dz = self.compute_gradient_z(hx, dz);
        let dhz_dx = self.compute_gradient_x(hz, dx);
        let curl_y = dhx_dz - dhz_dx;

        // ∂Hy/∂x - ∂Hx/∂y
        let dhy_dx = self.compute_gradient_x(hy, dx);
        let dhx_dy = self.compute_gradient_y(hx, dy);
        let curl_z = dhy_dx - dhx_dy;

        // Update E: E^{n+1} = E^n + (dt/ε) * (∇×H - σE)
        let dt_eps = eps.clone().recip().mul_scalar(dt);
        let conductivity_term = sigma.clone().mul_scalar(dt);

        let ex_new =
            ex.clone() + dt_eps.clone() * (curl_x - conductivity_term.clone() * ex.clone());
        let ey_new =
            ey.clone() + dt_eps.clone() * (curl_y - conductivity_term.clone() * ey.clone());
        let ez_new = ez.clone() + dt_eps * (curl_z - conductivity_term * ez.clone());

        (ex_new, ey_new, ez_new)
    }

    /// Compute PDE residual for physics-informed learning
    pub fn compute_pde_residual(
        &self,
        u: &Tensor<B, 4>, // [batch, x, y, t]
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 3> {
        // [batch, x, y]
        match physics_params.equation_type {
            EquationType::Wave => self.compute_wave_equation_residual(u, physics_params),
            EquationType::Heat => self.compute_heat_equation_residual(u, physics_params),
            EquationType::Diffusion => self.compute_diffusion_residual(u, physics_params),
            EquationType::NavierStokes => self.compute_navier_stokes_residual(u, physics_params),
        }
    }

    /// Compute wave equation residual: ∂²u/∂t² - c²∇²u
    fn compute_wave_equation_residual(
        &self,
        u: &Tensor<B, 4>,
        params: &PhysicsParameters,
    ) -> Tensor<B, 3> {
        let c = params.wave_speed.unwrap_or(343.0) as f32;

        // Compute second derivatives using finite differences
        // u shape: [batch, nx, ny, nt]
        let shape = u.shape();

        // ∂²u/∂t² using central differences
        let u_t_plus = u
            .clone()
            .slice([0..shape[0], 0..shape[1], 0..shape[2], 2..shape[3]]);
        let u_t_minus = u
            .clone()
            .slice([0..shape[0], 0..shape[1], 0..shape[2], 0..shape[3] - 2]);
        let u_center = u
            .clone()
            .slice([0..shape[0], 0..shape[1], 0..shape[2], 1..shape[3] - 1]);

        let d2u_dt2 = u_t_plus
            .sub(u_center.clone().mul_scalar(2.0))
            .add(u_t_minus)
            .div_scalar((params.dt as f32) * (params.dt as f32));

        // ∇²u = ∂²u/∂x² + ∂²u/∂y²
        let laplacian = self.compute_laplacian_2d(&u_center, params.dx, params.dy);

        // Residual: ∂²u/∂t² - c²∇²u
        let residual = d2u_dt2 - laplacian.mul_scalar(c * c);

        // Return 3D tensor by squeezing batch dimension if it's 1,
        // or just taking the first simulation in the batch for now
        // to match the Tensor<B, 3> return type.
        // In a production multi-simulation setup, we would return Tensor<B, 4>.
        let r_shape = residual.shape();
        residual
            .slice([0..1, 0..r_shape[1], 0..r_shape[2], 0..r_shape[3]])
            .squeeze()
    }

    /// Compute Laplacian in 2D
    fn compute_laplacian_2d(&self, u: &Tensor<B, 4>, dx: f64, dy: f64) -> Tensor<B, 4> {
        // ∂²u/∂x²
        let d2u_dx2 = self.compute_second_derivative_x(u, dx as f32);

        // ∂²u/∂y²
        let d2u_dy2 = self.compute_second_derivative_y(u, dy as f32);

        d2u_dx2 + d2u_dy2
    }

    /// Compute second derivative in x-direction
    fn compute_second_derivative_x(&self, field: &Tensor<B, 4>, dx: f32) -> Tensor<B, 4> {
        // f''(x) ≈ (f[i+1] - 2f[i] + f[i-1]) / dx²
        let shifted_right = field.clone().roll(&[1], &[1]);
        let shifted_left = field.clone().roll(&[-1], &[1]);

        shifted_right
            .add(shifted_left)
            .sub(field.clone().mul_scalar(2.0))
            .div_scalar(dx * dx)
    }

    /// Compute second derivative in y-direction
    fn compute_second_derivative_y(&self, field: &Tensor<B, 4>, dy: f32) -> Tensor<B, 4> {
        // f''(y) ≈ (f[j+1] - 2f[j] + f[j-1]) / dy²
        let shifted_up = field.clone().roll(&[1], &[2]);
        let shifted_down = field.clone().roll(&[-1], &[2]);

        shifted_up
            .add(shifted_down)
            .sub(field.clone().mul_scalar(2.0))
            .div_scalar(dy * dy)
    }

    /// Stub implementations for other PDEs (can be implemented as needed)
    fn compute_heat_equation_residual(
        &self,
        _u: &Tensor<B, 4>,
        _params: &PhysicsParameters,
    ) -> Tensor<B, 3> {
        Tensor::zeros([1, 1, 1], &self.device)
    }

    fn compute_diffusion_residual(
        &self,
        _u: &Tensor<B, 4>,
        _params: &PhysicsParameters,
    ) -> Tensor<B, 3> {
        Tensor::zeros([1, 1, 1], &self.device)
    }

    fn compute_navier_stokes_residual(
        &self,
        _u: &Tensor<B, 4>,
        _params: &PhysicsParameters,
    ) -> Tensor<B, 3> {
        Tensor::zeros([1, 1, 1], &self.device)
    }
}

/// Physics parameters for PDE computations
#[derive(Debug, Clone)]
pub struct PhysicsParameters {
    pub equation_type: EquationType,
    pub wave_speed: Option<f64>,
    pub diffusion_coefficient: Option<f64>,
    pub thermal_conductivity: Option<f64>,
    pub viscosity: Option<f64>,
    pub dt: f64,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

/// Types of PDE equations supported
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EquationType {
    Wave,
    Heat,
    Diffusion,
    NavierStokes,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enable_gpu: true,
            backend: "wgpu".to_string(),
            memory_strategy: MemoryStrategy::Dynamic,
            precision: Precision::F32,
        }
    }
}

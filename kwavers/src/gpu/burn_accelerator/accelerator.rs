use super::types::{EquationType, GpuConfig, PhysicsParameters};
use crate::core::error::{KwaversError, KwaversResult};
use crate::solver::forward::fdtd::FdtdGpuAccelerator;
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
        let p_tensor = self.array_to_tensor(pressure);
        let vx_tensor = self.array_to_tensor(velocity_x);
        let vy_tensor = self.array_to_tensor(velocity_y);
        let vz_tensor = self.array_to_tensor(velocity_z);
        let rho_tensor = self.array_to_tensor(density);
        let c_tensor = self.array_to_tensor(sound_speed);

        let divergence = self.compute_divergence(&vx_tensor, &vy_tensor, &vz_tensor, dx, dy, dz);

        let c_squared = c_tensor.clone().powf_scalar(2.0);
        let rho_c_squared = rho_tensor * c_squared;
        let pressure_update = (divergence * rho_c_squared).mul_scalar(dt as f32);
        let new_pressure = p_tensor - pressure_update;

        Ok(self.tensor_to_array(new_pressure))
    }

    fn compute_divergence(
        &self,
        vx: &Tensor<B, 3>,
        vy: &Tensor<B, 3>,
        vz: &Tensor<B, 3>,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> Tensor<B, 3> {
        let dvx_dx = self.compute_gradient_x(vx, dx as f32);
        let dvy_dy = self.compute_gradient_y(vy, dy as f32);
        let dvz_dz = self.compute_gradient_z(vz, dz as f32);
        dvx_dx + dvy_dy + dvz_dz
    }

    fn compute_gradient_x(&self, field: &Tensor<B, 3>, dx: f32) -> Tensor<B, 3> {
        let shifted_right = field.clone().roll(&[1], &[0]);
        let shifted_left = field.clone().roll(&[-1], &[0]);
        (shifted_right - shifted_left).div_scalar(2.0 * dx)
    }

    fn compute_gradient_y(&self, field: &Tensor<B, 3>, dy: f32) -> Tensor<B, 3> {
        let shifted_up = field.clone().roll(&[1], &[1]);
        let shifted_down = field.clone().roll(&[-1], &[1]);
        (shifted_up - shifted_down).div_scalar(2.0 * dy)
    }

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
        let ex_tensor = self.array_to_tensor(ex);
        let ey_tensor = self.array_to_tensor(ey);
        let ez_tensor = self.array_to_tensor(ez);
        let hx_tensor = self.array_to_tensor(hx);
        let hy_tensor = self.array_to_tensor(hy);
        let hz_tensor = self.array_to_tensor(hz);
        let eps_tensor = self.array_to_tensor(epsilon);
        let mu_tensor = self.array_to_tensor(mu);
        let sigma_tensor = self.array_to_tensor(sigma);

        let (hx_new, hy_new, hz_new) = self.update_magnetic_field(
            &hx_tensor, &hy_tensor, &hz_tensor, &ex_tensor, &ey_tensor, &ez_tensor, &mu_tensor,
            dt as f32, dx as f32, dy as f32, dz as f32,
        );

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

        Ok((
            self.tensor_to_array(ex_new),
            self.tensor_to_array(ey_new),
            self.tensor_to_array(ez_new),
            self.tensor_to_array(hx_new),
            self.tensor_to_array(hy_new),
            self.tensor_to_array(hz_new),
        ))
    }

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
        let dex_dy = self.compute_gradient_y(ez, dy);
        let dey_dz = self.compute_gradient_z(ey, dz);
        let curl_x = dex_dy - dey_dz;

        let dez_dx = self.compute_gradient_x(ex, dx);
        let dex_dz = self.compute_gradient_z(ez, dz);
        let curl_y = dex_dz - dez_dx;

        let dey_dx = self.compute_gradient_x(ey, dx);
        let dex_dy = self.compute_gradient_y(ex, dy);
        let curl_z = dey_dx - dex_dy;

        let dt_mu = mu.clone().recip().mul_scalar(dt);
        let hx_new = hx.clone() - dt_mu.clone() * curl_x;
        let hy_new = hy.clone() - dt_mu.clone() * curl_y;
        let hz_new = hz.clone() - dt_mu * curl_z;

        (hx_new, hy_new, hz_new)
    }

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
        let dhz_dy = self.compute_gradient_y(hz, dy);
        let dhy_dz = self.compute_gradient_z(hy, dz);
        let curl_x = dhz_dy - dhy_dz;

        let dhx_dz = self.compute_gradient_z(hx, dz);
        let dhz_dx = self.compute_gradient_x(hz, dx);
        let curl_y = dhx_dz - dhz_dx;

        let dhy_dx = self.compute_gradient_x(hy, dx);
        let dhx_dy = self.compute_gradient_y(hx, dy);
        let curl_z = dhy_dx - dhx_dy;

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
        u: &Tensor<B, 4>,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 3> {
        match physics_params.equation_type {
            EquationType::Wave => self.compute_wave_equation_residual(u, physics_params),
            EquationType::Heat => self.compute_heat_equation_residual(u, physics_params),
            EquationType::Diffusion => self.compute_diffusion_residual(u, physics_params),
            EquationType::NavierStokes => self.compute_navier_stokes_residual(u, physics_params),
        }
    }

    fn compute_wave_equation_residual(
        &self,
        u: &Tensor<B, 4>,
        params: &PhysicsParameters,
    ) -> Tensor<B, 3> {
        let c = params.wave_speed.unwrap_or(1480.0) as f32;

        let shape = u.shape();

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

        let laplacian = self.compute_laplacian_2d(&u_center, params.dx, params.dy);

        let residual = d2u_dt2 - laplacian.mul_scalar(c * c);

        let r_shape = residual.shape();
        residual
            .slice([0..1, 0..r_shape[1], 0..r_shape[2], 0..r_shape[3]])
            .squeeze()
    }

    fn compute_laplacian_2d(&self, u: &Tensor<B, 4>, dx: f64, dy: f64) -> Tensor<B, 4> {
        let d2u_dx2 = self.compute_second_derivative_x(u, dx as f32);
        let d2u_dy2 = self.compute_second_derivative_y(u, dy as f32);
        d2u_dx2 + d2u_dy2
    }

    fn compute_second_derivative_x(&self, field: &Tensor<B, 4>, dx: f32) -> Tensor<B, 4> {
        let shifted_right = field.clone().roll(&[1], &[1]);
        let shifted_left = field.clone().roll(&[-1], &[1]);
        shifted_right
            .add(shifted_left)
            .sub(field.clone().mul_scalar(2.0))
            .div_scalar(dx * dx)
    }

    fn compute_second_derivative_y(&self, field: &Tensor<B, 4>, dy: f32) -> Tensor<B, 4> {
        let shifted_up = field.clone().roll(&[1], &[2]);
        let shifted_down = field.clone().roll(&[-1], &[2]);
        shifted_up
            .add(shifted_down)
            .sub(field.clone().mul_scalar(2.0))
            .div_scalar(dy * dy)
    }

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

impl<B: Backend> FdtdGpuAccelerator for BurnGpuAccelerator<B> {
    fn propagate_acoustic_wave(
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
        BurnGpuAccelerator::propagate_acoustic_wave(
            self,
            pressure,
            velocity_x,
            velocity_y,
            velocity_z,
            density,
            sound_speed,
            dt,
            dx,
            dy,
            dz,
        )
    }
}

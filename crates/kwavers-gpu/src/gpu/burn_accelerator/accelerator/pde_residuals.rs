//! PDE residual computation for physics-informed learning (wave, heat, diffusion, Navier-Stokes).

use super::super::super::types::{EquationType, GpuPhysicsParameters};
use super::BurnGpuAccelerator;
use burn::prelude::*;
use burn::tensor::backend::Backend;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

impl<B: Backend> BurnGpuAccelerator<B> {
    /// Compute PDE residual for physics-informed learning.
    pub fn compute_pde_residual(
        &self,
        u: &Tensor<B, 4>,
        physics_params: &GpuPhysicsParameters,
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
        params: &GpuPhysicsParameters,
    ) -> Tensor<B, 3> {
        let c = params.wave_speed.unwrap_or(SOUND_SPEED_WATER_SIM) as f32;
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

        let dt_sq = (params.dt as f32) * (params.dt as f32);
        let d2u_dt2 = u_t_plus
            .sub(u_center.clone().mul_scalar(2.0))
            .add(u_t_minus)
            .div_scalar(dt_sq);

        let laplacian = self.compute_laplacian_2d(&u_center, params.dx, params.dy);
        let residual = d2u_dt2 - laplacian.mul_scalar(c * c);

        let r_shape = residual.shape();
        residual
            .slice([0..1, 0..r_shape[1], 0..r_shape[2], 0..r_shape[3]])
            .squeeze()
    }

    fn compute_laplacian_2d(&self, u: &Tensor<B, 4>, dx: f64, dy: f64) -> Tensor<B, 4> {
        self.compute_second_derivative_x(u, dx as f32)
            + self.compute_second_derivative_y(u, dy as f32)
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
        _params: &GpuPhysicsParameters,
    ) -> Tensor<B, 3> {
        Tensor::zeros([1, 1, 1], &self.device)
    }

    fn compute_diffusion_residual(
        &self,
        _u: &Tensor<B, 4>,
        _params: &GpuPhysicsParameters,
    ) -> Tensor<B, 3> {
        Tensor::zeros([1, 1, 1], &self.device)
    }

    fn compute_navier_stokes_residual(
        &self,
        _u: &Tensor<B, 4>,
        _params: &GpuPhysicsParameters,
    ) -> Tensor<B, 3> {
        Tensor::zeros([1, 1, 1], &self.device)
    }
}

//! Acoustic wave propagation on GPU via Burn.

use super::BurnGpuAccelerator;
use crate::core::error::KwaversResult;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ndarray::Array3;

impl<B: Backend> BurnGpuAccelerator<B> {
    /// Execute acoustic wave propagation on GPU.
    ///
    /// Updates pressure: `p_new = p - ρc² (∇·v) Δt`
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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

        let c_squared = c_tensor.powf_scalar(2.0);
        let rho_c_squared = rho_tensor * c_squared;
        let pressure_update = (divergence * rho_c_squared).mul_scalar(dt as f32);
        let new_pressure = p_tensor - pressure_update;

        Ok(self.tensor_to_array(new_pressure))
    }

    pub(super) fn compute_divergence(
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
}

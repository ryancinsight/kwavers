//! Electromagnetic wave propagation on GPU via Burn (Yee-scheme curl updates).

use super::BurnGpuAccelerator;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use kwavers_core::error::KwaversResult;
use ndarray::Array3;

impl<B: Backend> BurnGpuAccelerator<B> {
    /// Execute electromagnetic wave propagation on GPU (one Yee leap-frog step).
    ///
    /// Returns `(Ex, Ey, Ez, Hx, Hy, Hz)` after one dt.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
        let ex_t = self.array_to_tensor(ex);
        let ey_t = self.array_to_tensor(ey);
        let ez_t = self.array_to_tensor(ez);
        let hx_t = self.array_to_tensor(hx);
        let hy_t = self.array_to_tensor(hy);
        let hz_t = self.array_to_tensor(hz);
        let eps_t = self.array_to_tensor(epsilon);
        let mu_t = self.array_to_tensor(mu);
        let sig_t = self.array_to_tensor(sigma);

        let (hx_new, hy_new, hz_new) = self.update_magnetic_field(
            &hx_t, &hy_t, &hz_t, &ex_t, &ey_t, &ez_t, &mu_t, dt as f32, dx as f32, dy as f32,
            dz as f32,
        );

        let (ex_new, ey_new, ez_new) = self.update_electric_field(
            &ex_t, &ey_t, &ez_t, &hx_new, &hy_new, &hz_new, &eps_t, &sig_t, dt as f32, dx as f32,
            dy as f32, dz as f32,
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
        let curl_x = self.compute_gradient_y(ez, dy) - self.compute_gradient_z(ey, dz);
        let curl_y = self.compute_gradient_z(ez, dz) - self.compute_gradient_x(ex, dx);
        let curl_z = self.compute_gradient_x(ey, dx) - self.compute_gradient_y(ex, dy);

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
        let curl_x = self.compute_gradient_y(hz, dy) - self.compute_gradient_z(hy, dz);
        let curl_y = self.compute_gradient_z(hx, dz) - self.compute_gradient_x(hz, dx);
        let curl_z = self.compute_gradient_x(hy, dx) - self.compute_gradient_y(hx, dy);

        let dt_eps = eps.clone().recip().mul_scalar(dt);
        let cond_dt = sigma.clone().mul_scalar(dt);

        let ex_new = ex.clone() + dt_eps.clone() * (curl_x - cond_dt.clone() * ex.clone());
        let ey_new = ey.clone() + dt_eps.clone() * (curl_y - cond_dt.clone() * ey.clone());
        let ez_new = ez.clone() + dt_eps * (curl_z - cond_dt * ez.clone());

        (ex_new, ey_new, ez_new)
    }
}

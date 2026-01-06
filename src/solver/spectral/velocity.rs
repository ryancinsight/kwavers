//! Velocity field updates for spectral solver
//!
//! This module implements the momentum conservation equation:
//! du/dt = -1/rho0 * grad(p)

use crate::error::KwaversResult;
use super::solver::SpectralSolver;
use ndarray::Zip;
use crate::fft::Complex64;

impl SpectralSolver {
    /// Update velocity fields based on pressure gradients (Momentum Conservation)
    pub(super) fn update_velocity(&mut self, dt: f64) -> KwaversResult<()> {
        let i_img = Complex64::new(0.0, 1.0);

        // Transform pressure to k-space
        self.fft.forward_into(&self.p, &mut self.p_k);

        // Compute pressure gradients in k-space
        // grad_p_k = i * k * kappa * p_k

        // dx
        Zip::from(&mut self.grad_x_k)
            .and(&self.p_k)
            .and(&self.k_vec.0)
            .and(&self.kappa)
            .for_each(|grad, &p, &k, &kap| {
                *grad = i_img * k * kap * p;
            });

        // dy
        Zip::from(&mut self.grad_y_k)
            .and(&self.p_k)
            .and(&self.k_vec.1)
            .and(&self.kappa)
            .for_each(|grad, &p, &k, &kap| {
                *grad = i_img * k * kap * p;
            });

        // dz
        Zip::from(&mut self.grad_z_k)
            .and(&self.p_k)
            .and(&self.k_vec.2)
            .and(&self.kappa)
            .for_each(|grad, &p, &k, &kap| {
                *grad = i_img * k * kap * p;
            });

        // Transform gradients back to physical space and update velocity

        // X-direction
        self.fft
            .inverse_into(&self.grad_x_k, &mut self.dpx, &mut self.ux_k);
        Zip::from(&mut self.ux)
            .and(&self.dpx)
            .and(&self.rho0)
            .for_each(|u, &dp, &rho| {
                *u -= (dt / rho) * dp;
            });

        // Y-direction
        self.fft
            .inverse_into(&self.grad_y_k, &mut self.dpy, &mut self.uy_k);
        Zip::from(&mut self.uy)
            .and(&self.dpy)
            .and(&self.rho0)
            .for_each(|u, &dp, &rho| {
                *u -= (dt / rho) * dp;
            });

        // Z-direction
        self.fft
            .inverse_into(&self.grad_z_k, &mut self.dpz, &mut self.uz_k);
        Zip::from(&mut self.uz)
            .and(&self.dpz)
            .and(&self.rho0)
            .for_each(|u, &dp, &rho| {
                *u -= (dt / rho) * dp;
            });

        // Inject force sources (velocity sources)
        self.source_handler.inject_force_source(
            self.time_step_index,
            &mut self.ux,
            &mut self.uy,
            &mut self.uz,
        );

        // Apply PML to velocities
        self.apply_pml_to_velocity()?;

        Ok(())
    }

    /// Apply PML damping to velocity
    pub(super) fn apply_pml_to_velocity(&mut self) -> KwaversResult<()> {
        if let Some(ref mut boundary) = self.boundary {
            boundary.apply_acoustic(self.ux.view_mut(), &self.grid, self.time_step_index)?;
            boundary.apply_acoustic(self.uy.view_mut(), &self.grid, self.time_step_index)?;
            boundary.apply_acoustic(self.uz.view_mut(), &self.grid, self.time_step_index)?;
        }
        Ok(())
    }
}

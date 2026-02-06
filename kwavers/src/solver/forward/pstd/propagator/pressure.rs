//! Pressure and Density field updates for spectral solver.
//!
//! Implements:
//! 1. Equation of State (Pressure from Density)
//! 2. Mass Conservation (Density from Velocity Divergence)
//! 3. Power Law Absorption (Fractional Laplacian method)
//! 4. PML application to density

use crate::core::error::KwaversResult;
use crate::math::fft::Complex64;
use crate::solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use ndarray::Zip;

impl PSTDSolver {
    /// Update pressure field from density perturbation (Equation of State)
    pub(crate) fn update_pressure(&mut self) {
        // Combine split density components
        Zip::from(&mut self.div_u)
            .and(&self.rhox)
            .and(&self.rhoy)
            .and(&self.rhoz)
            .for_each(|rho_sum, &rx, &ry, &rz| {
                *rho_sum = rx + ry + rz;
            });

        if self.config.nonlinearity {
            Zip::from(&mut self.fields.p)
                .and(&self.div_u)
                .and(&self.materials.c0)
                .and(&self.bon)
                .and(&self.materials.rho0)
                .for_each(|p, &rho_sum, &c, &bon, &rho0| {
                    let linear = rho_sum;
                    // Taylor expansion for nonlinearity
                    // p = c0^2 * (rho + B/2A * rho^2 / rho0)
                    let nonlinear = (bon / (2.0 * rho0)) * rho_sum * rho_sum;
                    *p = c * c * (linear + nonlinear);
                });
        } else {
            Zip::from(&mut self.fields.p)
                .and(&self.div_u)
                .and(&self.materials.c0)
                .for_each(|p, &rho_sum, &c| {
                    *p = c * c * rho_sum;
                });
        }
    }

    /// Update density field based on velocity divergence (Mass Conservation)
    pub(crate) fn update_density(&mut self, dt: f64) -> KwaversResult<()> {
        let i_img = Complex64::new(0.0, 1.0);

        // Compute dux/dx, duy/dy, duz/dz in k-space
        let (ref kx, ref ky, ref kz) = self.k_vec;

        self.fft.forward_into(&self.fields.ux, &mut self.ux_k);
        Zip::from(&mut self.grad_x_k)
            .and(&self.ux_k)
            .and(kx)
            .and(&self.kappa)
            .for_each(|grad, &ux, &kx_val, &kap| {
                *grad = i_img * kap * Complex64::new(kx_val, 0.0) * ux;
            });
        self.fft
            .inverse_into(&self.grad_x_k, &mut self.dpx, &mut self.ux_k);

        self.fft.forward_into(&self.fields.uy, &mut self.uy_k);
        Zip::from(&mut self.grad_y_k)
            .and(&self.uy_k)
            .and(ky)
            .and(&self.kappa)
            .for_each(|grad, &uy, &ky_val, &kap| {
                *grad = i_img * kap * Complex64::new(ky_val, 0.0) * uy;
            });
        self.fft
            .inverse_into(&self.grad_y_k, &mut self.dpy, &mut self.uy_k);

        self.fft.forward_into(&self.fields.uz, &mut self.uz_k);
        Zip::from(&mut self.grad_z_k)
            .and(&self.uz_k)
            .and(kz)
            .and(&self.kappa)
            .for_each(|grad, &uz, &kz_val, &kap| {
                *grad = i_img * kap * Complex64::new(kz_val, 0.0) * uz;
            });
        self.fft
            .inverse_into(&self.grad_z_k, &mut self.dpz, &mut self.uz_k);

        // Update split density components
        Zip::from(&mut self.rhox)
            .and(&self.dpx)
            .and(&self.materials.rho0)
            .and(&self.fields.ux)
            .and(&self.grad_rho0_x)
            .for_each(|rho, &du, &rho0, &ux, &grx| {
                *rho -= dt * (rho0 * du + ux * grx);
            });

        Zip::from(&mut self.rhoy)
            .and(&self.dpy)
            .and(&self.materials.rho0)
            .and(&self.fields.uy)
            .and(&self.grad_rho0_y)
            .for_each(|rho, &du, &rho0, &uy, &gry| {
                *rho -= dt * (rho0 * du + uy * gry);
            });

        Zip::from(&mut self.rhoz)
            .and(&self.dpz)
            .and(&self.materials.rho0)
            .and(&self.fields.uz)
            .and(&self.grad_rho0_z)
            .for_each(|rho, &du, &rho0, &uz, &grz| {
                *rho -= dt * (rho0 * du + uz * grz);
            });

        // Update total density rho = rhox + rhoy + rhoz
        // Also update pressure field p = c^2 * rho for consistency
        Zip::from(&mut self.div_u)
            .and(&self.rhox)
            .and(&self.rhoy)
            .and(&self.rhoz)
            .for_each(|rho, &rx, &ry, &rz| {
                *rho = rx + ry + rz;
            });
        Zip::from(&self.div_u)
            .and(&self.materials.c0)
            .and(&mut self.fields.p)
            .for_each(|&rho, &c0, p| {
                *p = c0 * c0 * rho;
            });

        // NOTE: Mass sources are injected at the start of step_forward() (step 1)
        // We do NOT inject them again here to avoid double-counting and amplification.
        // This was the root cause of the 6.23× PSTD amplification bug.

        // Apply absorption and PML
        self.apply_absorption(dt)?;
        self.apply_pml_to_density()?;

        Ok(())
    }

    /// Apply PML to density field
    fn apply_pml_to_density(&mut self) -> KwaversResult<()> {
        if let Some(boundary) = self.boundary.as_deref_mut() {
            boundary.apply_acoustic(
                self.rhox.view_mut(),
                self.grid.as_ref(),
                self.time_step_index,
            )?;
            boundary.apply_acoustic(
                self.rhoy.view_mut(),
                self.grid.as_ref(),
                self.time_step_index,
            )?;
            boundary.apply_acoustic(
                self.rhoz.view_mut(),
                self.grid.as_ref(),
                self.time_step_index,
            )?;
        }
        Ok(())
    }
}

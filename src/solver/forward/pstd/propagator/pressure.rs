//! Pressure and Density field updates for spectral solver.
//!
//! Implements:
//! 1. Equation of State (Pressure from Density)
//! 2. Mass Conservation (Density from Velocity Divergence)
//! 3. Power Law Absorption (Fractional Laplacian method)
//! 4. PML application to density

use crate::core::error::KwaversResult;
use crate::math::fft::Complex64;
use crate::solver::forward::pstd::solver::PSTDSolver;
use ndarray::Zip;

impl PSTDSolver {
    /// Update pressure field from density perturbation (Equation of State)
    pub(crate) fn update_pressure(&mut self) {
        if self.config.nonlinearity {
            Zip::from(&mut self.fields.p)
                .and(&self.rho)
                .and(&self.materials.c0)
                .and(&self.bon)
                .and(&self.materials.rho0)
                .for_each(|p, &rho, &c, &bon, &rho0| {
                    let linear = rho;
                    // Taylor expansion for nonlinearity
                    // p = c0^2 * (rho + B/2A * rho^2 / rho0)
                    let nonlinear = (bon / (2.0 * rho0)) * rho * rho;
                    *p = c * c * (linear + nonlinear);
                });
        } else {
            Zip::from(&mut self.fields.p)
                .and(&self.rho)
                .and(&self.materials.c0)
                .for_each(|p, &rho, &c| {
                    *p = c * c * rho;
                });
        }
    }

    /// Update density field based on velocity divergence (Mass Conservation)
    pub(crate) fn update_density(&mut self, dt: f64) -> KwaversResult<()> {
        let i_img = Complex64::new(0.0, 1.0);

        // Compute velocity divergence in k-space
        self.fft.forward_into(&self.fields.ux, &mut self.ux_k);
        self.fft.forward_into(&self.fields.uy, &mut self.uy_k);
        self.fft.forward_into(&self.fields.uz, &mut self.uz_k);

        // div_u_k = i*kx*kappa*ux_k + i*ky*kappa*uy_k + i*kz*kappa*uz_k
        // Use grad_x_k as scratch for div_u_k
        let (ref kx, ref ky, ref kz) = self.k_vec;
        Zip::from(&mut self.grad_x_k)
            .and(&self.ux_k)
            .and(&self.uy_k)
            .and(kx)
            .and(ky)
            .for_each(|div, &ux, &uy, &kx_val, &ky_val| {
                *div = Complex64::new(kx_val, 0.0) * ux + Complex64::new(ky_val, 0.0) * uy;
            });

        Zip::from(&mut self.grad_x_k)
            .and(&self.uz_k)
            .and(kz)
            .and(&self.kappa)
            .for_each(|div, &uz, &kz_val, &kap| {
                *div = i_img * kap * (*div + Complex64::new(kz_val, 0.0) * uz);
            });

        // IFFT div_u_k -> div_u (physical)
        self.fft
            .inverse_into(&self.grad_x_k, &mut self.div_u, &mut self.ux_k);

        // Update density: rho -= dt * (rho0 * div_u + u.grad(rho0))
        Zip::from(&mut self.rho)
            .and(&self.div_u)
            .and(&self.materials.rho0)
            .and(&self.fields.ux)
            .and(&self.grad_rho0_x)
            .for_each(|rho, &du, &rho0, &ux, &grx| {
                *rho -= dt * (rho0 * du + ux * grx);
            });

        Zip::from(&mut self.rho)
            .and(&self.fields.uy)
            .and(&self.fields.uz)
            .and(&self.grad_rho0_y)
            .and(&self.grad_rho0_z)
            .for_each(|rho, &uy, &uz, &gry, &grz| {
                *rho -= dt * (uy * gry + uz * grz);
            });

        // Inject mass sources
        self.source_handler.inject_mass_source(
            self.time_step_index,
            &mut self.rho,
            &self.materials.c0,
        );

        // Apply absorption and PML
        self.apply_absorption(dt)?;
        self.apply_pml_to_density()?;

        Ok(())
    }

    /// Apply PML to density field
    fn apply_pml_to_density(&mut self) -> KwaversResult<()> {
        if let Some(boundary) = self.boundary.as_deref_mut() {
            boundary.apply_acoustic(
                self.rho.view_mut(),
                self.grid.as_ref(),
                self.time_step_index,
            )?;
        }
        Ok(())
    }
}

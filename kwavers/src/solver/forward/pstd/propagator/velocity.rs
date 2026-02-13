//! Velocity field updates for spectral solver
//!
//! This module implements the momentum conservation equation:
//! du/dt = -1/rho0 * grad(p)

use crate::core::error::KwaversResult;
use crate::math::fft::Complex64;
use crate::solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use ndarray::Zip;

impl PSTDSolver {
    /// Update velocity fields based on pressure gradients (Momentum Conservation)
    ///
    /// Uses staggered grid shift operators matching the C++ k-wave binary:
    ///   grad_x(p) = IFFT( ddx_k_shift_pos[x] * kappa[i,j,k] * FFT(p)[i,j,k] )
    pub(crate) fn update_velocity(&mut self, dt: f64) -> KwaversResult<()> {
        // Transform pressure to k-space
        self.fft.forward_into(&self.fields.p, &mut self.p_k);

        // Compute pressure gradients in k-space with staggered grid shifts
        // k-wave uses ddx_k_shift_pos for pressure→velocity (positive shift)
        let (nx, ny, nz) = self.p_k.dim();
        for i in 0..nx {
            let shift_x = self.ddx_k_shift_pos[i];
            for j in 0..ny {
                let shift_y = self.ddy_k_shift_pos[j];
                for k in 0..nz {
                    let shift_z = self.ddz_k_shift_pos[k];
                    let kap = Complex64::new(self.kappa[[i, j, k]], 0.0);
                    let p_val = self.p_k[[i, j, k]];
                    let e_kappa = kap * p_val;
                    self.grad_x_k[[i, j, k]] = shift_x * e_kappa;
                    self.grad_y_k[[i, j, k]] = shift_y * e_kappa;
                    self.grad_z_k[[i, j, k]] = shift_z * e_kappa;
                }
            }
        }

        // Transform gradients back to physical space and update velocity

        // X-direction
        self.fft
            .inverse_into(&self.grad_x_k, &mut self.dpx, &mut self.ux_k);
        Zip::from(&mut self.fields.ux)
            .and(&self.dpx)
            .and(&self.materials.rho0)
            .for_each(|u, &dp, &rho| {
                *u -= (dt / rho) * dp;
            });

        // Y-direction
        self.fft
            .inverse_into(&self.grad_y_k, &mut self.dpy, &mut self.uy_k);
        Zip::from(&mut self.fields.uy)
            .and(&self.dpy)
            .and(&self.materials.rho0)
            .for_each(|u, &dp, &rho| {
                *u -= (dt / rho) * dp;
            });

        // Z-direction
        self.fft
            .inverse_into(&self.grad_z_k, &mut self.dpz, &mut self.uz_k);
        Zip::from(&mut self.fields.uz)
            .and(&self.dpz)
            .and(&self.materials.rho0)
            .for_each(|u, &dp, &rho| {
                *u -= (dt / rho) * dp;
            });

        // Inject force sources (velocity sources)
        self.source_handler.inject_force_source(
            self.time_step_index,
            &mut self.fields.ux,
            &mut self.fields.uy,
            &mut self.fields.uz,
        );

        // Apply PML to velocities
        self.apply_pml_to_velocity()?;

        Ok(())
    }

    /// Apply PML damping to velocity
    pub(super) fn apply_pml_to_velocity(&mut self) -> KwaversResult<()> {
        if let Some(boundary) = self.boundary.as_deref_mut() {
            boundary.apply_acoustic(
                self.fields.ux.view_mut(),
                self.grid.as_ref(),
                self.time_step_index,
            )?;
            boundary.apply_acoustic(
                self.fields.uy.view_mut(),
                self.grid.as_ref(),
                self.time_step_index,
            )?;
            boundary.apply_acoustic(
                self.fields.uz.view_mut(),
                self.grid.as_ref(),
                self.time_step_index,
            )?;
        }
        Ok(())
    }
}

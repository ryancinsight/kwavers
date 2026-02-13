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
    ///
    /// In C++ k-wave, this is the final step: p = c² * (rhox + rhoy + rhoz)
    /// Absorption (if enabled) is also applied here, matching C++ computePressure().
    pub(crate) fn update_pressure(&mut self, dt: f64) -> KwaversResult<()> {
        // Apply absorption first (modifies density, matching C++ where absorption
        // is part of computePressure, not computeDensity)
        self.apply_absorption(dt)?;

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

        Ok(())
    }

    /// Update density field based on velocity divergence (Mass Conservation)
    ///
    /// Uses staggered grid shift operators matching the C++ k-wave binary:
    ///   dux/dx = IFFT( ddx_k_shift_neg[x] * kappa[i,j,k] * FFT(ux)[i,j,k] )
    pub(crate) fn update_density(&mut self, dt: f64) -> KwaversResult<()> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);

        // dux/dx with negative shift
        self.fft.forward_into(&self.fields.ux, &mut self.ux_k);
        for i in 0..nx {
            let shift = self.ddx_k_shift_neg[i];
            for j in 0..ny {
                for k in 0..nz {
                    let kap = Complex64::new(self.kappa[[i, j, k]], 0.0);
                    self.grad_x_k[[i, j, k]] = shift * kap * self.ux_k[[i, j, k]];
                }
            }
        }
        self.fft
            .inverse_into(&self.grad_x_k, &mut self.dpx, &mut self.ux_k);

        // duy/dy with negative shift
        self.fft.forward_into(&self.fields.uy, &mut self.uy_k);
        for i in 0..nx {
            for j in 0..ny {
                let shift = self.ddy_k_shift_neg[j];
                for k in 0..nz {
                    let kap = Complex64::new(self.kappa[[i, j, k]], 0.0);
                    self.grad_y_k[[i, j, k]] = shift * kap * self.uy_k[[i, j, k]];
                }
            }
        }
        self.fft
            .inverse_into(&self.grad_y_k, &mut self.dpy, &mut self.uy_k);

        // duz/dz with negative shift
        self.fft.forward_into(&self.fields.uz, &mut self.uz_k);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let shift = self.ddz_k_shift_neg[k];
                    let kap = Complex64::new(self.kappa[[i, j, k]], 0.0);
                    self.grad_z_k[[i, j, k]] = shift * kap * self.uz_k[[i, j, k]];
                }
            }
        }
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

        // Apply PML to density (matches C++ which integrates PML into density update)
        self.apply_pml_to_density()?;

        // NOTE: Pressure computation and absorption are handled separately in
        // update_pressure(). Source injection happens between density and pressure
        // updates (matching C++ k-wave binary time loop order).

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

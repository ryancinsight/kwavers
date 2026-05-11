use crate::core::error::KwaversResult;
use crate::solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use ndarray::{s, Zip};

impl PSTDSolver {
    /// Standard 3-D Cartesian density update.
    ///
    /// Uses staggered grid shift operators with kappa correction matching the C++ k-wave binary:
    ///   dux/dx = IFFT( ddx_k_shift_neg[x] * kappa[i,j,k] * FFT(ux)[i,j,k] )
    ///
    /// kappa IS applied here — Treeby & Cox (2010) Eq. 17 explicitly includes the k-space
    /// correction factor κ in the density update, same as in the velocity update (Eq. 16).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[inline]
    pub(crate) fn update_density_cartesian(&mut self, dt: f64) -> KwaversResult<()> {
        let has_y = self.grid.ny > 1;
        let has_z = self.grid.nz > 1;

        self.apply_pml_to_density()?; // pre: pml * rho_old

        let nz_c = self.ux_k.dim().2;

        // dux/dx with negative shift + kappa correction (Treeby & Cox 2010, Eq. 17).
        self.fft.forward_r2c_into(&self.fields.ux, &mut self.ux_k);
        {
            let ddx = self.ddx_k_shift_neg.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.ux_k.view())
                .and(self.kappa.slice(s![.., .., ..nz_c]))
                .par_for_each(|(i, _j, _k), gk, &u, &kap| {
                    *gk = (ddx[i] * u) * kap;
                });
        }
        self.fft
            .inverse_c2r_into(&self.grad_k, &mut self.dpx, &mut self.ux_k);
        self.div_ux.assign(&self.dpx);

        // duy/dy with negative shift + kappa (matches k-Wave Eq. 17).
        if has_y {
            self.fft.forward_r2c_into(&self.fields.uy, &mut self.uy_k);
            let ddy = self.ddy_k_shift_neg.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.uy_k.view())
                .and(self.kappa.slice(s![.., .., ..nz_c]))
                .par_for_each(|(_i, j, _k), gk, &u, &kap| {
                    *gk = (ddy[j] * u) * kap;
                });
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.dpy, &mut self.uy_k);
            self.div_uy.assign(&self.dpy);
        } else {
            self.div_uy.fill(0.0);
        }

        // duz/dz with negative shift + kappa (matches k-Wave Eq. 17).
        if has_z {
            self.fft.forward_r2c_into(&self.fields.uz, &mut self.uz_k);
            let ddz = self.ddz_k_shift_neg.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.uz_k.view())
                .and(self.kappa.slice(s![.., .., ..nz_c]))
                .par_for_each(|(_i, _j, k), gk, &u, &kap| {
                    *gk = (ddz[k] * u) * kap;
                });
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.dpz, &mut self.uz_k);
            self.div_uz.assign(&self.dpz);
        } else {
            self.div_uz.fill(0.0);
        }

        // ── Mass-conservation coefficient snapshot ──────────────────────────
        // Linear:    coef = rho0
        // Nonlinear: coef = rho0 + 2·(rhox + rhoy + rhoz)   [Westervelt]
        if self.config.nonlinearity {
            Zip::from(&mut self.div_u)
                .and(&self.materials.rho0)
                .and(&self.rhox)
                .and(&self.rhoy)
                .and(&self.rhoz)
                .par_for_each(|coef, &rho0, &rx, &ry, &rz| {
                    *coef = 2.0f64.mul_add(rx + ry + rz, rho0);
                });

            Zip::from(&mut self.rhox)
                .and(&self.dpx)
                .and(&self.div_u)
                .par_for_each(|rho, &du, &coef| {
                    *rho -= dt * coef * du;
                });

            if has_y {
                Zip::from(&mut self.rhoy)
                    .and(&self.dpy)
                    .and(&self.div_u)
                    .par_for_each(|rho, &du, &coef| {
                        *rho -= dt * coef * du;
                    });
            }

            if has_z {
                Zip::from(&mut self.rhoz)
                    .and(&self.dpz)
                    .and(&self.div_u)
                    .par_for_each(|rho, &du, &coef| {
                        *rho -= dt * coef * du;
                    });
            }
        } else {
            Zip::from(&mut self.rhox)
                .and(&self.dpx)
                .and(&self.materials.rho0)
                .par_for_each(|rho, &du, &rho0| {
                    *rho -= dt * rho0 * du;
                });

            if has_y {
                Zip::from(&mut self.rhoy)
                    .and(&self.dpy)
                    .and(&self.materials.rho0)
                    .par_for_each(|rho, &du, &rho0| {
                        *rho -= dt * rho0 * du;
                    });
            }

            if has_z {
                Zip::from(&mut self.rhoz)
                    .and(&self.dpz)
                    .and(&self.materials.rho0)
                    .par_for_each(|rho, &du, &rho0| {
                        *rho -= dt * rho0 * du;
                    });
            }
        }

        self.apply_pml_to_density()?; // post: pml * (pml*rho_old - dt*rho0*div_u)
        Ok(())
    }
}

use crate::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::Fft3dInOutExt;
use ndarray::Zip;

// Implementation note on divergence caching:
// `update_density_cartesian` writes ∂u_α/∂α directly into `div_ux`/`div_uy`/`div_uz`
// (eliminating the intermediate `dpx/dpy/dpz` copy that was present pre-Opt-3).
// `apply_absorption_to_pressure` fuses div_u* into `dpx` as a single Zip (Opt-7 + Opt-12),
// then IFFTs to L1 in `dpx` and L2 in `dpy`.  The absorption read is unaffected because it
// always reads from `div_u*`.  Saves 3 × N-element memcpy per step.

impl PSTDSolver {
    /// Standard 3-D Cartesian density update.
    ///
    /// Uses staggered grid shift operators with kappa correction matching the C++ k-wave binary:
    ///   dux/dx = IFFT( ddx_k_shift_neg[x] * kappa[i,j,k] * FFT(ux)[i,j,k] )
    ///
    /// kappa IS applied here — Treeby & Cox (2010) Eq. 17 explicitly includes the k-space
    /// correction factor κ in the density update, same as in the velocity update (Eq. 16).
    ///
    /// ## Optimisations applied
    ///
    /// **IFFT → div_u directly**: the IFFT result is written into `div_ux`/`div_uy`/`div_uz`
    /// rather than first into `dpx`/`dpy`/`dpz` and then copied via `.assign()`.  The
    /// absorption kernel reads from `div_u*` directly and is unaffected.  Saves 3 × N
    /// element memcpy operations per step.
    ///
    /// **Fused PML + density update**: when `self.pml_exp` is populated (CPML boundary,
    /// no Dirichlet bypass), the split-field PML is applied inline:
    /// ```text
    ///   ρ_x^{n+1}[i,j,k] = p[i] · (p[i] · ρ_x^n − Δt · coef · ∂u_x/∂x)
    /// ```
    /// where `p[i] = pml_den_x[i] = exp(-σ_x[i]·Δt/2)`.  This replaces the previous
    /// `apply_pml_to_density()` pre/post calls with a single Zip pass per density component,
    /// saving 2 × N element writes per axis per step (6 passes eliminated for 3D).
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[inline]
    pub(crate) fn update_density_cartesian(&mut self, dt: f64) -> KwaversResult<()> {
        let has_y = self.grid.ny > 1;
        let has_z = self.grid.nz > 1;

        // ── Opt: IFFT directly into div_u* — eliminates 3 × N memcpy ──────────
        // dux/dx with negative shift + kappa correction (Treeby & Cox 2010, Eq. 17).
        self.fft.forward_r2c_into(&self.fields.ux, &mut self.ux_k);
        {
            let ddx = self.ddx_k_shift_neg.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.ux_k.view())
                .and(self.kappa.view())
                .par_for_each(|(i, _j, _k), gk, &u, &kap| {
                    *gk = (ddx[i] * u) * kap;
                });
        }
        // Write IFFT result directly to div_ux; dpx is not used for density.
        self.fft
            .inverse_c2r_into(&self.grad_k, &mut self.div_ux, &mut self.ux_k);

        // duy/dy with negative shift + kappa (matches k-Wave Eq. 17).
        if has_y {
            self.fft.forward_r2c_into(&self.fields.uy, &mut self.ux_k);
            let ddy = self.ddy_k_shift_neg.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.ux_k.view())
                .and(self.kappa.view())
                .par_for_each(|(_i, j, _k), gk, &u, &kap| {
                    *gk = (ddy[j] * u) * kap;
                });
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.div_uy, &mut self.ux_k);
        } else {
            self.div_uy.fill(0.0);
        }

        // duz/dz with negative shift + kappa (matches k-Wave Eq. 17).
        if has_z {
            self.fft.forward_r2c_into(&self.fields.uz, &mut self.ux_k);
            let ddz = self.ddz_k_shift_neg.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.ux_k.view())
                .and(self.kappa.view())
                .par_for_each(|(_i, _j, k), gk, &u, &kap| {
                    *gk = (ddz[k] * u) * kap;
                });
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.div_uz, &mut self.ux_k);
        } else {
            self.div_uz.fill(0.0);
        }

        // ── Mass-conservation density update ───────────────────────────────────
        // Linear:    coef = rho0
        // Nonlinear: coef = rho0 + 2·(rhox + rhoy + rhoz)   [Westervelt]
        //
        // When pml_exp is available and there is no Dirichlet bypass, the PML is
        // fused into the density update: ρ = pml · (pml · ρ_old − Δt · coef · ∂u/∂α)
        // eliminating the two apply_pml_to_density() calls (pre + post, 6 field
        // passes for 3-D) in favour of one additional factor per element in the
        // already-necessary update loop.
        let use_fused = self.pml_exp.is_some() && self.dirichlet_pml_bypass_x.is_empty();

        if self.config.nonlinearity {
            Zip::from(&mut self.div_u)
                .and(&self.materials.rho0)
                .and(&self.rhox)
                .and(&self.rhoy)
                .and(&self.rhoz)
                .par_for_each(|coef, &rho0, &rx, &ry, &rz| {
                    *coef = 2.0f64.mul_add(rx + ry + rz, rho0);
                });

            if use_fused {
                let pml_exp = self.pml_exp.as_ref().ok_or_else(|| {
                    KwaversError::InternalError(
                        "pml_exp unexpectedly None in nonlinear density fused path".into(),
                    )
                })?;
                let pml_dx = pml_exp.den_x.as_slice().ok_or_else(|| {
                    KwaversError::InternalError("pml_den_x must be contiguous".into())
                })?;
                Zip::indexed(self.rhox.view_mut())
                    .and(&self.div_ux)
                    .and(&self.div_u)
                    .par_for_each(|(i, _j, _k), rho, &du, &coef| {
                        let p = pml_dx[i];
                        *rho = p * (p * *rho - dt * coef * du);
                    });

                if has_y {
                    let pml_dy = pml_exp.den_y.as_slice().ok_or_else(|| {
                        KwaversError::InternalError("pml_den_y must be contiguous".into())
                    })?;
                    Zip::indexed(self.rhoy.view_mut())
                        .and(&self.div_uy)
                        .and(&self.div_u)
                        .par_for_each(|(_i, j, _k), rho, &du, &coef| {
                            let p = pml_dy[j];
                            *rho = p * (p * *rho - dt * coef * du);
                        });
                }

                if has_z {
                    let pml_dz = pml_exp.den_z.as_slice().ok_or_else(|| {
                        KwaversError::InternalError("pml_den_z must be contiguous".into())
                    })?;
                    Zip::indexed(self.rhoz.view_mut())
                        .and(&self.div_uz)
                        .and(&self.div_u)
                        .par_for_each(|(_i, _j, k), rho, &du, &coef| {
                            let p = pml_dz[k];
                            *rho = p * (p * *rho - dt * coef * du);
                        });
                }
            } else {
                // Fallback: pre-PML → update → post-PML
                self.apply_pml_to_density()?;

                Zip::from(&mut self.rhox)
                    .and(&self.div_ux)
                    .and(&self.div_u)
                    .par_for_each(|rho, &du, &coef| {
                        *rho -= dt * coef * du;
                    });

                if has_y {
                    Zip::from(&mut self.rhoy)
                        .and(&self.div_uy)
                        .and(&self.div_u)
                        .par_for_each(|rho, &du, &coef| {
                            *rho -= dt * coef * du;
                        });
                }

                if has_z {
                    Zip::from(&mut self.rhoz)
                        .and(&self.div_uz)
                        .and(&self.div_u)
                        .par_for_each(|rho, &du, &coef| {
                            *rho -= dt * coef * du;
                        });
                }

                self.apply_pml_to_density()?;
            }
        } else {
            // Linear case
            if use_fused {
                let pml_exp = self.pml_exp.as_ref().ok_or_else(|| {
                    KwaversError::InternalError(
                        "pml_exp unexpectedly None in linear density fused path".into(),
                    )
                })?;
                let pml_dx = pml_exp.den_x.as_slice().ok_or_else(|| {
                    KwaversError::InternalError("pml_den_x must be contiguous".into())
                })?;
                Zip::indexed(self.rhox.view_mut())
                    .and(&self.div_ux)
                    .and(&self.materials.rho0)
                    .par_for_each(|(i, _j, _k), rho, &du, &rho0| {
                        let p = pml_dx[i];
                        *rho = p * (p * *rho - dt * rho0 * du);
                    });

                if has_y {
                    let pml_dy = pml_exp.den_y.as_slice().ok_or_else(|| {
                        KwaversError::InternalError("pml_den_y must be contiguous".into())
                    })?;
                    Zip::indexed(self.rhoy.view_mut())
                        .and(&self.div_uy)
                        .and(&self.materials.rho0)
                        .par_for_each(|(_i, j, _k), rho, &du, &rho0| {
                            let p = pml_dy[j];
                            *rho = p * (p * *rho - dt * rho0 * du);
                        });
                }

                if has_z {
                    let pml_dz = pml_exp.den_z.as_slice().ok_or_else(|| {
                        KwaversError::InternalError("pml_den_z must be contiguous".into())
                    })?;
                    Zip::indexed(self.rhoz.view_mut())
                        .and(&self.div_uz)
                        .and(&self.materials.rho0)
                        .par_for_each(|(_i, _j, k), rho, &du, &rho0| {
                            let p = pml_dz[k];
                            *rho = p * (p * *rho - dt * rho0 * du);
                        });
                }
            } else {
                // Fallback: pre-PML → update → post-PML
                self.apply_pml_to_density()?;

                Zip::from(&mut self.rhox)
                    .and(&self.div_ux)
                    .and(&self.materials.rho0)
                    .par_for_each(|rho, &du, &rho0| {
                        *rho -= dt * rho0 * du;
                    });

                if has_y {
                    Zip::from(&mut self.rhoy)
                        .and(&self.div_uy)
                        .and(&self.materials.rho0)
                        .par_for_each(|rho, &du, &rho0| {
                            *rho -= dt * rho0 * du;
                        });
                }

                if has_z {
                    Zip::from(&mut self.rhoz)
                        .and(&self.div_uz)
                        .and(&self.materials.rho0)
                        .par_for_each(|rho, &du, &rho0| {
                            *rho -= dt * rho0 * du;
                        });
                }

                self.apply_pml_to_density()?;
            }
        }

        Ok(())
    }
}

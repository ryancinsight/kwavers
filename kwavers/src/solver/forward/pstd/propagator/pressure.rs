//! Pressure and density field updates for spectral solver.
//!
//! # Physics: Mass Conservation and Equation of State
//!
//! ## Background
//! The linearized Euler continuity (mass conservation) equation is:
//! ```text
//!   ∂ρ/∂t = −ρ₀ ∇·u − u·∇ρ₀
//! ```
//! Using a split-density formulation ρ = ρx + ρy + ρz, each component is updated
//! by the corresponding velocity divergence term and pressure is recovered via EOS.
//!
//! ## Theorem: Spectral Divergence with Negative Staggered Shift and Kappa
//! The x-component of velocity divergence on the staggered grid is:
//! ```text
//!   ∂ux/∂x |ₓ = IFFT( iκₓ · exp(−iκₓ Δx/2) · κ(k) · FFT(ux) )
//! ```
//! The negative shift compensates for ux being evaluated at i+½; differencing back
//! to the cell center uses exp(−iκₓ Δx/2). The operator `iκₓ · exp(−iκₓ Δx/2)`
//! is stored in `ddx_k_shift_neg`, and `κ(k) = sinc(c_ref·dt·|k|/2)` is the
//! kappa k-space correction stored in `self.kappa`. Both factors are applied,
//! matching Treeby & Cox (2010) Eq. 17 and the k-Wave C++ implementation.
//!
//! ## Theorem: Split-Density Equation of State
//! The acoustic equation of state for linear propagation is:
//! ```text
//!   p = c₀² · (ρx + ρy + ρz)
//! ```
//! where (ρx, ρy, ρz) are the split density perturbation components in kg/m³.
//! For a desired pressure perturbation Δp, the required density injection per
//! component is:
//! ```text
//!   Δρx = Δρy = Δρz = Δp / (3 c₀²)
//! ```
//! This is the additive pressure source scaling implemented in `stepper.rs`.
//!
//! ## Split-Field PML Update Order (Density)
//! K-Wave's PML for density (Treeby & Cox 2010, Eq. 16):
//! ```text
//!   ρx^{n+1} = pml_x · (pml_x · ρx^n  −  Δt · ρ₀ · ∂ux/∂x^{n+½})
//! ```
//! where `pml_x = exp(−σₓ · Δt/2)` uses the **collocated (non-staggered) sigma**
//! because density ρx lives at the same cell-center position as pressure.
//! The double application gives:
//! - ρx^n decays by `pml_x² = exp(−σₓ · Δt)` per step.
//! - The divergence term is attenuated by `pml_x = exp(−σₓ · Δt/2)`.
//!
//! *Note:* The PML must be applied BEFORE the divergence update (pre-step) AND
//! again AFTER (post-step) to match k-Wave's formulation. Applying only post-step
//! gives an incorrect single-factor attenuation.
//!
//! ## Power-Law Absorption
//! Absorption is implemented via fractional Laplacian operators following
//! Treeby & Cox (2010) Eq. 9–10. The absorb_tau and absorb_eta fields encode
//! the power-law exponents for each grid cell.
//!
//! ## References
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314.
//! - Liu (1998). Geophysics 63(6), 2082–2089.
//! - Caputo (1967). Geophys. J. Int. 13(5), 529–539. (fractional calculus)

use crate::core::error::KwaversResult;
use crate::math::fft::Complex64;
use crate::solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use crate::solver::geometry::Geometry;
use ndarray::{s, Array2, Zip};

impl PSTDSolver {
    /// Update pressure field from density perturbation (Equation of State)
    ///
    /// In C++ k-wave, this is the final step: p = c² * (rhox + rhoy + rhoz)
    /// Absorption is applied in `update_density()`, not here, matching the ordering
    /// in k-Wave's `computeDensity()` function (Treeby & Cox 2010, Eq. 21).
    #[inline]
    pub(crate) fn update_pressure(&mut self, _dt: f64) -> KwaversResult<()> {
        // Combine split density components
        Zip::from(&mut self.div_u)
            .and(&self.rhox)
            .and(&self.rhoy)
            .and(&self.rhoz)
            .par_for_each(|rho_sum, &rx, &ry, &rz| {
                *rho_sum = rx + ry + rz;
            });

        if self.config.nonlinearity {
            Zip::from(&mut self.fields.p)
                .and(&self.div_u)
                .and(&self.materials.c0)
                .and(&self.bon)
                .and(&self.materials.rho0)
                .par_for_each(|p, &rho_sum, &c, &bon, &rho0| {
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
                .par_for_each(|p, &rho_sum, &c| {
                    *p = c * c * rho_sum;
                });
        }

        Ok(())
    }

    /// Update density field based on velocity divergence (Mass Conservation).
    ///
    /// Dispatches to [`update_density_as`] when `config.geometry == CylindricalAS`,
    /// otherwise uses the standard 3-D spectral path.
    #[inline]
    pub(crate) fn update_density(&mut self, dt: f64) -> KwaversResult<()> {
        if self.config.geometry == Geometry::CylindricalAS {
            return self.update_density_as(dt);
        }
        self.update_density_cartesian(dt)
    }

    /// Standard 3-D Cartesian density update.
    ///
    /// Uses staggered grid shift operators with kappa correction matching the C++ k-wave binary:
    ///   dux/dx = IFFT( ddx_k_shift_neg[x] * kappa[i,j,k] * FFT(ux)[i,j,k] )
    ///
    /// kappa IS applied here — Treeby & Cox (2010) Eq. 17 explicitly includes the k-space
    /// correction factor κ in the density update, same as in the velocity update (Eq. 16).
    #[inline]
    pub(crate) fn update_density_cartesian(&mut self, dt: f64) -> KwaversResult<()> {
        let has_y = self.grid.ny > 1;
        let has_z = self.grid.nz > 1;

        // k-Wave split-field PML for density (Treeby & Cox 2010, Eq. 16):
        //   rho_x_new = pml_x * (pml_x * rho_x_old - dt * rho0 * dux/dx)
        //
        // Same double-application rule as velocity: pml applied pre- and post-update.
        // Pre: attenuate old split-density component by pml_x / pml_y / pml_z.
        // Post: attenuate the complete result again.
        self.apply_pml_to_density()?; // pre: pml * rho_old

        // nz_c: half-spectrum z-length (nz/2+1); ux_k/grad_k have shape (nx, ny, nz_c).
        let nz_c = self.ux_k.dim().2;

        // dux/dx with negative shift + kappa correction (Treeby & Cox 2010, Eq. 17).
        // Full operator: iκₓ · exp(−iκₓ Δx/2) · κ(k) where κ(k)=sinc(c_ref·dt·|k|/2).
        // R2C forward: real velocity (nx,ny,nz) → half-spectrum (nx,ny,nz_c).
        // kappa sliced to (nx,ny,nz_c) — symmetric under kz sign flip.
        // grad_k is reused for each axis sequentially (one axis at a time).
        self.fft.forward_r2c_into(&self.fields.ux, &mut self.ux_k);
        {
            let ddx = self.ddx_k_shift_neg.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.ux_k.view())
                .and(self.kappa.slice(s![.., .., ..nz_c]))
                .par_for_each(|(i, _j, _k), gk, &u, &kap| {
                    *gk = ddx[i] * Complex64::new(kap, 0.0) * u;
                });
        }
        self.fft
            .inverse_c2r_into(&self.grad_k, &mut self.dpx, &mut self.ux_k);

        // duy/dy with negative shift + kappa (matches k-Wave Eq. 17).
        // Singleton embedding axes have k_y = 0 for all modes, so the spectral
        // divergence contribution is exactly zero and no FFT is required.
        if has_y {
            self.fft.forward_r2c_into(&self.fields.uy, &mut self.uy_k);
            let ddy = self.ddy_k_shift_neg.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.uy_k.view())
                .and(self.kappa.slice(s![.., .., ..nz_c]))
                .par_for_each(|(_i, j, _k), gk, &u, &kap| {
                    *gk = ddy[j] * Complex64::new(kap, 0.0) * u;
                });
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.dpy, &mut self.uy_k);
        }

        // duz/dz with negative shift + kappa (matches k-Wave Eq. 17).
        // For nz = 1, k_z = 0 and the z-divergence term is identically zero.
        // ddz has length nz_c (truncated in construction); k ∈ [0, nz_c) from grad_k.
        if has_z {
            self.fft.forward_r2c_into(&self.fields.uz, &mut self.uz_k);
            let ddz = self.ddz_k_shift_neg.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.uz_k.view())
                .and(self.kappa.slice(s![.., .., ..nz_c]))
                .par_for_each(|(_i, _j, k), gk, &u, &kap| {
                    *gk = ddz[k] * Complex64::new(kap, 0.0) * u;
                });
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.dpz, &mut self.uz_k);
        }

        // ── Mass-conservation coefficient snapshot ──────────────────────────
        // Linear:    coef = rho0
        // Nonlinear: coef = rho0 + 2·(rhox + rhoy + rhoz)   [Westervelt]
        //
        // Matches k-Wave MATLAB kspaceFirstOrder3D.m lines 919–924:
        //   rho0_plus_rho = 2*(rhox+rhoy+rhoz) + rho0
        //   rhox = pml_x * (pml_x*rhox - dt * rho0_plus_rho * pml_x * duxdx)
        //
        // The snapshot is taken BEFORE any rhox/rhoy/rhoz modification so
        // all three axis updates see the same coefficient (matches k-Wave's
        // fully-coupled formulation). `self.div_u` is reused as scratch: it
        // is rewritten unconditionally at the top of `update_pressure` that
        // follows, so overwriting it here is safe and allocation-free.
        if self.config.nonlinearity {
            Zip::from(&mut self.div_u)
                .and(&self.materials.rho0)
                .and(&self.rhox)
                .and(&self.rhoy)
                .and(&self.rhoz)
                .par_for_each(|coef, &rho0, &rx, &ry, &rz| {
                    *coef = rho0 + 2.0 * (rx + ry + rz);
                });

            // Update split density components using the nonlinear coefficient snapshot.
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
            // Linear k-Wave form: rho_axis -= dt * rho0 * du_axis/daxis.
            // Read rho0 directly to avoid copying it into div_u every step.
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

        // Apply power-law absorption correction per axis (Treeby & Cox 2010, Eq. 21).
        // Must be called AFTER the continuity-equation update (so dpx/dpy/dpz hold
        // ∂u_α/∂α) and BEFORE the post-PML multiply, matching k-Wave's computeDensity().
        self.apply_absorption(dt)?;

        self.apply_pml_to_density()?; // post: pml * (pml*rho_old - dt*rho0*div_u + absorption)

        // NOTE: Pressure computation and absorption are handled separately in
        // update_pressure(). Source injection happens between density and pressure
        // updates (matching C++ k-wave binary time loop order).

        Ok(())
    }

    /// Axisymmetric WSWA-FFT density update.
    ///
    /// Updates `rhox` (axial split density) and `rhoz` (radial split density).
    /// `rhoy` is not used (ny = 1 in AS mode; remains zero).
    ///
    /// # Equations (k-Wave AS, linearised)
    /// ```text
    /// rhox -= dt * rho0 * dux/dx              (axial continuity)
    /// rhoz -= dt * rho0 * (dur/dr + ur/r)     (cylindrical radial continuity)
    /// ```
    pub(crate) fn update_density_as(&mut self, dt: f64) -> KwaversResult<()> {
        let nx = self.grid.nx;

        self.apply_pml_to_density()?; // pre-step PML

        // Take AsContext out of the Option to enable split borrows with
        // self.fields / self.materials / self.rhox / self.rhoz.
        // No heap allocation: take/replace are pointer moves only.
        let mut ctx = self
            .as_ctx
            .take()
            .expect("AsContext must be Some for CylindricalAS");

        ctx.compute_density_divs(
            self.fields.ux.slice(s![.., 0, ..]),
            self.fields.uz.slice(s![.., 0, ..]),
        );

        // Compute rho0-based coefficient (or nonlinear coefficient).
        let nr = ctx.nr;
        let mut coef = Array2::<f64>::zeros((nx, nr));
        if self.config.nonlinearity {
            Zip::from(&mut coef)
                .and(self.materials.rho0.slice(s![.., 0, ..]))
                .and(self.rhox.slice(s![.., 0, ..]))
                .and(self.rhoz.slice(s![.., 0, ..]))
                .for_each(|c, &rho0, &rx, &rz| {
                    *c = rho0 + 2.0 * (rx + rz);
                });
        } else {
            coef.assign(&self.materials.rho0.slice(s![.., 0, ..]));
        }

        Zip::from(self.rhox.slice_mut(s![.., 0, ..]))
            .and(&ctx.duxdx)
            .and(&coef)
            .for_each(|rho, &du, &c| {
                *rho -= dt * c * du;
            });

        Zip::from(self.rhoz.slice_mut(s![.., 0, ..]))
            .and(&ctx.duzdr)
            .and(&coef)
            .for_each(|rho, &du, &c| {
                *rho -= dt * c * du;
            });

        // Copy divergences into 3-D scratch buffers for absorption, then restore ctx.
        self.dpx.slice_mut(s![.., 0, ..]).assign(&ctx.duxdx);
        self.dpy.fill(0.0);
        self.dpz.slice_mut(s![.., 0, ..]).assign(&ctx.duzdr);
        self.as_ctx = Some(ctx);
        self.apply_absorption(dt)?;

        self.apply_pml_to_density()?; // post-step PML
        Ok(())
    }

    /// Apply split-field directional PML damping to split-density components.
    ///
    /// Each density component is damped only by its corresponding directional sigma,
    /// matching k-Wave's formulation: `rho_x *= pml_x`, `rho_y *= pml_y`, `rho_z *= pml_z`.
    fn apply_pml_to_density(&mut self) -> KwaversResult<()> {
        if let Some(boundary) = self.boundary.as_deref_mut() {
            boundary.apply_acoustic_directional(
                self.rhox.view_mut(),
                self.grid.as_ref(),
                self.time_step_index,
                0, // x-direction sigma for rhox
            )?;
            boundary.apply_acoustic_directional(
                self.rhoy.view_mut(),
                self.grid.as_ref(),
                self.time_step_index,
                1, // y-direction sigma for rhoy
            )?;
            boundary.apply_acoustic_directional(
                self.rhoz.view_mut(),
                self.grid.as_ref(),
                self.time_step_index,
                2, // z-direction sigma for rhoz
            )?;
        }
        Ok(())
    }
}

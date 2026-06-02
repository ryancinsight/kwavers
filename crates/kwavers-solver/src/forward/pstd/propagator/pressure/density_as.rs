use kwavers_core::error::{KwaversError, KwaversResult};
use crate::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use ndarray::{s, Zip};

impl PSTDSolver {
    /// Axisymmetric WSWA-FFT density update.
    ///
    /// Updates `rhox` (axial split density) and `rhoz` (radial split density).
    /// `rhoy` is not used (ny = 1 in AS mode; remains zero).
    ///
    /// # Equations: split-field PML (Treeby & Cox 2010, Eq. 16)
    /// ```text
    /// rhox^{n+1}[i,k] = pml_x[i] · (pml_x[i] · rhox^n − Δt · coef · ∂ux/∂x)
    /// rhoz^{n+1}[i,k] = pml_z[k] · (pml_z[k] · rhoz^n − Δt · coef · (∂ur/∂r + ur/r))
    /// ```
    /// where `pml_x[i] = exp(-σ_x[i]·Δt/2)` (collocated sigma) and
    ///       `pml_z[k] = exp(-σ_z[k]·Δt/2)` (collocated sigma, r-axis mapped to z).
    ///
    /// Linear: `coef = ρ₀`.  Nonlinear (Westervelt): `coef = ρ₀ + 2·(ρx + ρz)`.
    ///
    /// **Fused path** (CPML, no Dirichlet bypass): pre-computed `pml_den_x/z` from
    /// `self.pml_exp` are applied inline — eliminates the two `apply_pml_to_density()`
    /// calls (each evaluating per-element `exp()` for all AS cells).
    ///
    /// **Fallback path**: original pre-PML → update → post-PML structure preserved.
    ///
    /// **Divergence cache**: divergences are written into `div_ux`/`div_uz` (not `dpx`/`dpz`)
    /// so that `apply_absorption_to_pressure` reads the correct values at Step 1
    /// (which fuses div_u* into `dpx` via a single Zip — Opt-7 + Opt-12).
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    /// - Returns [`KwaversError::InternalError`] if `AsContext` is unexpectedly `None`
    ///   for `CylindricalAS` geometry.
    ///
    pub(crate) fn update_density_as(&mut self, dt: f64) -> KwaversResult<()> {
        let use_fused = self.pml_exp.is_some() && self.dirichlet_pml_bypass_x.is_empty();

        if !use_fused {
            self.apply_pml_to_density()?; // pre-step PML (fallback only)
        }

        // Take AsContext out of the Option to enable split borrows with
        // self.fields / self.materials / self.rhox / self.rhoz.
        // No heap allocation: take/replace are pointer moves only.
        let mut ctx = self.as_ctx.take().ok_or_else(|| {
            KwaversError::InternalError("AsContext unexpectedly None for CylindricalAS".into())
        })?;

        ctx.compute_density_divs(
            self.fields.ux.slice(s![.., 0, ..]),
            self.fields.uz.slice(s![.., 0, ..]),
        );

        // Populate the pre-allocated coefficient scratch (no heap allocation).
        // In fused mode, coef uses current (old) rhox/rhoz — consistent with
        // update_density_cartesian fused path.  In fallback mode, coef uses the
        // pre-PML'd rhox/rhoz (same as the previous non-fused AS implementation).
        if self.config.nonlinearity {
            Zip::from(&mut ctx.coef)
                .and(self.materials.rho0.slice(s![.., 0, ..]))
                .and(self.rhox.slice(s![.., 0, ..]))
                .and(self.rhoz.slice(s![.., 0, ..]))
                .par_for_each(|c, &rho0, &rx, &rz| {
                    *c = 2.0f64.mul_add(rx + rz, rho0);
                });
        } else {
            ctx.coef.assign(&self.materials.rho0.slice(s![.., 0, ..]));
        }

        if use_fused {
            // Fused: rhox = pml_x[i]·(pml_x[i]·rhox − dt·coef·duxdx)
            //        rhoz = pml_z[k]·(pml_z[k]·rhoz − dt·coef·duzdr)
            // In the 2-D slice (nx, nr), Zip::indexed returns (i, k).
            let pml_exp = self.pml_exp.as_ref().ok_or_else(|| {
                KwaversError::InternalError("pml_exp unexpectedly None in fused AS density path".into())
            })?;
            let pml_dx = pml_exp.den_x.as_slice().ok_or_else(|| {
                KwaversError::InternalError("pml_den_x must be contiguous".into())
            })?;
            let pml_dz = pml_exp.den_z.as_slice().ok_or_else(|| {
                KwaversError::InternalError("pml_den_z must be contiguous".into())
            })?;

            Zip::indexed(self.rhox.slice_mut(s![.., 0, ..]))
                .and(&ctx.duxdx)
                .and(&ctx.coef)
                .par_for_each(|(i, _k), rho, &du, &c| {
                    let p = pml_dx[i];
                    *rho = p * (p * *rho - dt * c * du);
                });

            Zip::indexed(self.rhoz.slice_mut(s![.., 0, ..]))
                .and(&ctx.duzdr)
                .and(&ctx.coef)
                .par_for_each(|(_i, k), rho, &du, &c| {
                    let p = pml_dz[k];
                    *rho = p * (p * *rho - dt * c * du);
                });
        } else {
            Zip::from(self.rhox.slice_mut(s![.., 0, ..]))
                .and(&ctx.duxdx)
                .and(&ctx.coef)
                .par_for_each(|rho, &du, &c| {
                    *rho -= dt * c * du;
                });

            Zip::from(self.rhoz.slice_mut(s![.., 0, ..]))
                .and(&ctx.duzdr)
                .and(&ctx.coef)
                .par_for_each(|rho, &du, &c| {
                    *rho -= dt * c * du;
                });

            self.apply_pml_to_density()?; // post-step PML (fallback only)
        }

        // Write divergences into div_ux/div_uz (the divergence cache).
        // apply_absorption_to_pressure fuses div_ux/div_uy/div_uz → dpx at Step 1 (Opt-7+12).
        // Writing to div_u* here (not dpx) ensures absorption receives the correct AS values.
        self.div_ux.slice_mut(s![.., 0, ..]).assign(&ctx.duxdx);
        self.div_uy.fill(0.0);
        self.div_uz.slice_mut(s![.., 0, ..]).assign(&ctx.duzdr);
        self.as_ctx = Some(ctx);
        Ok(())
    }
}

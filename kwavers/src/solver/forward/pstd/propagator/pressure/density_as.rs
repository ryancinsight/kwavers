use crate::core::error::KwaversResult;
use crate::solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use ndarray::{s, Zip};

impl PSTDSolver {
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if `AsContext must be Some for CylindricalAS`.
    ///
    pub(crate) fn update_density_as(&mut self, dt: f64) -> KwaversResult<()> {
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

        // Populate the pre-allocated coefficient scratch (no heap allocation).
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

        // Write divergences into div_ux/div_uy/div_uz (the divergence cache).
        // `apply_absorption_to_pressure` reads from div_u* at Step 1 to build
        // ρ₀·∇·u — writing to dpx/dpy/dpz here left the AS divergences out of
        // the absorption L1 term.
        self.div_ux.slice_mut(s![.., 0, ..]).assign(&ctx.duxdx);
        self.div_uy.fill(0.0);
        self.div_uz.slice_mut(s![.., 0, ..]).assign(&ctx.duzdr);
        self.as_ctx = Some(ctx);

        self.apply_pml_to_density()?; // post-step PML
        Ok(())
    }
}

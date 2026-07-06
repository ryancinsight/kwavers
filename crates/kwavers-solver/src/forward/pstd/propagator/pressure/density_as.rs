use crate::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use kwavers_core::error::{KwaversError, KwaversResult};
use moirai_parallel::{enumerate_mut_with, Adaptive};
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2};

#[derive(Clone, Copy)]
enum AsAxis {
    X,
    R,
}

#[inline]
fn dense_indices(index: usize, nr: usize) -> (usize, usize) {
    (index / nr, index % nr)
}

#[inline]
fn pml_index(axis: AsAxis, i: usize, k: usize) -> usize {
    match axis {
        AsAxis::X => i,
        AsAxis::R => k,
    }
}

fn compute_axisymmetric_coefficient(
    coefficient: &mut Array2<f64>,
    rho0: ArrayView2<'_, f64>,
    rhox: ArrayView2<'_, f64>,
    rhoz: ArrayView2<'_, f64>,
) {
    assert_eq!(
        coefficient.shape(),
        rho0.shape(),
        "invariant: AS density coefficient shape matches rho0"
    );
    assert_eq!(
        coefficient.shape(),
        rhox.shape(),
        "invariant: AS density coefficient shape matches rhox"
    );
    assert_eq!(
        coefficient.shape(),
        rhoz.shape(),
        "invariant: AS density coefficient shape matches rhoz"
    );

    if let (Some(coef_values), Some(rho0_values), Some(rx_values), Some(rz_values)) = (
        coefficient.as_slice_memory_order_mut(),
        rho0.as_slice_memory_order(),
        rhox.as_slice_memory_order(),
        rhoz.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(coef_values, |index, coefficient| {
            *coefficient = 2.0f64.mul_add(rx_values[index] + rz_values[index], rho0_values[index]);
        });
        return;
    }

    let (nx, nr) = coefficient.dim();
    for k in 0..nr {
        for i in 0..nx {
            coefficient[[i, k]] = 2.0f64.mul_add(rhox[[i, k]] + rhoz[[i, k]], rho0[[i, k]]);
        }
    }
}

fn update_axisymmetric_density_fused(
    mut density: ArrayViewMut2<'_, f64>,
    divergence: &Array2<f64>,
    coefficient: &Array2<f64>,
    pml: &[f64],
    axis: AsAxis,
    dt: f64,
) {
    assert_eq!(
        density.shape(),
        divergence.shape(),
        "invariant: AS density shape matches divergence"
    );
    assert_eq!(
        density.shape(),
        coefficient.shape(),
        "invariant: AS density shape matches update coefficient"
    );

    let (_nx, nr) = density.dim();
    if let (Some(density_values), Some(div_values), Some(coef_values)) = (
        density.as_slice_memory_order_mut(),
        divergence.as_slice_memory_order(),
        coefficient.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(density_values, |index, density| {
            let (i, k) = dense_indices(index, nr);
            let p = pml[pml_index(axis, i, k)];
            *density = p * (p * *density - dt * coef_values[index] * div_values[index]);
        });
        return;
    }

    let (nx, nr) = density.dim();
    for k in 0..nr {
        for i in 0..nx {
            let p = pml[pml_index(axis, i, k)];
            density[[i, k]] =
                p * (p * density[[i, k]] - dt * coefficient[[i, k]] * divergence[[i, k]]);
        }
    }
}

fn update_axisymmetric_density_unfused(
    mut density: ArrayViewMut2<'_, f64>,
    divergence: &Array2<f64>,
    coefficient: &Array2<f64>,
    dt: f64,
) {
    assert_eq!(
        density.shape(),
        divergence.shape(),
        "invariant: AS density shape matches divergence"
    );
    assert_eq!(
        density.shape(),
        coefficient.shape(),
        "invariant: AS density shape matches update coefficient"
    );

    if let (Some(density_values), Some(div_values), Some(coef_values)) = (
        density.as_slice_memory_order_mut(),
        divergence.as_slice_memory_order(),
        coefficient.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(density_values, |index, density| {
            *density -= dt * coef_values[index] * div_values[index];
        });
        return;
    }

    let (nx, nr) = density.dim();
    for k in 0..nr {
        for i in 0..nx {
            density[[i, k]] -= dt * coefficient[[i, k]] * divergence[[i, k]];
        }
    }
}

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
            compute_axisymmetric_coefficient(
                &mut ctx.coef,
                self.materials.rho0.slice(s![.., 0, ..]),
                self.rhox.slice(s![.., 0, ..]),
                self.rhoz.slice(s![.., 0, ..]),
            );
        } else {
            ctx.coef.assign(&self.materials.rho0.slice(s![.., 0, ..]));
        }

        if use_fused {
            // Fused: rhox = pml_x[i]·(pml_x[i]·rhox − dt·coef·duxdx)
            //        rhoz = pml_z[k]·(pml_z[k]·rhoz − dt·coef·duzdr)
            // In the 2-D slice (nx, nr), dense row-major indexing maps to (i, k).
            let pml_exp = self.pml_exp.as_ref().ok_or_else(|| {
                KwaversError::InternalError(
                    "pml_exp unexpectedly None in fused AS density path".into(),
                )
            })?;
            let pml_dx = pml_exp.den_x.as_slice().ok_or_else(|| {
                KwaversError::InternalError("pml_den_x must be contiguous".into())
            })?;
            let pml_dz = pml_exp.den_z.as_slice().ok_or_else(|| {
                KwaversError::InternalError("pml_den_z must be contiguous".into())
            })?;

            update_axisymmetric_density_fused(
                self.rhox.slice_mut(s![.., 0, ..]),
                &ctx.duxdx,
                &ctx.coef,
                pml_dx,
                AsAxis::X,
                dt,
            );

            update_axisymmetric_density_fused(
                self.rhoz.slice_mut(s![.., 0, ..]),
                &ctx.duzdr,
                &ctx.coef,
                pml_dz,
                AsAxis::R,
                dt,
            );
        } else {
            update_axisymmetric_density_unfused(
                self.rhox.slice_mut(s![.., 0, ..]),
                &ctx.duxdx,
                &ctx.coef,
                dt,
            );

            update_axisymmetric_density_unfused(
                self.rhoz.slice_mut(s![.., 0, ..]),
                &ctx.duzdr,
                &ctx.coef,
                dt,
            );

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

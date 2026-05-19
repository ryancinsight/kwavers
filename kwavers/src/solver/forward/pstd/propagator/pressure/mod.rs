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

mod density_as;
mod density_cartesian;

use crate::core::error::KwaversResult;
use crate::solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use crate::solver::geometry::SolverGeometry;
use ndarray::Zip;

impl PSTDSolver {
    /// Update pressure field from density perturbation (Equation of State)
    /// and apply pressure-side power-law absorption.
    ///
    /// Matches the C++ k-Wave binary's `computePressureLinearPowerLaw` /
    /// `sumPressureTermsLinear`: pressure is first set from the EOS
    /// `p = c² · ρ_total` (linear) or with the Westervelt nonlinearity
    /// expansion, then the fractional-Laplacian absorption correction
    /// `p += c² · (τ · L1 − η · L2)` is added algebraically (no Δt) via
    /// [`Self::apply_absorption_to_pressure`].
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[inline]
    pub(crate) fn update_pressure(&mut self, _dt: f64) -> KwaversResult<()> {
        // ── EOS: populate div_u = ρ_total (needed by absorption L2,
        // Treeby & Cox 2010 Eq. 21) and set pressure simultaneously.
        //
        // Linear path (most common): single fused Zip — 6 arrays, within ndarray arity.
        //   OLD: Pass 1 writes div_u (3 reads+1 write), Pass 2 reads div_u→writes p (2+1)
        //   NEW: 1 pass writes div_u AND p (4 reads+2 writes) — saves 1 div_u read/write pair.
        //
        // Nonlinear path: 2 passes (8 arrays total exceed ndarray Zip arity=6).
        //   Pass 1: div_u = ρ_total  (4 arrays)
        //   Pass 2: p = c²·(ρ_total + bon/(2·ρ₀)·ρ_total²)  (5 arrays)
        //   Both passes use par_for_each (Rayon parallelism preserved).
        if self.config.nonlinearity {
            // SAFETY: `bon.is_some() ↔ config.nonlinearity` — enforced at construction.
            let bon = self.bon.as_ref().expect("bon populated iff nonlinearity");
            // Pass 1: accumulate split densities → ρ_total.
            Zip::from(&mut self.div_u)
                .and(&self.rhox)
                .and(&self.rhoy)
                .and(&self.rhoz)
                .par_for_each(|rho_sum, &rx, &ry, &rz| {
                    *rho_sum = rx + ry + rz;
                });
            // Pass 2: nonlinear EOS — p = c²·(ρ_total + bon/(2·ρ₀)·ρ_total²).
            Zip::from(&mut self.fields.p)
                .and(&self.div_u)
                .and(&self.materials.c0)
                .and(bon)
                .and(&self.materials.rho0)
                .par_for_each(|p, &rho_sum, &c, &bon_val, &rho0| {
                    let nonlinear = (bon_val / (2.0 * rho0)) * rho_sum * rho_sum;
                    *p = c * c * (rho_sum + nonlinear);
                });
        } else {
            // Fused single pass: div_u = ρ_total AND p = c²·ρ_total.
            Zip::from(&mut self.div_u)
                .and(&mut self.fields.p)
                .and(&self.rhox)
                .and(&self.rhoy)
                .and(&self.rhoz)
                .and(&self.materials.c0)
                .par_for_each(|div, p, &rx, &ry, &rz, &c| {
                    let rho_sum = rx + ry + rz;
                    *div = rho_sum;
                    *p = c * c * rho_sum;
                });
        }

        self.apply_absorption_to_pressure()?;
        Ok(())
    }

    /// Update density field based on velocity divergence (Mass Conservation).
    ///
    /// Dispatches to [`update_density_as`] when `config.geometry == CylindricalAS`,
    /// otherwise uses the standard 3-D spectral path.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[inline]
    pub(crate) fn update_density(&mut self, dt: f64) -> KwaversResult<()> {
        if self.config.geometry == SolverGeometry::CylindricalAS {
            return self.update_density_as(dt);
        }
        self.update_density_cartesian(dt)
    }

    /// Apply split-field directional PML damping to split-density components.
    ///
    /// Each density component is damped only by its corresponding directional sigma,
    /// matching k-Wave's formulation: `rho_x *= pml_x`, `rho_y *= pml_y`, `rho_z *= pml_z`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(crate) fn apply_pml_to_density(&mut self) -> KwaversResult<()> {
        let Some(mut boundary) = self.boundary.take() else {
            return Ok(());
        };

        let result = (|| -> KwaversResult<()> {
            if self.dirichlet_pml_bypass_x.is_empty() {
                boundary.apply_acoustic_directional(
                    self.rhox.view_mut(),
                    self.grid.as_ref(),
                    self.time_step_index,
                    0,
                )?;
                boundary.apply_acoustic_directional(
                    self.rhoy.view_mut(),
                    self.grid.as_ref(),
                    self.time_step_index,
                    1,
                )?;
                boundary.apply_acoustic_directional(
                    self.rhoz.view_mut(),
                    self.grid.as_ref(),
                    self.time_step_index,
                    2,
                )?;
            } else {
                self.resize_pml_bypass_scratch();
                let rows = self.dirichlet_pml_bypass_x.as_slice();
                let grid = self.grid.as_ref();
                let step = self.time_step_index;

                Self::apply_x_plane_pml_bypass(
                    &mut self.rhox,
                    rows,
                    &mut self.pml_bypass_plane_scratch,
                    |field| boundary.apply_acoustic_directional(field, grid, step, 0),
                )?;
                Self::apply_x_plane_pml_bypass(
                    &mut self.rhoy,
                    rows,
                    &mut self.pml_bypass_plane_scratch,
                    |field| boundary.apply_acoustic_directional(field, grid, step, 1),
                )?;
                Self::apply_x_plane_pml_bypass(
                    &mut self.rhoz,
                    rows,
                    &mut self.pml_bypass_plane_scratch,
                    |field| boundary.apply_acoustic_directional(field, grid, step, 2),
                )?;
            }
            Ok(())
        })();

        self.boundary = Some(boundary);
        result
    }
}

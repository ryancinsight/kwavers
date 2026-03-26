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
//! ## Theorem: Spectral Divergence with Negative Staggered Shift
//! The x-component of velocity divergence on the staggered grid is:
//! ```text
//!   ∂ux/∂x |ₓ = IFFT( iκₓ · exp(−iκₓ Δx/2) · κ(k) · FFT(ux) )
//! ```
//! The negative shift compensates for ux being evaluated at i+½; differencing back
//! to the cell center uses exp(−iκₓ Δx/2). The operator `iκₓ · exp(−iκₓ Δx/2)`
//! is stored in `ddx_k_shift_neg`. Full derivation mirrors the velocity gradient
//! in velocity.rs (shift theorem + spectral derivative).
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

        // k-Wave split-field PML for density (Treeby & Cox 2010, Eq. 16):
        //   rho_x_new = pml_x * (pml_x * rho_x_old - dt * rho0 * dux/dx)
        //
        // Same double-application rule as velocity: pml applied pre- and post-update.
        // Pre: attenuate old split-density component by pml_x / pml_y / pml_z.
        // Post: attenuate the complete result again.
        self.apply_pml_to_density()?; // pre: pml * rho_old

        // dux/dx with negative shift
        // Uses ndarray Zip::indexed to eliminate bounds checks and enable LLVM vectorization.
        // Shift operators are 1D (index i only) and are read via an immutable borrow that
        // is disjoint from the mutable borrow of grad_x_k (different struct fields).
        self.fft.forward_into(&self.fields.ux, &mut self.ux_k);
        {
            let ddx = self.ddx_k_shift_neg.view();
            Zip::indexed(self.grad_x_k.view_mut())
                .and(self.ux_k.view())
                .and(self.kappa.view())
                .for_each(|(i, _j, _k), gx, &u, &kap| {
                    *gx = ddx[i] * Complex64::new(kap, 0.0) * u;
                });
        }
        self.fft
            .inverse_into(&self.grad_x_k, &mut self.dpx, &mut self.ux_k);

        // duy/dy with negative shift
        self.fft.forward_into(&self.fields.uy, &mut self.uy_k);
        {
            let ddy = self.ddy_k_shift_neg.view();
            Zip::indexed(self.grad_y_k.view_mut())
                .and(self.uy_k.view())
                .and(self.kappa.view())
                .for_each(|(_i, j, _k), gy, &u, &kap| {
                    *gy = ddy[j] * Complex64::new(kap, 0.0) * u;
                });
        }
        self.fft
            .inverse_into(&self.grad_y_k, &mut self.dpy, &mut self.uy_k);

        // duz/dz with negative shift
        self.fft.forward_into(&self.fields.uz, &mut self.uz_k);
        {
            let ddz = self.ddz_k_shift_neg.view();
            Zip::indexed(self.grad_z_k.view_mut())
                .and(self.uz_k.view())
                .and(self.kappa.view())
                .for_each(|(_i, _j, k), gz, &u, &kap| {
                    *gz = ddz[k] * Complex64::new(kap, 0.0) * u;
                });
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

        self.apply_pml_to_density()?; // post: pml * (pml*rho_old - dt*rho0*div_u)

        // NOTE: Pressure computation and absorption are handled separately in
        // update_pressure(). Source injection happens between density and pressure
        // updates (matching C++ k-wave binary time loop order).

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

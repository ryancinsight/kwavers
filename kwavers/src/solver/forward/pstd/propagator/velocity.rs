//! Velocity field updates for spectral solver
//!
//! # Momentum Conservation with Split-Field PML
//!
//! ## Background
//! The linearized Euler momentum equation in a heterogeneous medium is:
//! ```text
//!   ∂u/∂t = −(1/ρ₀) ∇p
//! ```
//! On a staggered grid (k-Wave convention), pressure lives at cell centers and velocity
//! at cell edges shifted by ½ cell in each respective axis.
//!
//! ## Theorem: Spectral Gradient with Staggered Shift
//! Let p̂ = FFT(p) be the 3-D DFT of the pressure field. The spectral derivative
//! with a positive half-grid-point shift is:
//! ```text
//!   ∂p/∂x |ₓ₊Δₓ/₂ = IFFT( iκₓ · exp(+iκₓ Δx/2) · κ(k) · p̂ )
//! ```
//! where κₓ = 2π n / (Nₓ Δx) is the wavenumber, and κ(k) = sinc(c_ref Δt |k|/2) is
//! the k-space correction factor that improves temporal accuracy to spectral order.
//! The operator `iκₓ · exp(+iκₓ Δx/2)` is stored in `ddx_k_shift_pos`.
//!
//! *Proof:* By the shift theorem of the DFT, shifting by Δx/2 multiplies each mode
//! by exp(+iκₓ Δx/2). Multiplication by iκₓ implements the spectral x-derivative.
//! The k-space correction κ(k) = sinc(c_ref Δt |k|/2) reduces temporal phase
//! error to O(Δt²) for all spatial frequencies simultaneously (Liu 1998, §3).
//!
//! ## Split-Field PML Update Order
//! K-Wave's multiplicative split-field PML (Treeby & Cox 2010, Eq. 17) applies
//! the PML factor **twice** per time step for each velocity component:
//! ```text
//!   u_x^{n+1} = pml_x_sgx · (pml_x_sgx · u_x^n  −  Δt/ρ₀ · ∂p/∂x^{n+½})
//! ```
//! where `pml_x_sgx = exp(−σₓ_sg · Δt/2)` uses the **staggered-grid sigma** evaluated
//! at the half-cell-shifted position. The double application means:
//! - u_x^n is damped by `pml_x_sgx²  = exp(−σₓ_sg · Δt)` per step.
//! - The gradient term is damped by `pml_x_sgx = exp(−σₓ_sg · Δt/2)`.
//!
//! *Why staggered sigma?* The velocity u_x lives at position i+½, so the PML must
//! be evaluated there. Using the collocated sigma (at position i) over-damps velocity
//! by ≈20% at the deepest PML cell, where the staggered sigma is only
//! `(pml_size − 0.5)^4 / pml_size^4 ≈ 0.71 × σ_max` rather than σ_max.
//!
//! ## References
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314.
//! - Liu (1998). Geophysics 63(6), 2082–2089. (k-space PSTD method)
//! - Berenger (1994). J. Comput. Phys. 114(2), 185–200. (split-field PML)

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
        // k-Wave split-field PML for velocity (Treeby & Cox 2010, Eq. 17):
        //   u_new = pml * (pml * u_old - dt/rho * grad_p)
        //
        // The PML multiplier `pml = exp(-sigma * dt/2)` is applied TWICE per step:
        //   1. Pre-update: attenuate stored velocity by pml  → pml * u_old
        //   2. Post-update: attenuate the complete result    → pml * (pml*u_old - dt/rho*grad_p)
        //
        // This double application means u_old decays by pml^2 = exp(-sigma*dt) per step,
        // while the injected gradient term decays by only pml = exp(-sigma*dt/2).
        // Using only one application (pre or post) produces incorrect amplitude.
        self.apply_pml_to_velocity()?; // pre: pml * u_old

        // Transform pressure to k-space
        self.fft.forward_into(&self.fields.p, &mut self.p_k);

        // Compute pressure gradients in k-space with staggered grid shifts.
        // k-wave uses ddx_k_shift_pos for pressure→velocity (positive shift).
        // Zip::indexed eliminates bounds checks; all 3 gradient components are
        // written in a single pass over the k-space array (better cache utilisation).
        {
            let ddx = self.ddx_k_shift_pos.view();
            let ddy = self.ddy_k_shift_pos.view();
            let ddz = self.ddz_k_shift_pos.view();
            Zip::indexed(self.grad_x_k.view_mut())
                .and(self.grad_y_k.view_mut())
                .and(self.grad_z_k.view_mut())
                .and(self.p_k.view())
                .and(self.kappa.view())
                .for_each(|(i, j, k), gx, gy, gz, &p_val, &kap| {
                    let e_kappa = Complex64::new(kap, 0.0) * p_val;
                    *gx = ddx[i] * e_kappa;
                    *gy = ddy[j] * e_kappa;
                    *gz = ddz[k] * e_kappa;
                });
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

        // NOTE: Velocity source injection is NOT performed here.
        // It happens in step_forward() after update_velocity() returns,
        // matching the C++ k-wave binary time loop order (Step 2: addVelocitySource).

        self.apply_pml_to_velocity()?; // post: pml * (pml*u_old - dt/rho*grad_p)

        Ok(())
    }

    /// Apply split-field directional PML damping to velocity components.
    ///
    /// Each velocity component is damped only by its corresponding directional sigma,
    /// matching k-Wave's formulation: `ux *= pml_x`, `uy *= pml_y`, `uz *= pml_z`.
    /// Apply split-field directional PML to velocity components using staggered-grid sigma.
    ///
    /// Velocity fields are staggered at half-cell positions relative to pressure/density.
    /// K-Wave therefore uses `pml_x_sgx` / `pml_y_sgy` / `pml_z_sgz` (computed at i+0.5)
    /// rather than the collocated `pml_x` / `pml_y` / `pml_z` used for density.
    ///
    /// The staggered sigma is smaller at PML boundary cells (~70% of σ_max at deepest cell),
    /// so using non-staggered sigma for velocity over-damps it by ≈ 20%.
    pub(super) fn apply_pml_to_velocity(&mut self) -> KwaversResult<()> {
        if let Some(boundary) = self.boundary.as_deref_mut() {
            boundary.apply_velocity_pml_directional(
                self.fields.ux.view_mut(),
                self.grid.as_ref(),
                self.time_step_index,
                0, // staggered x-sigma for ux (pml_x_sgx)
            )?;
            boundary.apply_velocity_pml_directional(
                self.fields.uy.view_mut(),
                self.grid.as_ref(),
                self.time_step_index,
                1, // staggered y-sigma for uy (pml_y_sgy)
            )?;
            boundary.apply_velocity_pml_directional(
                self.fields.uz.view_mut(),
                self.grid.as_ref(),
                self.time_step_index,
                2, // staggered z-sigma for uz (pml_z_sgz)
            )?;
        }
        Ok(())
    }
}

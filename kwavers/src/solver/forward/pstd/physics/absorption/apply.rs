//! Pressure-side fractional Laplacian absorption correction for `PSTDSolver`.
//!
//! ## Theorem: Pressure-Side Fractional Laplacian Absorption
//! Treeby & Cox (2010) Eqs. 19–21, integrated with the equation of state into the
//! pressure update following the C++ k-Wave binary
//! (`KSpaceFirstOrderSolver::computePressureLinearPowerLaw` /
//! `sumPressureTermsLinear`):
//!
//! ```text
//!   p ← p  +  c₀² · ( τ · L1  −  η · L2 )
//! ```
//! with
//! ```text
//!   L1(x) = IFFT( |k|^(y−2) · FFT( ρ₀(x) · ∇·u(x) ) )
//!   L2(x) = IFFT( |k|^(y−1) · FFT( ρ_total(x) ) )
//! ```
//! and τ, η, |k|^(y−2), |k|^(y−1) precomputed in
//! [`super::init::initialize_absorption_operators`]. The correction is **algebraic**:
//! it is added once per step and **does not carry a Δt factor**, because the
//! fractional-Laplacian terms are part of the EOS rather than a time-integrated
//! source. This is the formulation already implemented by the GPU WGSL shader
//! (`kwavers/src/gpu/shaders/pstd.wgsl::absorb_pressure_correction`) and by
//! k-Wave MATLAB (`kspaceFirstOrder3D.m` lines following the EOS computation)
//! and k-wave-python (`kwave/solvers/kspace_solver.py:613`).
//!
//! ## Why divergence is recomputed locally
//! `update_density_cartesian` already computes the kappa-corrected per-axis
//! velocity divergences `∂u_α/∂α` and stores them in `self.dpx/dpy/dpz`, but
//! `apply_pressure_sources` (which runs between `update_density` and
//! `update_pressure`) clears `dpx` and reuses it as a source-injection scratch
//! (see `stepper/sources.rs::apply_pressure_sources`). By the time
//! `update_pressure` is reached, `dpx` no longer holds `∂u_x/∂x`, and `L1`
//! computed from a zeroed `dpx` collapses to zero — exactly the failure mode
//! observed empirically before this fix landed (max|L1| = 0 every step).
//!
//! Rather than introducing an extra persistent backup buffer, we recompute the
//! kappa-corrected divergence from `self.fields.u{x,y,z}` here. The cost is
//! one additional forward + inverse R2C FFT per active axis per step, on the
//! absorbing path only (`Lossless` returns immediately). The reused operators
//! (`ddx_k_shift_neg`, `kappa`) are identical to those in `update_density`,
//! so the divergence is bit-for-bit the same as the value originally written
//! into `dpx` before `apply_pressure_sources` clobbered it.
//!
//! ## References
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314, Eqs. 9–10, 19–21.
//! - Treeby, Jaros, Rendell, Cox (2012). J. Acoust. Soc. Am. 131(6), 4324, §II.D.
//! - k-Wave C++: `KSpaceFirstOrderSolver::computePressureTermsLinearPowerLaw` +
//!   `computePowerLawAbsorbtionTerm` + `sumPressureTermsLinear`.
//! - k-wave-python: `kspace_solver.py:613` — the canonical reference run by
//!   the parity scripts.

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::math::fft::Complex64;
use crate::physics::acoustics::mechanics::absorption::AbsorptionMode;
use crate::solver::pstd::PSTDSolver;
use ndarray::{s, Zip};

impl PSTDSolver {
    /// Apply pressure-side power-law absorption correction.
    ///
    /// Must be called inside `update_pressure` **after** the EOS step so that:
    ///   - `self.fields.p`   ← `c₀² · ρ_total`
    ///   - `self.div_u`      ← `ρ_total`
    ///   - `self.fields.u{x,y,z}` ← latest velocity field
    ///
    /// On return `self.fields.p` includes the absorption correction
    /// `c₀² · (τ·L1 − η·L2)`. Scratch buffers `dpx`, `dpy`, `dpz`, `grad_k`,
    /// `ux_k`, `uy_k`, `uz_k` are clobbered.
    ///
    /// ## References
    /// - Treeby & Cox (2010) Eqs. 19–21.
    /// - GPU shader equivalent: `pstd.wgsl::absorb_pressure_correction`.
    pub(crate) fn apply_absorption_to_pressure(&mut self) -> KwaversResult<()> {
        match self.config.absorption_mode {
            AbsorptionMode::Lossless => return Ok(()),
            AbsorptionMode::Stokes | AbsorptionMode::PowerLaw { .. } => {}
            AbsorptionMode::MultiRelaxation { .. } | AbsorptionMode::Causal { .. } => {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "Relaxation absorption modes are not supported by spectral solver"
                            .to_string(),
                    },
                ));
            }
        }

        let Some(ref abs) = self.absorption else {
            return Ok(());
        };

        let has_y = self.grid.ny > 1;
        let has_z = self.grid.nz > 1;
        // R2C output is half-spectrum along z: shape (nx, ny, nz_c=nz/2+1).
        let nz_c = self.grad_k.dim().2;

        // ── Step 1: recompute kappa-corrected per-axis divergences
        // ∂u_α/∂α = IFFT( iκ_α · exp(−iκ_α·dx/2) · κ(k) · FFT(u_α) ). Identical
        // operator to `update_density_cartesian`. We recompute here rather than
        // reuse `dpx`/`dpy`/`dpz` because `apply_pressure_sources` overwrites
        // `dpx` between the density update and this call.

        // X-axis: dpx ← ∂u_x/∂x.
        self.fft.forward_r2c_into(&self.fields.ux, &mut self.ux_k);
        {
            let ddx = self.ddx_k_shift_neg.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.ux_k.view())
                .and(self.kappa.slice(s![.., .., ..nz_c]))
                .for_each(|(i, _j, _k), gk, &u, &kap| {
                    *gk = ddx[i] * Complex64::new(kap, 0.0) * u;
                });
        }
        self.fft
            .inverse_c2r_into(&self.grad_k, &mut self.dpx, &mut self.ux_k);

        // Y-axis: dpy ← ∂u_y/∂y if has_y, else 0.
        if has_y {
            self.fft.forward_r2c_into(&self.fields.uy, &mut self.uy_k);
            let ddy = self.ddy_k_shift_neg.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.uy_k.view())
                .and(self.kappa.slice(s![.., .., ..nz_c]))
                .for_each(|(_i, j, _k), gk, &u, &kap| {
                    *gk = ddy[j] * Complex64::new(kap, 0.0) * u;
                });
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.dpy, &mut self.uy_k);
        } else {
            self.dpy.fill(0.0);
        }

        // Z-axis: dpz ← ∂u_z/∂z if has_z, else 0.
        if has_z {
            self.fft.forward_r2c_into(&self.fields.uz, &mut self.uz_k);
            let ddz = self.ddz_k_shift_neg.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.uz_k.view())
                .and(self.kappa.slice(s![.., .., ..nz_c]))
                .for_each(|(_i, _j, k), gk, &u, &kap| {
                    *gk = ddz[k] * Complex64::new(kap, 0.0) * u;
                });
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.dpz, &mut self.uz_k);
        } else {
            self.dpz.fill(0.0);
        }

        // ── Step 2: build ρ₀·∇·u into dpz (overwriting it).
        // *d on the LHS reads original dpz first, then assigns the sum.
        Zip::from(&mut self.dpz)
            .and(&self.dpx)
            .and(&self.dpy)
            .and(&self.materials.rho0)
            .for_each(|d, &x, &y, &r0| {
                *d = r0 * (x + y + *d);
            });

        // ── Step 3: L1 = IFFT( |k|^(y−2) · FFT(ρ₀·∇·u) ) → dpx (clobbered).
        self.fft.forward_r2c_into(&self.dpz, &mut self.grad_k);
        {
            let n1 = abs.nabla1.slice(s![.., .., ..nz_c]);
            Zip::from(&mut self.grad_k)
                .and(&n1)
                .for_each(|gk, &n| {
                    *gk *= Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_c2r_into(&self.grad_k, &mut self.dpx, &mut self.ux_k);
        // dpx now holds L1.

        // ── Step 4: L2 = IFFT( |k|^(y−1) · FFT(ρ_total) ) → dpy (clobbered).
        // div_u still holds ρ_total from the EOS step in update_pressure.
        self.fft.forward_r2c_into(&self.div_u, &mut self.grad_k);
        {
            let n2 = abs.nabla2.slice(s![.., .., ..nz_c]);
            Zip::from(&mut self.grad_k)
                .and(&n2)
                .for_each(|gk, &n| {
                    *gk *= Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_c2r_into(&self.grad_k, &mut self.dpy, &mut self.ux_k);
        // dpy now holds L2.

        // ── Step 5: p += c₀² · (τ · L1 − η · L2).
        Zip::from(&mut self.fields.p)
            .and(&self.materials.c0)
            .and(&abs.tau)
            .and(&abs.eta)
            .and(&self.dpx)
            .and(&self.dpy)
            .par_for_each(|p, &c, &t, &e, &l1, &l2| {
                *p += c * c * (t * l1 - e * l2);
            });

        Ok(())
    }
}

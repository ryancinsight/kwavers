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
//! ## Divergence cache: eliminating per-step FFT recomputation
//! `update_density_cartesian` computes ∂u_α/∂α and immediately writes each
//! axis result into `self.div_ux`/`div_uy`/`div_uz` (the divergence cache).
//! `apply_pressure_sources` then zeros `dpx` for source injection, but never
//! touches `div_ux`/`div_uy`/`div_uz`.  `apply_absorption_to_pressure` reads
//! the cache directly, replacing 3 forward + 3 inverse FFT calls with three
//! `Array3::assign` (memcpy) operations per step on the absorbing path.
//!
//! ## References
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314, Eqs. 9–10, 19–21.
//! - Treeby, Jaros, Rendell, Cox (2012). J. Acoust. Soc. Am. 131(6), 4324, §II.D.
//! - k-Wave C++: `KSpaceFirstOrderSolver::computePressureTermsLinearPowerLaw` +
//!   `computePowerLawAbsorbtionTerm` + `sumPressureTermsLinear`.
//! - k-wave-python: `kspace_solver.py:613` — the canonical reference run by
//!   the parity scripts.

use kwavers_math::fft::Fft3dInOutExt;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use crate::pstd::PSTDSolver;
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
    /// `c₀² · (τ·L1 − η·L2)`. Scratch buffers `dpx`, `dpy`, `grad_k`, and
    /// `ux_k` are clobbered (`dpz` removed in Opt-12; `uy_k`/`uz_k` in Opt-8).
    ///
    /// ## References
    /// - Treeby & Cox (2010) Eqs. 19–21.
    /// - GPU shader equivalent: `pstd.wgsl::absorb_pressure_correction`.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub(crate) fn apply_absorption_to_pressure(&mut self) -> KwaversResult<()> {
        match self.config.absorption_mode {
            AbsorptionMode::Lossless => return Ok(()),
            AbsorptionMode::Stokes | AbsorptionMode::PowerLaw { .. } => {}
            AbsorptionMode::MultiRelaxation { .. } | AbsorptionMode::Causal { .. } => {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "Relaxation absorption modes are not supported by spectral solver"
                            .to_owned(),
                    },
                ));
            }
        }

        let Some(ref abs) = self.absorption else {
            return Ok(());
        };

        // R2C output is half-spectrum along z: shape (nx, ny, nz_c=nz/2+1).
        let nz_c = self.grad_k.dim().2;

        // ── Step 1 (Opt-7 + Opt-12): build ρ₀·∇·u directly into dpx from div_u* cache.
        //
        // `update_density_cartesian` writes ∂u_α/∂α into `div_ux`/`div_uy`/`div_uz`
        // (the divergence cache).  `apply_pressure_sources` zeros `dpx` but never
        // touches `div_ux`/`div_uy`/`div_uz`, so the cache is always current here.
        //
        // Opt-7: eliminated 3 N-element assigns (div_u* → dpx/dpy/dpz) with a
        // single fused Zip writing the accumulator directly.
        // Opt-12: `dpz` is eliminated; `dpx` replaces it as the accumulator.
        //   - Safety: dpx was last written by the velocity z-axis gradient IFFT and
        //     its content was fully consumed by the uz Zip before this step runs.
        //   - dpx (Step 1 content) is consumed by the Step 3 FFT before the Step 3
        //     IFFT overwrites dpx with L1 — no aliasing.
        //
        // Saves: 3 × N-element memcpy (≈ 6.3 MB bandwidth at N=64³) per step (Opt-7)
        // + 1 × N×8 bytes of solver memory (Opt-12: dpz eliminated).
        Zip::from(&mut self.dpx)
            .and(&self.div_ux)
            .and(&self.div_uy)
            .and(&self.div_uz)
            .and(&self.materials.rho0)
            .par_for_each(|d, &x, &y, &z, &r0| {
                *d = r0 * (x + y + z);
            });

        // ── Step 3: L1 = IFFT( |k|^(y−2) · FFT(ρ₀·∇·u) ) → dpx (clobbered).
        // FFT reads dpx (Step 1 content); IFFT then safely overwrites dpx with L1.
        self.fft.forward_r2c_into(&self.dpx, &mut self.grad_k);
        {
            let n1 = abs.nabla1.slice(s![.., .., ..nz_c]);
            Zip::from(&mut self.grad_k).and(&n1).par_for_each(|gk, &n| {
                *gk *= n; // n: f64; nabla1 is real-valued
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
            Zip::from(&mut self.grad_k).and(&n2).par_for_each(|gk, &n| {
                *gk *= n; // n: f64; nabla2 is real-valued
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
                *p += c * c * t.mul_add(l1, -(e * l2));
            });

        Ok(())
    }
}

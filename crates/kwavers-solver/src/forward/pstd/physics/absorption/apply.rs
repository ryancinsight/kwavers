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

use crate::pstd::PSTDSolver;
use kwavers_core::error::KwaversResult;
use kwavers_math::fft::{Complex64, Fft3dInOutExt};
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use moirai_parallel::{enumerate_mut_with, Adaptive};
use ndarray::{s, Array3, ArrayView3};

#[inline]
fn dense_indices(index: usize, ny: usize, nz: usize) -> (usize, usize, usize) {
    let plane = ny * nz;
    let i = index / plane;
    let rem = index % plane;
    let j = rem / nz;
    let k = rem % nz;
    (i, j, k)
}

fn build_weighted_divergence(
    output: &mut Array3<f64>,
    div_x: &Array3<f64>,
    div_y: &Array3<f64>,
    div_z: &Array3<f64>,
    rho0: &Array3<f64>,
) {
    assert_eq!(
        output.shape(),
        div_x.shape(),
        "invariant: absorption weighted divergence shape matches div_x"
    );
    assert_eq!(
        output.shape(),
        div_y.shape(),
        "invariant: absorption weighted divergence shape matches div_y"
    );
    assert_eq!(
        output.shape(),
        div_z.shape(),
        "invariant: absorption weighted divergence shape matches div_z"
    );
    assert_eq!(
        output.shape(),
        rho0.shape(),
        "invariant: absorption weighted divergence shape matches rho0"
    );

    if let (Some(output_values), Some(x_values), Some(y_values), Some(z_values), Some(rho_values)) = (
        output.as_slice_memory_order_mut(),
        div_x.as_slice_memory_order(),
        div_y.as_slice_memory_order(),
        div_z.as_slice_memory_order(),
        rho0.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(output_values, |index, output| {
            *output = rho_values[index] * (x_values[index] + y_values[index] + z_values[index]);
        });
        return;
    }

    let (nx, ny, nz) = output.dim();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                output[[i, j, k]] =
                    rho0[[i, j, k]] * (div_x[[i, j, k]] + div_y[[i, j, k]] + div_z[[i, j, k]]);
            }
        }
    }
}

fn multiply_spectral_operator(spectrum: &mut Array3<Complex64>, operator: ArrayView3<'_, f64>) {
    assert_eq!(
        spectrum.shape(),
        operator.shape(),
        "invariant: absorption spectral field shape matches spectral operator"
    );

    let (_nx, ny, nz) = spectrum.dim();
    if let Some(spectrum_values) = spectrum.as_slice_memory_order_mut() {
        enumerate_mut_with::<Adaptive, _, _>(spectrum_values, |index, spectrum| {
            let (i, j, k) = dense_indices(index, ny, nz);
            *spectrum *= operator[[i, j, k]];
        });
        return;
    }

    let (nx, ny, nz) = spectrum.dim();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                spectrum[[i, j, k]] *= operator[[i, j, k]];
            }
        }
    }
}

fn accumulate_stratum(
    accumulator: &mut Array3<f64>,
    values: &Array3<f64>,
    bracket_lo: &Array3<u32>,
    weight_hi: &Array3<f64>,
    stratum: u32,
) {
    assert_eq!(
        accumulator.shape(),
        values.shape(),
        "invariant: absorption stratum accumulator shape matches values"
    );
    assert_eq!(
        accumulator.shape(),
        bracket_lo.shape(),
        "invariant: absorption stratum accumulator shape matches bracket indices"
    );
    assert_eq!(
        accumulator.shape(),
        weight_hi.shape(),
        "invariant: absorption stratum accumulator shape matches weights"
    );

    if let (Some(acc_values), Some(value_values), Some(lo_values), Some(weight_values)) = (
        accumulator.as_slice_memory_order_mut(),
        values.as_slice_memory_order(),
        bracket_lo.as_slice_memory_order(),
        weight_hi.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(acc_values, |index, accumulator| {
            let lower = lo_values[index];
            let weight = if lower == stratum {
                1.0 - weight_values[index]
            } else if lower + 1 == stratum {
                weight_values[index]
            } else {
                0.0
            };
            *accumulator += weight * value_values[index];
        });
        return;
    }

    let (nx, ny, nz) = accumulator.dim();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let lower = bracket_lo[[i, j, k]];
                let weight = if lower == stratum {
                    1.0 - weight_hi[[i, j, k]]
                } else if lower + 1 == stratum {
                    weight_hi[[i, j, k]]
                } else {
                    0.0
                };
                accumulator[[i, j, k]] += weight * values[[i, j, k]];
            }
        }
    }
}

fn apply_pressure_absorption(
    pressure: &mut Array3<f64>,
    c0: &Array3<f64>,
    tau: &Array3<f64>,
    eta: &Array3<f64>,
    l1: &Array3<f64>,
    l2: &Array3<f64>,
) {
    assert_eq!(
        pressure.shape(),
        c0.shape(),
        "invariant: absorption pressure shape matches sound speed"
    );
    assert_eq!(
        pressure.shape(),
        tau.shape(),
        "invariant: absorption pressure shape matches tau"
    );
    assert_eq!(
        pressure.shape(),
        eta.shape(),
        "invariant: absorption pressure shape matches eta"
    );
    assert_eq!(
        pressure.shape(),
        l1.shape(),
        "invariant: absorption pressure shape matches L1"
    );
    assert_eq!(
        pressure.shape(),
        l2.shape(),
        "invariant: absorption pressure shape matches L2"
    );

    if let (
        Some(pressure_values),
        Some(c0_values),
        Some(tau_values),
        Some(eta_values),
        Some(l1_values),
        Some(l2_values),
    ) = (
        pressure.as_slice_memory_order_mut(),
        c0.as_slice_memory_order(),
        tau.as_slice_memory_order(),
        eta.as_slice_memory_order(),
        l1.as_slice_memory_order(),
        l2.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(pressure_values, |index, pressure| {
            let c = c0_values[index];
            *pressure += c
                * c
                * tau_values[index]
                    .mul_add(l1_values[index], -(eta_values[index] * l2_values[index]));
        });
        return;
    }

    let (nx, ny, nz) = pressure.dim();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = c0[[i, j, k]];
                pressure[[i, j, k]] += c
                    * c
                    * tau[[i, j, k]].mul_add(l1[[i, j, k]], -(eta[[i, j, k]] * l2[[i, j, k]]));
            }
        }
    }
}

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
            // PowerLaw, Stokes, and the relaxation modes (MultiRelaxation, Causal)
            // are all realized through the fractional-Laplacian kernel built in
            // `initialize_absorption_operators`; the apply path is identical.
            AbsorptionMode::Stokes
            | AbsorptionMode::PowerLaw { .. }
            | AbsorptionMode::MultiRelaxation { .. }
            | AbsorptionMode::Causal { .. } => {}
        }

        let Some(ref abs) = self.absorption else {
            return Ok(());
        };

        // R2C output is half-spectrum along z: shape (nx, ny, nz_c=nz/2+1).
        let nz_c = self.grad_k.dim().2;

        if let Some(strata) = &abs.strata {
            // ── Stratified path: spatially-varying exponent y(x) (beyond k-Wave).
            // For each Laplacian, accumulate the per-stratum operator weighted by
            // the per-voxel bracket weights, reproducing each tissue's own power
            // law. One forward FFT is re-formed per stratum (no extra buffers);
            // strata exist only for genuinely heterogeneous-y media.
            let m_count = strata.exponents.len();

            // L1 = Σ_m w_m(x) · IFFT( |k|^(y_m−2) · FFT(ρ₀·∇·u) ) → dpx.
            self.dpx.fill(0.0);
            for m in 0..m_count {
                // Rebuild ρ₀·∇·u into dpy from the divergence cache (cheap, no FFT).
                build_weighted_divergence(
                    &mut self.dpy,
                    &self.div_ux,
                    &self.div_uy,
                    &self.div_uz,
                    &self.materials.rho0,
                );
                self.fft.forward_r2c_into(&self.dpy, &mut self.grad_k);
                multiply_spectral_operator(&mut self.grad_k, strata.nabla1[m].view());
                self.fft
                    .inverse_c2r_into(&self.grad_k, &mut self.dpy, &mut self.ux_k);
                accumulate_stratum(
                    &mut self.dpx,
                    &self.dpy,
                    &strata.bracket_lo,
                    &strata.weight_hi,
                    m as u32,
                );
            }

            // L2 = Σ_m w_m(x) · IFFT( |k|^(y_m−1) · FFT(ρ_total) ) → dpy.
            // div_u holds ρ_total; div_ux is reused as the per-stratum scratch
            // (the divergence cache is no longer needed this step).
            self.dpy.fill(0.0);
            for m in 0..m_count {
                self.fft.forward_r2c_into(&self.div_u, &mut self.grad_k);
                multiply_spectral_operator(&mut self.grad_k, strata.nabla2[m].view());
                self.fft
                    .inverse_c2r_into(&self.grad_k, &mut self.div_ux, &mut self.ux_k);
                accumulate_stratum(
                    &mut self.dpy,
                    &self.div_ux,
                    &strata.bracket_lo,
                    &strata.weight_hi,
                    m as u32,
                );
            }
        } else {
            // ── Uniform path (single global exponent, k-Wave-equivalent).
            //
            // Step 1 (Opt-7 + Opt-12): build ρ₀·∇·u directly into dpx from div_u*
            // cache. `update_density_cartesian` writes ∂u_α/∂α into
            // `div_ux`/`div_uy`/`div_uz`; `apply_pressure_sources` zeros `dpx` but
            // never touches the cache, so it is current here. dpx (Step 1 content)
            // is consumed by the Step 3 FFT before the IFFT overwrites dpx with L1.
            build_weighted_divergence(
                &mut self.dpx,
                &self.div_ux,
                &self.div_uy,
                &self.div_uz,
                &self.materials.rho0,
            );

            // Step 3: L1 = IFFT( |k|^(y−2) · FFT(ρ₀·∇·u) ) → dpx (clobbered).
            self.fft.forward_r2c_into(&self.dpx, &mut self.grad_k);
            {
                let n1 = abs.nabla1.slice(s![.., .., ..nz_c]);
                multiply_spectral_operator(&mut self.grad_k, n1);
            }
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.dpx, &mut self.ux_k);
            // dpx now holds L1.

            // Step 4: L2 = IFFT( |k|^(y−1) · FFT(ρ_total) ) → dpy (clobbered).
            // div_u still holds ρ_total from the EOS step in update_pressure.
            self.fft.forward_r2c_into(&self.div_u, &mut self.grad_k);
            {
                let n2 = abs.nabla2.slice(s![.., .., ..nz_c]);
                multiply_spectral_operator(&mut self.grad_k, n2);
            }
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.dpy, &mut self.ux_k);
            // dpy now holds L2.
        }

        // ── Step 5: p += c₀² · (τ · L1 − η · L2).
        apply_pressure_absorption(
            &mut self.fields.p,
            &self.materials.c0,
            &abs.tau,
            &abs.eta,
            &self.dpx,
            &self.dpy,
        );

        Ok(())
    }
}

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

use crate::forward::lanes::for_each_z_lane;
use crate::pstd::PSTDSolver;
use kwavers_core::error::KwaversResult;
use kwavers_math::fft::{Complex64, Fft3dInOutExt};
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use leto::Array3 as LetoArray3;
use leto::Array3;

fn build_weighted_divergence(
    output: &mut LetoArray3<f64>,
    div_x: &LetoArray3<f64>,
    div_y: &LetoArray3<f64>,
    div_z: &LetoArray3<f64>,
    rho0: &LetoArray3<f64>,
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

    let [_nx, ny, nz] = output.shape();
    if let (Some(output_values), Some(x_values), Some(y_values), Some(z_values), Some(rho_values)) = (
        output.as_slice_mut(),
        div_x.as_slice(),
        div_y.as_slice(),
        div_z.as_slice(),
        rho0.as_slice(),
    ) {
        for_each_z_lane(
            output_values,
            [ny, nz],
            5 * size_of::<f64>(),
            |start, _, _, lane| {
                let end = start + nz;
                let inputs = x_values[start..end]
                    .iter()
                    .zip(&y_values[start..end])
                    .zip(&z_values[start..end])
                    .zip(&rho_values[start..end]);
                for (output, (((&x, &y), &z), &rho)) in lane.iter_mut().zip(inputs) {
                    *output = rho * (x + y + z);
                }
            },
        );
        return;
    }

    let [nx, ny, nz] = output.shape();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                output[[i, j, k]] =
                    rho0[[i, j, k]] * (div_x[[i, j, k]] + div_y[[i, j, k]] + div_z[[i, j, k]]);
            }
        }
    }
}

fn multiply_spectral_operator(spectrum: &mut LetoArray3<Complex64>, operator: &LetoArray3<f64>) {
    assert_eq!(
        spectrum.shape(),
        operator.shape(),
        "invariant: absorption spectral field shape matches spectral operator"
    );

    let [_nx, ny, nz] = spectrum.shape();
    if let (Some(spectrum_values), Some(operator_values)) =
        (spectrum.as_slice_mut(), operator.as_slice())
    {
        let element_bytes = size_of::<Complex64>() + size_of::<f64>();
        for_each_z_lane(
            spectrum_values,
            [ny, nz],
            element_bytes,
            |start, _, _, lane| {
                for (value, &factor) in lane.iter_mut().zip(&operator_values[start..start + nz]) {
                    *value *= factor;
                }
            },
        );
        return;
    }

    let [nx, ny, nz] = spectrum.shape();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                spectrum[[i, j, k]] *= operator[[i, j, k]];
            }
        }
    }
}

fn accumulate_stratum(
    accumulator: &mut LetoArray3<f64>,
    values: &LetoArray3<f64>,
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

    let [_nx, ny, nz] = accumulator.shape();
    if let (Some(acc_values), Some(value_values), Some(lo_values), Some(weight_values)) = (
        accumulator.as_slice_mut(),
        values.as_slice(),
        bracket_lo.as_slice(),
        weight_hi.as_slice(),
    ) {
        let element_bytes = 3 * size_of::<f64>() + size_of::<u32>();
        for_each_z_lane(acc_values, [ny, nz], element_bytes, |start, _, _, lane| {
            let end = start + nz;
            let inputs = value_values[start..end]
                .iter()
                .zip(&lo_values[start..end])
                .zip(&weight_values[start..end]);
            for (accumulator, ((&value, &lower), &weight_hi)) in lane.iter_mut().zip(inputs) {
                let weight = if lower == stratum {
                    1.0 - weight_hi
                } else if lower + 1 == stratum {
                    weight_hi
                } else {
                    0.0
                };
                *accumulator += weight * value;
            }
        });
        return;
    }

    let [nx, ny, nz] = accumulator.shape();
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
    pressure: &mut LetoArray3<f64>,
    c0: &LetoArray3<f64>,
    tau: &Array3<f64>,
    eta: &Array3<f64>,
    l1: &LetoArray3<f64>,
    l2: &LetoArray3<f64>,
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

    let [_nx, ny, nz] = pressure.shape();
    if let (
        Some(pressure_values),
        Some(c0_values),
        Some(tau_values),
        Some(eta_values),
        Some(l1_values),
        Some(l2_values),
    ) = (
        pressure.as_slice_mut(),
        c0.as_slice(),
        tau.as_slice(),
        eta.as_slice(),
        l1.as_slice(),
        l2.as_slice(),
    ) {
        for_each_z_lane(
            pressure_values,
            [ny, nz],
            6 * size_of::<f64>(),
            |start, _, _, lane| {
                let end = start + nz;
                let inputs = c0_values[start..end]
                    .iter()
                    .zip(&tau_values[start..end])
                    .zip(&eta_values[start..end])
                    .zip(&l1_values[start..end])
                    .zip(&l2_values[start..end]);
                for (pressure, ((((&c, &tau), &eta), &l1), &l2)) in lane.iter_mut().zip(inputs) {
                    *pressure += c * c * tau.mul_add(l1, -(eta * l2));
                }
            },
        );
        return;
    }

    let shape = pressure.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
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
    /// - Returns [`crate::KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
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
                multiply_spectral_operator(&mut self.grad_k, &strata.nabla1[m]);
                self.fft.inverse_c2r_into(&mut self.grad_k, &mut self.dpy);
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
                multiply_spectral_operator(&mut self.grad_k, &strata.nabla2[m]);
                self.fft
                    .inverse_c2r_into(&mut self.grad_k, &mut self.div_ux);
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
            // Construction stores the operators on the half spectrum already.
            multiply_spectral_operator(&mut self.grad_k, &abs.nabla1);
            self.fft.inverse_c2r_into(&mut self.grad_k, &mut self.dpx);
            // dpx now holds L1.

            // Step 4: L2 = IFFT( |k|^(y−1) · FFT(ρ_total) ) → dpy (clobbered).
            // div_u still holds ρ_total from the EOS step in update_pressure.
            self.fft.forward_r2c_into(&self.div_u, &mut self.grad_k);
            multiply_spectral_operator(&mut self.grad_k, &abs.nabla2);
            self.fft.inverse_c2r_into(&mut self.grad_k, &mut self.dpy);
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

#[cfg(test)]
mod tests;

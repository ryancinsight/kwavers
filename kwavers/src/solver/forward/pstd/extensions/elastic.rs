//! PSTD elastic extension — spectral isotropic-elastic stress/velocity
//! kernels that turn the canonical [`super::super::PSTDSolver`] into a
//! μ-aware elastic stepper.
//!
//! # Theorem (acoustic-fluid limit)
//!
//! For Lamé parameters with `μ ≡ 0` everywhere on the grid, both spectral
//! kernels in this module reduce to the linear acoustic-fluid stress-velocity
//! formulation that the baseline PSTD stepper implements. Therefore enabling
//! [`PstdElasticPlugin`] with `μ = 0` is mathematically identical to running
//! [`super::super::PSTDSolver`] without the plugin.
//!
//! ## Proof
//!
//! Inspect [`apply_stress_update_in_place`]. The shear-stress pass writes
//!
//! ```text
//! σ̃ₐᵦ = dt · μ · (i·k_β · ṽ_α + i·k_α · ṽ_β)     (α ≠ β)
//! ```
//!
//! With `μ = 0` every shear component is identically zero, so the velocity
//! update in [`apply_velocity_update_in_place`] reads only the divergences
//! of the normal stresses
//!
//! ```text
//! σ̃ₐₐ = dt · (λ · ∇·ṽ + 2μ · i·k_α · ṽ_α)  →  dt · λ · ∇·ṽ        (μ = 0).
//! ```
//!
//! Substituting, each component of the velocity update becomes
//!
//! ```text
//! ṽ_α(t+dt) = ṽ_α(t) + (dt²/ρ) · i·k_α · (λ · ∇·ṽ),
//! ```
//!
//! which is the spectral form of `ρ ∂v/∂t = −∇p` with the linear acoustic
//! equation of state `p = −λ ∇·v ≡ −ρ c² ∇·v` integrated over one step. ∎
//!
//! # History
//!
//! The two `apply_*_in_place` functions previously lived as
//! `pub(crate)` methods on
//! [`crate::physics::acoustics::mechanics::elastic_wave::ElasticWave`] in
//! `solver/forward/elastic_wave.rs`. That file also carried an
//! `impl AcousticWaveModel for ElasticWave` whose `update_wave` hard-coded
//! `μ = 0` (`mu_scratch.fill(0.0)`) and therefore reduced to a duplicate
//! pseudospectral acoustic-fluid stepper, parallel to
//! [`super::super::PSTDSolver`]. Per the canonical solver matrix in
//! [`crate::solver::forward`], that duplicate has been deleted; the spectral
//! primitives — the genuinely useful piece — live here as a PSTD plugin.

use crate::physics::acoustics::mechanics::elastic_wave::{
    parameters::{StressUpdateParams, VelocityUpdateParams},
    spectral_fields::{SpectralStressFields, SpectralVelocityFields},
};
use ndarray::Zip;
use num_complex::Complex;

/// Inputs to a single spectral elastic stress update.
///
/// Type alias for the existing [`StressUpdateParams`] so the public PSTD
/// extension surface is self-documenting (`SpectralStressUpdateInputs`)
/// without forcing readers to navigate into the physics module to understand
/// the kernel's contract.
pub type SpectralStressUpdateInputs<'a> = StressUpdateParams<'a>;

/// Inputs to a single spectral elastic velocity update.
pub type SpectralVelocityUpdateInputs<'a> = VelocityUpdateParams<'a>;

/// Configuration knobs for [`PstdElasticPlugin`].
///
/// The plugin currently has no tunable knobs beyond the shape it inherits
/// from the host PSTD solver; this struct exists as the dedicated extension
/// surface so that future flags (e.g. anisotropic stiffness, viscoelastic
/// memory variables) land here rather than on the host config.
#[derive(Debug, Clone, Default)]
pub struct SpectralElasticConfig {
    /// Reserved — held to make `Default` non-empty if future flags require
    /// initialization. Behaviour is identical regardless of value today.
    pub _reserved: (),
}

/// PSTD elastic extension plugin.
///
/// A handle that callers construct once per simulation and consult inside
/// the PSTD step loop. Holds no per-step state today; the API is shaped so
/// that the planned `SolverType::ElasticPSTD` orchestrator (see the `[arch]`
/// entry in `backlog.md`) can attach scratch buffers later without breaking
/// callers.
#[derive(Debug, Default)]
pub struct PstdElasticPlugin {
    config: SpectralElasticConfig,
}

impl PstdElasticPlugin {
    /// Construct a new plugin from its config.
    #[must_use]
    pub fn new(config: SpectralElasticConfig) -> Self {
        Self { config }
    }

    /// Borrow the active config.
    #[must_use]
    pub fn config(&self) -> &SpectralElasticConfig {
        &self.config
    }

    /// Update the spectral stress tensor in place.
    ///
    /// # Mathematical specification
    ///
    /// Hooke's law for an isotropic elastic medium in the spectral domain,
    /// with zero initial stress (stress state is not persisted between
    /// steps; the host solver re-derives σ from velocity each step):
    ///
    /// ```text
    ///   σ̃ₓₓ = dt · (λ · div_v + 2μ · ikₓ ṽₓ)
    ///   σ̃ᵧᵧ = dt · (λ · div_v + 2μ · ikᵧ ṽᵧ)
    ///   σ̃ᵤᵤ = dt · (λ · div_v + 2μ · ikᵤ ṽᵤ)
    ///   σ̃ₓᵧ = dt · μ · (ikᵧ ṽₓ + ikₓ ṽᵧ)
    ///   σ̃ₓᵤ = dt · μ · (ikᵤ ṽₓ + ikₓ ṽᵤ)
    ///   σ̃ᵧᵤ = dt · μ · (ikᵤ ṽᵧ + ikᵧ ṽᵤ)
    /// ```
    ///
    /// Parallelised with `Zip::indexed` + `par_for_each`; no heap
    /// allocation per call. For `μ ≡ 0` the shear pass produces an
    /// all-zero output and LLVM eliminates it as dead code, recovering
    /// baseline PSTD performance — see the module-level theorem.
    ///
    /// # Panics
    /// - Panics if `kx`/`ky`/`kz` slices are not contiguous.
    pub fn apply_stress_update_in_place(
        &self,
        params: &SpectralStressUpdateInputs<'_>,
        out: &mut SpectralStressFields,
    ) {
        let c_dt = Complex::new(params.dt, 0.0);

        // ndarray 0.16 Zip supports at most 6 producers (index counts as 1).
        // Pass A — normal stresses: out = current + dt · (λ·∇·v + 2μ·∂vα/∂α).
        Zip::indexed(out.txx.view_mut())
            .and(out.tyy.view_mut())
            .and(out.tzz.view_mut())
            .par_for_each(|(i, j, k), o_txx, o_tyy, o_tzz| {
                let dkx = params.dkx_op[[i, 0, 0]];
                let dky = params.dky_op[[j, 0, 0]];
                let dkz = params.dkz_op[[k, 0, 0]];

                let vx = params.vx_fft[[i, j, k]];
                let vy = params.vy_fft[[i, j, k]];
                let vz = params.vz_fft[[i, j, k]];

                let lambda = params.lame_lambda[[i, j, k]];
                let mu = params.lame_mu[[i, j, k]];
                let div_v = dkx * vx + dky * vy + dkz * vz;

                *o_txx = params.txx_fft[[i, j, k]]
                    + c_dt * (lambda * div_v + 2.0 * mu * (dkx * vx));
                *o_tyy = params.tyy_fft[[i, j, k]]
                    + c_dt * (lambda * div_v + 2.0 * mu * (dky * vy));
                *o_tzz = params.tzz_fft[[i, j, k]]
                    + c_dt * (lambda * div_v + 2.0 * mu * (dkz * vz));
            });

        // Pass B — shear stresses: out = current + dt · μ · (∂vα/∂β + ∂vβ/∂α).
        Zip::indexed(out.txy.view_mut())
            .and(out.txz.view_mut())
            .and(out.tyz.view_mut())
            .par_for_each(|(i, j, k), o_txy, o_txz, o_tyz| {
                let dkx = params.dkx_op[[i, 0, 0]];
                let dky = params.dky_op[[j, 0, 0]];
                let dkz = params.dkz_op[[k, 0, 0]];

                let vx = params.vx_fft[[i, j, k]];
                let vy = params.vy_fft[[i, j, k]];
                let vz = params.vz_fft[[i, j, k]];
                let mu = params.lame_mu[[i, j, k]];

                *o_txy = params.txy_fft[[i, j, k]] + c_dt * mu * (dky * vx + dkx * vy);
                *o_txz = params.txz_fft[[i, j, k]] + c_dt * mu * (dkz * vx + dkx * vz);
                *o_tyz = params.tyz_fft[[i, j, k]] + c_dt * mu * (dkz * vy + dky * vz);
            });
    }

    /// Update the spectral velocity field in place.
    ///
    /// # Mathematical specification
    ///
    /// Newton's second law in the spectral domain with the full elastic
    /// stress tensor (acoustic-fluid limit recovers `−∇p`):
    ///
    /// ```text
    ///   ṽₓ(t+dt) = ṽₓ(t) + (dt/ρ) · (ikₓ σ̃ₓₓ + ikᵧ σ̃ₓᵧ + ikᵤ σ̃ₓᵤ)
    ///   ṽᵧ(t+dt) = ṽᵧ(t) + (dt/ρ) · (ikₓ σ̃ₓᵧ + ikᵧ σ̃ᵧᵧ + ikᵤ σ̃ᵧᵤ)
    ///   ṽᵤ(t+dt) = ṽᵤ(t) + (dt/ρ) · (ikₓ σ̃ₓᵤ + ikᵧ σ̃ᵧᵤ + ikᵤ σ̃ᵤᵤ)
    /// ```
    ///
    /// At points where `ρ ≤ 0` (invalid medium) the velocity is preserved
    /// rather than divided by zero — matches the conservative behaviour of
    /// the host PSTD solver's stability guards.
    ///
    /// # Panics
    /// - Panics if `kx`/`ky`/`kz` slices are not contiguous.
    pub fn apply_velocity_update_in_place(
        &self,
        params: &SpectralVelocityUpdateInputs<'_>,
        out: &mut SpectralVelocityFields,
    ) {
        Zip::indexed(out.vx.view_mut())
            .and(out.vy.view_mut())
            .and(out.vz.view_mut())
            .par_for_each(|(i, j, k), o_vx, o_vy, o_vz| {
                let rho = params.density[[i, j, k]];
                if rho <= 0.0 {
                    *o_vx = params.vx_fft[[i, j, k]];
                    *o_vy = params.vy_fft[[i, j, k]];
                    *o_vz = params.vz_fft[[i, j, k]];
                    return;
                }

                let dkx = params.dkx_op[[i, 0, 0]];
                let dky = params.dky_op[[j, 0, 0]];
                let dkz = params.dkz_op[[k, 0, 0]];
                let c_dt_rho = Complex::new(params.dt / rho, 0.0);

                let dtxx_dx = dkx * params.txx_fft[[i, j, k]];
                let dtxy_dy = dky * params.txy_fft[[i, j, k]];
                let dtxz_dz = dkz * params.txz_fft[[i, j, k]];

                let dtxy_dx = dkx * params.txy_fft[[i, j, k]];
                let dtyy_dy = dky * params.tyy_fft[[i, j, k]];
                let dtyz_dz = dkz * params.tyz_fft[[i, j, k]];

                let dtxz_dx = dkx * params.txz_fft[[i, j, k]];
                let dtyz_dy = dky * params.tyz_fft[[i, j, k]];
                let dtzz_dz = dkz * params.tzz_fft[[i, j, k]];

                *o_vx = params.vx_fft[[i, j, k]] + c_dt_rho * (dtxx_dx + dtxy_dy + dtxz_dz);
                *o_vy = params.vy_fft[[i, j, k]] + c_dt_rho * (dtxy_dx + dtyy_dy + dtyz_dz);
                *o_vz = params.vz_fft[[i, j, k]] + c_dt_rho * (dtxz_dx + dtyz_dy + dtzz_dz);
            });
    }
}

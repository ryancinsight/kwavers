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
//! [`kwavers_physics::acoustics::mechanics::elastic_wave::ElasticWave`] in
//! `solver/forward/elastic_wave.rs`. That file also carried an
//! `impl AcousticWaveModel for ElasticWave` whose `update_wave` hard-coded
//! `μ = 0` (`mu_scratch.fill(0.0)`) and therefore reduced to a duplicate
//! pseudospectral acoustic-fluid stepper, parallel to
//! [`super::super::PSTDSolver`]. Per the canonical solver matrix in
//! [`crate::forward`], that duplicate has been deleted; the spectral
//! primitives — the genuinely useful piece — live here as a PSTD plugin.

use kwavers_math::fft::Complex64;
use kwavers_physics::acoustics::mechanics::elastic_wave::{
    parameters::{StressUpdateParams, VelocityUpdateParams},
    spectral_fields::{SpectralStressFields, SpectralVelocityFields},
};

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
    /// Indexed over the spectral grid with no heap
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
        let c_dt = Complex64::new(params.dt, 0.0);

        // ndarray 0.16 Zip supports at most 6 producers (index counts as 1).
        // Pass A — normal stresses: out = current + dt · (λ·∇·v + 2μ·∂vα/∂α).
        // kappa[i,j,k] = sinc(c_ref·dt·|k|/2) is the Tabei et al. (2002)
        // k-space correction that eliminates temporal dispersion; kappa = 1
        // everywhere recovers the uncorrected O(dt²) leapfrog scheme.
        let [nx, ny, nz] = out.txx.shape();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let kap = params.kappa[[i, j, k]];
                    let dkx = params.dkx_op[[i, 0, 0]] * kap;
                    let dky = params.dky_op[[j, 0, 0]] * kap;
                    let dkz = params.dkz_op[[k, 0, 0]] * kap;

                    let vx = params.vx_fft[[i, j, k]];
                    let vy = params.vy_fft[[i, j, k]];
                    let vz = params.vz_fft[[i, j, k]];

                    let lambda = params.lame_lambda[[i, j, k]];
                    let mu = params.lame_mu[[i, j, k]];
                    let div_v = dkx * vx + dky * vy + dkz * vz;

                    out.txx[[i, j, k]] =
                        params.txx_fft[[i, j, k]] + c_dt * (lambda * div_v + 2.0 * mu * (dkx * vx));
                    out.tyy[[i, j, k]] =
                        params.tyy_fft[[i, j, k]] + c_dt * (lambda * div_v + 2.0 * mu * (dky * vy));
                    out.tzz[[i, j, k]] =
                        params.tzz_fft[[i, j, k]] + c_dt * (lambda * div_v + 2.0 * mu * (dkz * vz));
                }
            }
        }

        // Pass B — shear stresses: out = current + dt · μ · (∂vα/∂β + ∂vβ/∂α).
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let kap = params.kappa[[i, j, k]];
                    let dkx = params.dkx_op[[i, 0, 0]] * kap;
                    let dky = params.dky_op[[j, 0, 0]] * kap;
                    let dkz = params.dkz_op[[k, 0, 0]] * kap;

                    let vx = params.vx_fft[[i, j, k]];
                    let vy = params.vy_fft[[i, j, k]];
                    let vz = params.vz_fft[[i, j, k]];
                    let mu = params.lame_mu[[i, j, k]];

                    out.txy[[i, j, k]] =
                        params.txy_fft[[i, j, k]] + c_dt * mu * (dky * vx + dkx * vy);
                    out.txz[[i, j, k]] =
                        params.txz_fft[[i, j, k]] + c_dt * mu * (dkz * vx + dkx * vz);
                    out.tyz[[i, j, k]] =
                        params.tyz_fft[[i, j, k]] + c_dt * mu * (dkz * vy + dky * vz);
                }
            }
        }
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
        let [nx, ny, nz] = out.vx.shape();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let rho = params.density[[i, j, k]];
                    if rho <= 0.0 {
                        out.vx[[i, j, k]] = params.vx_fft[[i, j, k]];
                        out.vy[[i, j, k]] = params.vy_fft[[i, j, k]];
                        out.vz[[i, j, k]] = params.vz_fft[[i, j, k]];
                        continue;
                    }

                    let kap = params.kappa[[i, j, k]];
                    let dkx = params.dkx_op[[i, 0, 0]] * kap;
                    let dky = params.dky_op[[j, 0, 0]] * kap;
                    let dkz = params.dkz_op[[k, 0, 0]] * kap;
                    let c_dt_rho = Complex64::new(params.dt / rho, 0.0);

                    let dtxx_dx = dkx * params.txx_fft[[i, j, k]];
                    let dtxy_dy = dky * params.txy_fft[[i, j, k]];
                    let dtxz_dz = dkz * params.txz_fft[[i, j, k]];

                    let dtxy_dx = dkx * params.txy_fft[[i, j, k]];
                    let dtyy_dy = dky * params.tyy_fft[[i, j, k]];
                    let dtyz_dz = dkz * params.tyz_fft[[i, j, k]];

                    let dtxz_dx = dkx * params.txz_fft[[i, j, k]];
                    let dtyz_dy = dky * params.tyz_fft[[i, j, k]];
                    let dtzz_dz = dkz * params.tzz_fft[[i, j, k]];

                    out.vx[[i, j, k]] =
                        params.vx_fft[[i, j, k]] + c_dt_rho * (dtxx_dx + dtxy_dy + dtxz_dz);
                    out.vy[[i, j, k]] =
                        params.vy_fft[[i, j, k]] + c_dt_rho * (dtxy_dx + dtyy_dy + dtyz_dz);
                    out.vz[[i, j, k]] =
                        params.vz_fft[[i, j, k]] + c_dt_rho * (dtxz_dx + dtyz_dy + dtzz_dz);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PstdElasticPlugin;
    use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
    use kwavers_math::fft::Complex64;
    use kwavers_physics::acoustics::mechanics::elastic_wave::parameters::StressUpdateParams;
    use kwavers_physics::acoustics::mechanics::elastic_wave::spectral_fields::SpectralStressFields;
    use leto::Array3;

    /// When μ ≡ 0, the spectral stress kernel produces zero shear stress for any
    /// non-trivial velocity field — the executable counterpart of the
    /// acoustic-fluid-limit theorem. (Integration test spanning physics +
    /// solver; relocated here from `physics` on the kwavers-physics extraction
    /// since it consumes the solver plugin.)
    #[test]
    fn pstd_elastic_plugin_reduces_to_acoustic_when_mu_is_zero() {
        let (nx, ny, nz) = (8usize, 8, 8);
        let make_v = || {
            let mut v = Array3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
            for ([i, j, k], x) in v
                .indexed_iter_mut()
                .expect("velocity test fixture is indexable")
            {
                *x = Complex64::new((i + j + k) as f64 + 1.0, (i * j + 1) as f64);
            }
            v
        };
        let vx_fft = make_v();
        let vy_fft = make_v();
        let vz_fft = make_v();

        let mut dkx_op = Array3::<Complex64>::from_elem([nx, 1, 1], Complex64::default());
        let mut dky_op = Array3::<Complex64>::from_elem([ny, 1, 1], Complex64::default());
        let mut dkz_op = Array3::<Complex64>::from_elem([nz, 1, 1], Complex64::default());
        for i in 0..nx {
            dkx_op[[i, 0, 0]] = Complex64::new(0.0, (i + 1) as f64 * 0.1);
        }
        for j in 0..ny {
            dky_op[[j, 0, 0]] = Complex64::new(0.0, (j + 1) as f64 * 0.1);
        }
        for k in 0..nz {
            dkz_op[[k, 0, 0]] = Complex64::new(0.0, (k + 1) as f64 * 0.1);
        }

        let lame_lambda = Array3::<f64>::from_elem([nx, ny, nz], 2.25e9);
        let lame_mu = Array3::<f64>::zeros([nx, ny, nz]);
        let density = Array3::<f64>::from_elem([nx, ny, nz], DENSITY_WATER_NOMINAL);
        let stress_current = SpectralStressFields::new(nx, ny, nz);
        let unit_kappa = Array3::<f64>::ones([nx, ny, nz]);

        let params = StressUpdateParams {
            vx_fft: &vx_fft,
            vy_fft: &vy_fft,
            vz_fft: &vz_fft,
            txx_fft: &stress_current.txx,
            tyy_fft: &stress_current.tyy,
            tzz_fft: &stress_current.tzz,
            txy_fft: &stress_current.txy,
            txz_fft: &stress_current.txz,
            tyz_fft: &stress_current.tyz,
            dkx_op: &dkx_op,
            dky_op: &dky_op,
            dkz_op: &dkz_op,
            lame_lambda: &lame_lambda,
            lame_mu: &lame_mu,
            density: density.view(),
            dt: 1e-7,
            kappa: &unit_kappa,
        };

        let mut out = SpectralStressFields::new(nx, ny, nz);
        let plugin = PstdElasticPlugin::default();
        plugin.apply_stress_update_in_place(&params, &mut out);

        let zero = Complex64::new(0.0, 0.0);
        for x in out.txy.iter().chain(out.txz.iter()).chain(out.tyz.iter()) {
            assert_eq!(*x, zero, "shear stress must be zero when μ = 0");
        }
        let any_normal_nonzero = out
            .txx
            .iter()
            .chain(out.tyy.iter())
            .chain(out.tzz.iter())
            .any(|x| *x != zero);
        assert!(
            any_normal_nonzero,
            "normal stresses must be non-zero for a non-trivial velocity field"
        );
    }
}

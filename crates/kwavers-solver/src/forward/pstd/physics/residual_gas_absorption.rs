//! Broadband residual-gas (bubble-cloud) absorption operator for `PSTDSolver`.
//!
//! ## Why this exists
//! A residual microbubble cloud attenuates differently at every frequency: the
//! Commander–Prosperetti (1989) attenuation `α_cp(ω)` is a *resonant* spectrum
//! (a peak near the Minnaert frequency, broadband tails), **not** a power law.
//! Folding it into the medium's power-law prefactor `α₀·f^y` (see
//! `multiphysics::residual_gas_coupling::apply_residual_gas_shielding_in_place`)
//! is exact only at a single frequency; for genuinely broadband content
//! (subharmonic f/2, ultraharmonics, inertial broadband) the `f^y`
//! extrapolation is wrong off that frequency.
//!
//! ## Theorem: separable spectral absorption generalises the fractional Laplacian
//! For a monodisperse residual cloud at a representative radius `R`, the
//! attenuation separates exactly into a spatial magnitude and a global spectral
//! shape:
//! ```text
//!   α_cp(ω, x) = m(x) · ĝ(ω),   ĝ(ω) = α_cp(ω) / α_cp(ω_drive),
//!   m(x)       = α_cp(ω_drive, β(x))     [Np/m, exact in the local void β(x)]
//! ```
//! (the resonant *shape* `ĝ` is set by `R` and is void-fraction independent in
//! the dilute regime CP is derived in; only the *magnitude* scales with `β`).
//! A wave component at wavenumber `k` has angular frequency `ω = c·|k|` and
//! travels `c·dt` per step, so its amplitude decays by `α_cp(ω,x)·c·dt`:
//! ```text
//!   p(x) ← p(x) − dt · c₀(x) · m(x) · IFFT( ĝ(c·|k|) · FFT(p) )(x)
//! ```
//! The spectral factor `ĝ(c·|k|)` is applied in k-space (exact per frequency);
//! the spatial magnitude `m(x)` is applied after the inverse transform — the
//! same structure as Treeby & Cox's `τ·IFFT(|k|^p·FFT(·))`. The power-law
//! operator is the special case `ĝ(ω)=α₀ω^y ⇒ ĝ(c|k|)∝|k|^y`; here `ĝ` is the
//! *true* CP spectrum, so the response is correct at **every** frequency, not
//! just the drive. For a uniform host this reproduces `exp(−α_cp(ω)·L)` over a
//! path `L` exactly (first order in `dt`, matching the leapfrog scheme order).
//!
//! ## Dispersion (frequency-dependent phase velocity)
//! The real part of the CP complex wavenumber gives a frequency-dependent phase
//! velocity `c_p(ω) = ω/Re(k_m)`. This operator carries it as an algebraic EOS
//! stiffness correction on the total density (the structure of Treeby & Cox's
//! `η·L2` dispersion term):
//! ```text
//!   p(x) ← p(x) + c₀² · s(x) · IFFT( ĥ(c·|k|) · FFT(ρ_total) )(x)
//! ```
//! with `ĥ(ω) = (c_p(ω)² − c₀²)/c₀²` (the per-`β_ref` dimensionless stiffness
//! deviation) and `s(x) = β(x)/β_ref`. Combined with the base EOS
//! `p = c₀²·ρ_total` this yields `p = c_p(ω)²·ρ_total` — the exact dispersive
//! constitutive relation at every frequency. Its `ω→0` limit reproduces the Wood
//! mixture sound speed (to first order in `β`), so the operator subsumes the Wood
//! collapse: the **complete, approximation-free** configuration runs on the
//! *unmodified host medium* (`c₀ = c_liquid`) with this operator supplying both
//! the loss and the full dispersive phase velocity. (Do not also bake Wood into
//! the medium in that configuration — the dispersion term already includes it.)
//!
//! ## References
//! - Commander & Prosperetti (1989). J. Acoust. Soc. Am. 85(2), 732–746.
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314, Eqs. 9–10, 19–21.

use crate::multiphysics::residual_gas_coupling::BubblyMediumProps;
use crate::pstd::PSTDSolver;
use kwavers_core::error::KwaversResult;
use kwavers_math::fft::Fft3dInOutExt;
use kwavers_physics::acoustics::bubble_dynamics::{
    commander_prosperetti_attenuation, commander_prosperetti_phase_velocity,
};
use ndarray::{Array3, ArrayView3, Zip};
use std::f64::consts::TAU;

/// Reference void fraction used to evaluate the (β-independent) spectral *shape*
/// ratio. It cancels in `α_cp(ω)/α_cp(ω_drive)`, so any dilute value works.
const SHAPE_REFERENCE_VOID_FRACTION: f64 = 1.0e-4;

/// True broadband bubble-cloud attenuation spectral shape `ĝ(f)` at the
/// representative radius: the Commander–Prosperetti attenuation at frequency
/// `freq_hz` normalised by its value at the drive frequency.
///
/// Returns `0.0` for non-positive frequencies (the DC bin carries no
/// attenuation). This is the genuine resonant spectrum — it peaks near the
/// Minnaert resonance and is **not** a power law of frequency.
#[must_use]
pub fn cp_spectral_shape(
    freq_hz: f64,
    drive_freq_hz: f64,
    c_liquid: f64,
    rho_liquid: f64,
    props: &BubblyMediumProps,
) -> f64 {
    if !freq_hz.is_finite() || freq_hz <= 0.0 {
        return 0.0;
    }
    let alpha = |f: f64| {
        commander_prosperetti_attenuation(
            f,
            SHAPE_REFERENCE_VOID_FRACTION,
            props.bubble_radius,
            c_liquid,
            rho_liquid,
            props.mu_liquid,
            props.p0,
            props.polytropic,
        )
    };
    let denom = alpha(drive_freq_hz);
    if denom > 0.0 {
        alpha(freq_hz) / denom
    } else {
        0.0
    }
}

/// Dimensionless bubble-cloud dispersion stiffness deviation at the reference
/// void fraction: `ĥ(f) = (c_p(f)² − c_liquid²)/c_liquid²`, evaluated at
/// [`SHAPE_REFERENCE_VOID_FRACTION`]. Negative below resonance (slowdown, the
/// Wood regime at low `f`), positive above resonance (anomalous dispersion).
/// Returns `0.0` for non-positive frequencies (the DC bin carries no dispersion).
#[must_use]
pub fn cp_dispersion_stiffness(
    freq_hz: f64,
    c_liquid: f64,
    rho_liquid: f64,
    props: &BubblyMediumProps,
) -> f64 {
    if !freq_hz.is_finite() || freq_hz <= 0.0 || c_liquid <= 0.0 || c_liquid.is_nan() {
        return 0.0;
    }
    let cp = commander_prosperetti_phase_velocity(
        freq_hz,
        SHAPE_REFERENCE_VOID_FRACTION,
        props.bubble_radius,
        c_liquid,
        rho_liquid,
        props.mu_liquid,
        props.p0,
        props.polytropic,
    );
    (cp * cp - c_liquid * c_liquid) / (c_liquid * c_liquid)
}

/// Precomputed broadband residual-gas absorption + dispersion operator.
///
/// Allocated only when a residual gas cloud is present; `None` otherwise.
/// Carries both the frequency-dependent amplitude loss and the dispersive phase
/// velocity (whose `ω→0` limit reproduces the Wood collapse).
pub(crate) struct ResidualGasAbsorption {
    /// Per-voxel loss magnitude `m(x)` [Np/m]: the true Commander–Prosperetti
    /// attenuation at the drive frequency for the local void fraction `β(x)`
    /// (exact in `β`). Shape `(nx, ny, nz)`.
    magnitude: Array3<f64>,
    /// Global loss spectral shape `ĝ(|k|) = α_cp(c·|k|)/α_cp(ω_drive)` in r2c
    /// half-spectrum k-space order `(nx, ny, nz_c)`. Dimensionless; carries the
    /// full resonant CP spectrum (no power-law extrapolation).
    shape_k: Array3<f64>,
    /// Per-voxel dispersion scale `s(x) = β(x)/β_ref` [-]. Shape `(nx, ny, nz)`.
    disp_scale: Array3<f64>,
    /// Global dispersion stiffness shape `ĥ(|k|) = (c_p(c·|k|)²−c₀²)/c₀²` at the
    /// reference void fraction, r2c half-spectrum order `(nx, ny, nz_c)`.
    disp_shape_k: Array3<f64>,
}

impl ResidualGasAbsorption {
    /// Build the operator from the raw half-spectrum wavenumber magnitude
    /// `k_mag_half`, the void-fraction field, and the host-liquid properties.
    ///
    /// `c_liquid` maps wavenumber to frequency (`ω = c_liquid·|k|`) and feeds the
    /// CP model as the host sound speed; `rho_liquid` is the host density.
    /// Returns `None` when there is no gas anywhere or the drive-frequency
    /// attenuation is degenerate (then the caller leaves shielding off).
    pub(crate) fn build(
        k_mag_half: &Array3<f64>,
        void_fraction: ArrayView3<'_, f64>,
        c_liquid: f64,
        rho_liquid: f64,
        props: &BubblyMediumProps,
    ) -> Option<Self> {
        let magnitude = void_fraction.mapv(|beta| {
            if beta > 0.0 {
                commander_prosperetti_attenuation(
                    props.frequency,
                    beta,
                    props.bubble_radius,
                    c_liquid,
                    rho_liquid,
                    props.mu_liquid,
                    props.p0,
                    props.polytropic,
                )
            } else {
                0.0
            }
        });
        if !magnitude.iter().any(|&m| m > 0.0) {
            return None;
        }

        // ĝ(|k|) = α_cp(f)/α_cp(f_drive) with f = c_liquid·|k| / 2π.
        let shape_k = k_mag_half.mapv(|k| {
            let freq = c_liquid * k / TAU;
            cp_spectral_shape(freq, props.frequency, c_liquid, rho_liquid, props)
        });

        // Dispersion: ĥ(|k|) = (c_p(f)²−c₀²)/c₀² at β_ref; scale s(x)=β(x)/β_ref.
        let disp_scale = void_fraction.mapv(|beta| beta.max(0.0) / SHAPE_REFERENCE_VOID_FRACTION);
        let disp_shape_k = k_mag_half.mapv(|k| {
            let freq = c_liquid * k / TAU;
            cp_dispersion_stiffness(freq, c_liquid, rho_liquid, props)
        });

        Some(Self {
            magnitude,
            shape_k,
            disp_scale,
            disp_shape_k,
        })
    }
}

impl PSTDSolver {
    /// Install (or refresh) the broadband residual-gas absorption operator from a
    /// void-fraction field. Call between pulses; the operator is then applied
    /// every step inside [`Self::update_pressure`]. Returns `true` when gas was
    /// present and the operator was installed, `false` when it was cleared.
    ///
    /// `c_liquid`/`rho_liquid` are the host-liquid sound speed and density of the
    /// residual-gas region (used both for the `ω = c·|k|` map and the CP model).
    pub fn set_residual_gas_absorption(
        &mut self,
        void_fraction: ArrayView3<'_, f64>,
        c_liquid: f64,
        rho_liquid: f64,
        props: &BubblyMediumProps,
    ) -> bool {
        match ResidualGasAbsorption::build(
            &self.k_mag_half,
            void_fraction,
            c_liquid,
            rho_liquid,
            props,
        ) {
            Some(op) => {
                self.residual_gas_absorption = Some(op);
                true
            }
            None => {
                self.residual_gas_absorption = None;
                false
            }
        }
    }

    /// Remove any installed residual-gas absorption operator.
    pub fn clear_residual_gas_absorption(&mut self) {
        self.residual_gas_absorption = None;
    }

    /// Apply the broadband residual-gas correction to the pressure field:
    /// the frequency-dependent attenuation
    /// `p ← p − dt·c₀·m(x)·IFFT(ĝ(|k|)·FFT(p))` followed by the dispersive EOS
    /// stiffness term `p ← p + c₀²·s(x)·IFFT(ĥ(|k|)·FFT(ρ_total))`. No-op when no
    /// operator is installed. Independent of `absorption_mode`, so it shields even
    /// an otherwise-lossless simulation.
    ///
    /// Reads `div_u` as `ρ_total` (the EOS step leaves it there, as the power-law
    /// absorption pass relies on). Clobbers the `grad_k`, `dpx`, and `ux_k` scratch
    /// buffers (free at this point in the step).
    /// # Errors
    /// Propagates FFT transform failures.
    pub(crate) fn apply_residual_gas_absorption(&mut self) -> KwaversResult<()> {
        let Some(ref op) = self.residual_gas_absorption else {
            return Ok(());
        };
        let dt = self.config.dt;

        // ── Attenuation: grad_k ← FFT(p); grad_k *= ĝ(|k|); dpx ← IFFT(grad_k).
        self.fft.forward_r2c_into(&self.fields.p, &mut self.grad_k);
        Zip::from(&mut self.grad_k)
            .and(&op.shape_k)
            .par_for_each(|gk, &s| {
                *gk *= s; // s: f64 spectral shape; grad_k is the half-spectrum
            });
        self.fft
            .inverse_c2r_into(&self.grad_k, &mut self.dpx, &mut self.ux_k);
        // p -= dt · c₀ · m(x) · IFFT(ĝ·FFT(p)).
        Zip::from(&mut self.fields.p)
            .and(&self.materials.c0)
            .and(&op.magnitude)
            .and(&self.dpx)
            .par_for_each(|p, &c0, &m, &loss| {
                *p -= dt * c0 * m * loss;
            });

        // ── Dispersion: grad_k ← FFT(ρ_total); grad_k *= ĥ(|k|); dpx ← IFFT.
        // ρ_total is held in div_u by the EOS step (same source the power-law
        // dispersion term reads).
        self.fft.forward_r2c_into(&self.div_u, &mut self.grad_k);
        Zip::from(&mut self.grad_k)
            .and(&op.disp_shape_k)
            .par_for_each(|gk, &h| {
                *gk *= h; // h: f64 stiffness deviation at β_ref
            });
        self.fft
            .inverse_c2r_into(&self.grad_k, &mut self.dpx, &mut self.ux_k);
        // p += c₀² · s(x) · IFFT(ĥ·FFT(ρ_total))  ⇒  effective p = c_p(ω)²·ρ_total.
        Zip::from(&mut self.fields.p)
            .and(&self.materials.c0)
            .and(&op.disp_scale)
            .and(&self.dpx)
            .par_for_each(|p, &c0, &s, &disp| {
                *p += c0 * c0 * s * disp;
            });
        Ok(())
    }
}

#[cfg(test)]
mod tests;

//! Heterogeneous power-law absorption for the FDTD solver, by relaxation
//! memory variables.
//!
//! # Why memory variables here
//!
//! The other two absorbing paths in kwavers are spectral. PSTD applies the
//! Treeby–Cox fractional Laplacian, whose symbol `|k|^(y−s)` lives in `k`-space
//! and needs an FFT pair per Laplacian; the viscoacoustic solver carries
//! relaxation memory variables but takes its derivatives pseudospectrally. An
//! FDTD path has no transform at all, so the fractional operator is unavailable
//! to it — but memory variables are **purely local in space**, which is exactly
//! what a finite-difference stencil wants, and is why Fullwave 2.5 (an FDTD
//! code) models attenuation this way.
//!
//! # Model
//!
//! For `L` Maxwell arms over an equilibrium modulus `M_∞`, driven by the
//! velocity divergence `D = ∇·v`,
//!
//! ```text
//!   ∂σₗ/∂t = −σₗ/τₗ − ΔMₗ·D
//!   ∂p/∂t  = −M_U·D − Σₗ σₗ/τₗ            M_U = M_∞ + Σₗ ΔMₗ
//! ```
//!
//! Each arm is advanced by the **exact** exponential integrator over the step
//! (`σ ← e^{-Δt/τ}σ − ΔMτ(1−e^{-Δt/τ})D`), so `Δt` is bounded by the wave CFL
//! and never by the smallest `τ`. The arm's contribution to the pressure update
//! is trapezoidal in `σ`.
//!
//! Both the arm strengths `ΔMₗ(x)` and the equilibrium modulus `M_∞(x)` come
//! from [`kwavers_medium::absorption::relaxation_fit`], which fits a shared
//! relaxation-time grid to the medium's own `α₀(x)` and `γ(x)` — so the
//! **exponent varies per voxel**, not merely the coefficient. The times are
//! shared across the domain because this component holds one memory field per
//! arm for the whole grid.
//!
//! # What this replaces in the pressure update
//!
//! The lossless update uses `ρ₀c₀²`. With relaxation the coefficient becomes
//! the **unrelaxed** modulus `M_U`, which is stiffer, and the relaxation sum is
//! subtracted alongside it. [`RelaxationAbsorption::unrelaxed_modulus`] is what
//! the pressure update must multiply the divergence by; passing `ρ₀c₀²` instead
//! would run the medium at the wrong (relaxed) sound speed.
//!
//! # References
//!
//! - Pinton, G.F. et al. (2009). "A heterogeneous nonlinear attenuating
//!   full-wave model of ultrasound." *IEEE Trans. UFFC* 56(3), 474–488.
//! - Blanch, J.O., Robertsson, J.O.A. & Symes, W.W. (1995). "Modeling of a
//!   constant Q." *Geophysics* 60(1), 176–184.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_medium::absorption::{fit_power_law_fields, FitBand, RelaxationTimePlacement};
use kwavers_medium::material_fields::MaterialFields;
use kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_m;
use leto::{Array3, ArrayView3};

use super::config::FdtdAbsorption;

#[cfg(test)]
mod tests;

/// One relaxation arm's precomputed per-voxel exponential-integrator
/// coefficients.
#[derive(Debug, Clone)]
struct Arm {
    /// `e^{-Δt/τₗ(x)}` — decay over one step.
    decay: Array3<f64>,
    /// `−ΔMₗ(x)·τₗ(x)·(1 − e^{-Δt/τₗ})` — coefficient of `D` in the σ update.
    gain: Array3<f64>,
    /// `1/τₗ(x)` \[s⁻¹] — for the trapezoidal pressure contribution.
    inv_tau: Array3<f64>,
    /// Memory field `σₗ(x)` \[Pa].
    sigma: Array3<f64>,
}

/// Relaxation-based power-law absorption state for one FDTD grid.
#[derive(Debug, Clone)]
pub struct RelaxationAbsorption {
    /// Unrelaxed modulus `M_U(x) = M_∞(x) + Σₗ ΔMₗ(x)` \[Pa].
    unrelaxed_modulus: Array3<f64>,
    arms: Vec<Arm>,
    /// Accumulator for `Σₗ ½(σₗ + σₗ_new)/τₗ`, reused every step.
    relaxation_sum: Array3<f64>,
    /// Worst per-voxel relative error of the fitted `α(f)` over the fit band.
    fit_error: f64,
    /// Shared relaxation times \[s].
    relaxation_times: Vec<f64>,
}

impl RelaxationAbsorption {
    /// Fit a relaxation spectrum to the medium's power law and allocate the
    /// memory fields.
    ///
    /// `alpha0_db` is the medium's k-Wave prefactor in `dB/(MHz^y·cm)`; it is
    /// converted to `Np·m⁻¹` at the configured reference frequency, which is
    /// the form the fit is posed in.
    ///
    /// # Errors
    /// - Non-positive density or sound speed anywhere on the grid.
    /// - A fit band the fitter rejects, or a voxel whose power law it cannot
    ///   represent.
    pub fn new(
        materials: &MaterialFields,
        settings: &PowerLawRelaxationSettings,
    ) -> KwaversResult<Self> {
        let shape = materials.rho0.shape();
        if materials.rho0.iter().any(|&r| !(r > 0.0)) || materials.c0.iter().any(|&c| !(c > 0.0)) {
            return Err(KwaversError::InvalidInput(
                "FDTD absorption requires positive density and sound speed everywhere".to_owned(),
            ));
        }

        // Convert the medium's prefactor to Np/m at the reference frequency,
        // per voxel and with that voxel's own exponent.
        let mut alpha_np_m = Array3::<f64>::zeros(shape);
        for (dst, (&alpha_db, &gamma)) in alpha_np_m
            .iter_mut()
            .zip(materials.alpha0_db.iter().zip(materials.alpha_power.iter()))
        {
            *dst =
                power_law_db_cm_to_np_m(alpha_db, gamma, settings.reference_frequency_hz).max(0.0);
        }

        let mut band = FitBand::new(
            settings.band_min_hz,
            settings.band_max_hz,
            settings.relaxation_arms,
        )?;
        band.placement = RelaxationTimePlacement::Optimized;

        let fit = fit_power_law_fields(
            &alpha_np_m,
            &materials.alpha_power,
            &materials.c0,
            &materials.rho0,
            settings.reference_frequency_hz,
            &band,
        )?;

        let dt = settings.dt;
        let mut unrelaxed_modulus = fit.equilibrium_modulus().clone();
        let mut arms = Vec::with_capacity(fit.weights().len());
        for (delta_m, &tau) in fit.weights().iter().zip(fit.relaxation_times()) {
            let decay = Array3::from_elem(shape, (-dt / tau).exp());
            let inv_tau = Array3::from_elem(shape, 1.0 / tau);
            let one_minus_decay = 1.0 - (-dt / tau).exp();
            let gain = delta_m.mapv(|dm| -dm * tau * one_minus_decay);
            for (modulus, &dm) in unrelaxed_modulus.iter_mut().zip(delta_m.iter()) {
                *modulus += dm;
            }
            arms.push(Arm {
                decay,
                gain,
                inv_tau,
                sigma: Array3::zeros(shape),
            });
        }

        Ok(Self {
            unrelaxed_modulus,
            arms,
            relaxation_sum: Array3::zeros(shape),
            fit_error: fit.max_relative_error(),
            relaxation_times: fit.relaxation_times().to_vec(),
        })
    }

    /// The coefficient the pressure update multiplies `∇·v` by \[Pa].
    ///
    /// This is `M_U`, **not** `ρ₀c₀²`: the relaxed modulus would propagate the
    /// medium at its low-frequency speed and double-count the dispersion the
    /// arms already supply.
    #[must_use]
    pub fn unrelaxed_modulus(&self) -> &Array3<f64> {
        &self.unrelaxed_modulus
    }

    /// Worst relative error of the fitted `α(f)` against the prescribed power
    /// law, over the fit band and the whole grid.
    #[must_use]
    pub fn fit_error(&self) -> f64 {
        self.fit_error
    }

    /// Shared relaxation times \[s].
    #[must_use]
    pub fn relaxation_times(&self) -> &[f64] {
        &self.relaxation_times
    }

    /// Number of memory fields carried per voxel.
    #[must_use]
    pub fn arm_count(&self) -> usize {
        self.arms.len()
    }

    /// Advance every memory field with this step's divergence and return the
    /// `(unrelaxed modulus, relaxation term)` pair the pressure update needs.
    ///
    /// Both are returned together from one reborrow because the caller needs
    /// them simultaneously and the accumulation is `&mut self`.
    ///
    /// `divergence` must be the **same** `∇·v` the pressure update consumes,
    /// after any CPML correction — the arms and the pressure would otherwise
    /// integrate different fields and drift apart inside the layer.
    pub fn accumulate(&mut self, divergence: ArrayView3<'_, f64>) -> (&Array3<f64>, &Array3<f64>) {
        debug_assert_eq!(
            divergence.shape(),
            self.relaxation_sum.shape(),
            "FDTD relaxation divergence shape must match the grid"
        );
        self.advance(divergence);
        (&self.unrelaxed_modulus, &self.relaxation_sum)
    }

    /// Advance the memory fields and fill the relaxation accumulator.
    fn advance(&mut self, divergence: ArrayView3<'_, f64>) {
        self.relaxation_sum.fill(0.0);

        let Some(divergence_values) = divergence.as_slice() else {
            // Non-contiguous view: fall back to indexed access rather than
            // silently skipping absorption.
            self.accumulate_indexed(divergence);
            return;
        };
        let sum_values = self
            .relaxation_sum
            .as_slice_mut()
            .expect("invariant: FDTD relaxation accumulator is contiguous");

        for arm in &mut self.arms {
            let (Some(sigma), Some(decay), Some(gain), Some(inv_tau)) = (
                arm.sigma.as_slice_mut(),
                arm.decay.as_slice(),
                arm.gain.as_slice(),
                arm.inv_tau.as_slice(),
            ) else {
                unreachable!("invariant: FDTD relaxation arm fields are contiguous")
            };
            for (index, sum_value) in sum_values.iter_mut().enumerate() {
                let old = sigma[index];
                let new = decay[index].mul_add(old, gain[index] * divergence_values[index]);
                *sum_value += 0.5 * (old + new) * inv_tau[index];
                sigma[index] = new;
            }
        }
    }

    /// Indexed fallback for a non-contiguous divergence view.
    fn accumulate_indexed(&mut self, divergence: ArrayView3<'_, f64>) {
        let [nx, ny, nz] = self.relaxation_sum.shape();
        for arm in &mut self.arms {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let index = [i, j, k];
                        let old = arm.sigma[index];
                        let new =
                            arm.decay[index].mul_add(old, arm.gain[index] * divergence[index]);
                        self.relaxation_sum[index] += 0.5 * (old + new) * arm.inv_tau[index];
                        arm.sigma[index] = new;
                    }
                }
            }
        }
    }

    /// Zero every memory field — used when a simulation restarts from a fresh
    /// initial condition, so absorption history does not leak across runs.
    pub fn reset(&mut self) {
        for arm in &mut self.arms {
            arm.sigma.fill(0.0);
        }
        self.relaxation_sum.fill(0.0);
    }
}

/// Resolved settings for the power-law relaxation path.
#[derive(Debug, Clone, Copy)]
pub struct PowerLawRelaxationSettings {
    /// Time step \[s].
    pub dt: f64,
    /// Frequency at which the medium's `α₀` is quoted \[Hz].
    pub reference_frequency_hz: f64,
    /// Lower edge of the band the power law must hold over \[Hz].
    pub band_min_hz: f64,
    /// Upper edge of the band the power law must hold over \[Hz].
    pub band_max_hz: f64,
    /// Number of relaxation arms, i.e. memory fields per voxel.
    pub relaxation_arms: usize,
}

impl PowerLawRelaxationSettings {
    /// Resolve from the solver configuration, or `None` when the configured
    /// absorption is [`FdtdAbsorption::Lossless`].
    #[must_use]
    pub fn from_config(absorption: &FdtdAbsorption, dt: f64) -> Option<Self> {
        match *absorption {
            FdtdAbsorption::Lossless => None,
            FdtdAbsorption::PowerLawRelaxation {
                reference_frequency_hz,
                band_min_hz,
                band_max_hz,
                relaxation_arms,
            } => Some(Self {
                dt,
                reference_frequency_hz,
                band_min_hz,
                band_max_hz,
                relaxation_arms,
            }),
        }
    }
}

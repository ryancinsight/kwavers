//! Multiscale frequency-continuation FWI driver (Bunks et al. 1995).
//!
//! # Specification
//!
//! The least-squares FWI objective is non-convex in the model: a poor starting
//! model that mispredicts an arrival by more than half a period drives the
//! gradient toward the wrong cycle (cycle skipping). Bunks et al. (1995) resolve
//! this by inverting a hierarchy of increasingly broadband data: at low
//! frequencies the half-period is large, so the same model error stays within
//! the basin of attraction of the global minimum; the recovered low-wavenumber
//! model then serves as the starting model for the next, higher-frequency band.
//!
//! ## Theorem (band-limited basin widening)
//!
//! Let `d(t)` have dominant period `T₀` and let `F_c` be a zero-phase low-pass
//! with corner `f_c < 1/T₀`. The filtered data `F_c d` has dominant period
//! `T_c ≈ 1/f_c > T₀`. The cycle-skipping threshold — the maximum traveltime
//! error tolerated by the L2 objective — is half the dominant period, so it
//! grows from `T₀/2` to `T_c/2`. Inverting low→high therefore admits starting
//! models with proportionally larger traveltime error.
//!
//! ## Adjoint consistency
//!
//! Band limiting is applied identically to observed and synthetic traces inside
//! the misfit and adjoint-source evaluation (see
//! [`FwiProcessor::compute_misfit_objective`](super::FwiProcessor) and
//! `compute_adjoint_source`). With a zero-phase filter `F` (`Fᵀ = F`), the
//! filtered objective `J(F d_syn, F d_obs)` has adjoint source
//! `Fᵀ ∂J/∂(F d_syn) = F · g̃`, i.e. the band-limited adjoint source is filtered
//! once more — preserving the discrete adjoint identity per stage.
//!
//! # References
//! - Bunks, C., Saleck, F.M., Zaleski, S., Chavent, G. (1995). *Multiscale
//!   seismic waveform inversion.* Geophysics 60(5), 1457–1473.
//! - Virieux, J., Operto, S. (2009). *An overview of full-waveform inversion in
//!   exploration geophysics.* Geophysics 74(6), WCC1–WCC26.

use super::{geometry::FwiGeometry, FwiProcessor};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use kwavers_math::fft::apply_spectral_response_1d;
use leto::{
    Array1,
    Array2,
    Array3,
};

/// Butterworth magnitude order for the zero-phase multiscale low-pass.
///
/// Order 4 gives a −24 dB/octave roll-off — steep enough to suppress the carrier
/// above the corner while leaving a smooth, ringing-free pass band.
const BUTTERWORTH_ORDER: i32 = 4;

/// Apply a zero-phase low-pass with `corner_hz` to every trace (row) of `data`.
///
/// The response is the Butterworth magnitude `|H(f)| = 1/√(1 + (f/f_c)^{2n})`
/// applied as a real, frequency-symmetric multiplier on the spectrum (folded so
/// bins above Nyquist receive the same gain as their conjugate partners). A real
/// symmetric multiplier introduces zero phase shift, so peak positions and
/// arrival times are preserved — the property the multiscale adjoint identity
/// relies on.
#[must_use]
pub(super) fn lowpass_band_limit(data: &Array2<f64>, dt: f64, corner_hz: f64) -> Array2<f64> {
    let fs = 1.0 / dt;
    let mut filtered = Array2::zeros(data.shape());
    for (row, mut out) in data
        .view()
        .axis_iter::<1>(0)
        .expect("invariant: axis 0 within 2-D trace matrix")
        .zip(
            filtered
                .view_mut()
                .axis_iter_mut::<1>(0)
                .expect("invariant: axis 0 within 2-D trace matrix"),
        )
    {
        let trace: Array1<f64> = row.to_contiguous();
        let response = apply_spectral_response_1d(&trace, fs, |_, freq, nyquist| {
            let f_eff = freq.min(2.0 * nyquist - freq).max(0.0);
            let ratio = (f_eff / corner_hz).powi(2 * BUTTERWORTH_ORDER);
            1.0 / (1.0 + ratio).sqrt()
        });
        out.assign(&response);
    }
    filtered
}

/// Validate a multiscale low-pass schedule: non-empty, finite, positive, and
/// strictly ascending. Single source of truth for the contract enforced by
/// [`FwiProcessor::invert_multiscale`].
///
/// # Errors
/// Returns [`KwaversError::Validation`] when any condition is violated.
pub(super) fn validate_corner_schedule(corner_hz_ascending: &[f64]) -> KwaversResult<()> {
    if corner_hz_ascending.is_empty() {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: "invert_multiscale requires at least one corner frequency".to_owned(),
            },
        ));
    }
    let mut previous = 0.0_f64;
    for &corner in corner_hz_ascending {
        if !corner.is_finite() || corner <= 0.0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "multiscale corner frequencies must be finite and positive; got {corner}"
                    ),
                },
            ));
        }
        if corner <= previous {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "multiscale corner frequencies must be strictly ascending; \
                         {corner} follows {previous}"
                    ),
                },
            ));
        }
        previous = corner;
    }
    Ok(())
}

impl FwiProcessor {
    /// Zero-phase low-pass the trace matrix at `corner_hz` for multiscale FWI.
    #[must_use]
    pub(super) fn band_limit(&self, data: &Array2<f64>, corner_hz: f64) -> Array2<f64> {
        lowpass_band_limit(data, self.parameters.dt, corner_hz)
    }

    /// Multiscale frequency-continuation FWI (single source).
    ///
    /// Inverts the supplied ascending sequence of low-pass corner frequencies in
    /// order, each stage starting from the previous stage's recovered model. The
    /// final corner sets the achieved resolution; pass a corner at or above the
    /// data bandwidth as the last entry for a full-band finish.
    ///
    /// Each stage reuses [`Self::invert`] with the stage corner applied through
    /// the band-limited misfit/adjoint path, so the per-stage gradient remains
    /// the exact discrete adjoint of the band-limited objective.
    ///
    /// # Errors
    /// - [`KwaversError::Validation`] if `corner_hz_ascending` is empty, contains
    ///   a non-finite or non-positive corner, or is not strictly ascending.
    /// - Propagates any [`KwaversError`] from the underlying [`Self::invert`].
    pub fn invert_multiscale(
        &self,
        observed_data: &Array2<f64>,
        initial_model: &Array3<f64>,
        geometry: &FwiGeometry,
        grid: &Grid,
        corner_hz_ascending: &[f64],
    ) -> KwaversResult<Array3<f64>> {
        validate_corner_schedule(corner_hz_ascending)?;

        let mut model = initial_model.clone();
        for (stage, &corner) in corner_hz_ascending.iter().enumerate() {
            let stage_processor = self.clone().with_band_limit(Some(corner));
            log::info!(
                "FWI multiscale stage {} / {}: low-pass corner = {:.3} Hz",
                stage + 1,
                (corner_hz_ascending.len()),
                corner
            );
            model = stage_processor.invert(observed_data, &model, geometry, grid)?;
        }
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse::fwi::time_domain::FwiProcessor;
    use crate::inverse::reconstruction::seismic::{MisfitFunction, MisfitType};
    use crate::inverse::seismic::parameters::FwiParameters;
    use std::f64::consts::TAU;

    const N: usize = 256;
    const DT: f64 = 1.0; // unit sampling → fs = 1 Hz, Nyquist = 0.5 Hz

    fn tone(freq_hz: f64) -> Array2<f64> {
        let mut data = Array2::zeros((1, N));
        for j in 0..N {
            data[[0, j]] = (TAU * freq_hz * j as f64).sin();
        }
        data
    }

    fn morlet(center: f64, period_samples: f64) -> Array2<f64> {
        let sigma = 4.0 * period_samples;
        let mut data = Array2::zeros((1, N));
        for j in 0..N {
            let u = j as f64 - center;
            data[[0, j]] = (-(u / sigma).powi(2)).exp() * (TAU * u / period_samples).cos();
        }
        data
    }

    fn energy(data: &Array2<f64>) -> f64 {
        data.iter().map(|&x| x * x).sum()
    }

    #[test]
    fn lowpass_passes_low_tone_and_rejects_high_tone() {
        // Corner 0.05 Hz: a 0.02 Hz tone is in-band, a 0.25 Hz tone is far above.
        let low = tone(0.02);
        let high = tone(0.25);

        let low_filtered = lowpass_band_limit(&low, DT, 0.05);
        let high_filtered = lowpass_band_limit(&high, DT, 0.05);

        let low_retained = energy(&low_filtered) / energy(&low);
        let high_retained = energy(&high_filtered) / energy(&high);

        assert!(
            low_retained > 0.9,
            "low tone must pass the band limit: retained {low_retained:.4}"
        );
        assert!(
            high_retained < 0.05,
            "high tone must be rejected: retained {high_retained:.4}"
        );
    }

    #[test]
    fn lowpass_is_zero_phase_preserving_symmetry() {
        // A symmetric Gaussian bump must stay symmetric about its centre under a
        // zero-phase filter (a phase-shifting filter would move the peak).
        let mut bump = Array2::zeros((1, N));
        let center = (N / 2) as f64;
        for j in 0..N {
            let u = j as f64 - center;
            bump[[0, j]] = (-(u / 20.0).powi(2)).exp();
        }
        let filtered = lowpass_band_limit(&bump, DT, 0.05);
        // Peak stays at the centre sample.
        let (peak_idx, _) = filtered.index_axis::<1>(0, 0).unwrap().iter().enumerate().fold(
            (0usize, f64::NEG_INFINITY),
            |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            },
        );
        assert_eq!(
            peak_idx,
            N / 2,
            "zero-phase filter must preserve peak position"
        );
        // Symmetry: f[center-k] ≈ f[center+k].
        for k in 1..40 {
            let left = filtered[[0, N / 2 - k]];
            let right = filtered[[0, N / 2 + k]];
            assert!(
                (left - right).abs() < 1e-9,
                "filtered bump must remain symmetric at offset {k}: {left:.3e} vs {right:.3e}"
            );
        }
    }

    #[test]
    fn band_limited_objective_matches_externally_filtered_misfit() {
        // The band-limit wiring must apply the same zero-phase low-pass to both
        // observed and synthetic before the misfit, i.e. evaluate J(F·d_obs,
        // F·d_syn). Verify against an externally filtered pair.
        let corner = 0.03;
        let dt = DT;
        let observed = morlet(128.0, 16.0);
        let synthetic = morlet(132.0, 16.0);

        let processor = FwiProcessor::new(FwiParameters {
            nt: N,
            dt,
            ..FwiParameters::default()
        })
        .with_band_limit(Some(corner));

        let objective = processor
            .compute_misfit_objective(&observed, &synthetic)
            .expect("band-limited objective");

        let obs_f = lowpass_band_limit(&observed, dt, corner);
        let syn_f = lowpass_band_limit(&synthetic, dt, corner);
        let diff = &syn_f - &obs_f;
        let expected = 0.5 * dt * diff.iter().map(|&x| x * x).sum::<f64>();

        assert!(
            (objective - expected).abs() <= 1e-12 * expected.max(1.0),
            "band-limited objective must equal L2 on filtered traces: got {objective:e}, expected {expected:e}"
        );
    }

    #[test]
    fn lower_frequency_widens_l2_monotone_basin() {
        // Multiscale principle (Bunks et al. 1995): the L2 misfit-vs-shift curve
        // first turns over near half the dominant period, so the monotone basin
        // half-width scales with the period (∝ 1/frequency). Lowering the
        // frequency — what the multiscale low-pass achieves on broadband data —
        // widens the basin of attraction of the global minimum.
        let l2 = MisfitFunction::new(MisfitType::L2Norm);

        // First shift at which the misfit stops increasing (the cycle-skip peak).
        let first_turn = |period: f64| -> usize {
            let observed = morlet(128.0, period);
            let mut previous = 0.0_f64;
            for shift in 1..(period as usize * 2) {
                let synthetic = morlet(128.0 + shift as f64, period);
                let misfit = l2.compute(&observed, &synthetic).expect("l2");
                if misfit < previous {
                    return shift - 1;
                }
                previous = misfit;
            }
            period as usize * 2
        };

        let turn_high = first_turn(16.0);
        let turn_low = first_turn(32.0);
        assert!(
            turn_low > turn_high,
            "lower frequency must widen the monotone basin: turn_low={turn_low}, turn_high={turn_high}"
        );
    }

    #[test]
    fn invert_multiscale_schedule_contract() {
        // Solver-free contract check on the shared schedule validator used by
        // invert_multiscale (full inversion is exercised by the integration FWI
        // tests). Ascending positive schedules pass.
        validate_corner_schedule(&[1.0, 2.0, 4.0]).expect("ascending schedule is valid");

        let err = validate_corner_schedule(&[]).expect_err("empty must fail");
        assert!(format!("{err:?}").contains("at least one corner"));
        let err = validate_corner_schedule(&[5.0, 5.0]).expect_err("non-ascending must fail");
        assert!(format!("{err:?}").contains("strictly ascending"));
        let err = validate_corner_schedule(&[-1.0]).expect_err("non-positive must fail");
        assert!(format!("{err:?}").contains("finite and positive"));
    }
}

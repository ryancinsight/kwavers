//! Ultrasound dosimetry and safety indices for book chapter ch15.
//!
//! Covers: Mechanical Index (MI), Thermal Index soft tissue (TIS), bone (TIB),
//! and cranial bone (TIC), a closed-loop thermal fixture, and FDA output
//! limits. Shared biological-response laws live in Asclepius.

use aequitas::systems::si::quantities::{ThermodynamicTemperature, Time};
use asclepius::{
    response::thermal::{Cem43, TemperatureSamples},
    EquivalentExposure,
};
use kwavers_core::constants::medical::{
    IEC_TIB_DIVISOR, IEC_TIC_COEFFICIENT_MW_PER_CM, IEC_TIS_DIVISOR,
};
use kwavers_core::constants::numerical::SECONDS_PER_MINUTE;
use kwavers_core::constants::thermodynamic::KELVIN_OFFSET_C;
use kwavers_core::error::{KwaversError, KwaversResult};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Mechanical Index (MI).
///
/// ```text
/// MI = |p_r| / (1e6 · √(f_MHz))   [dimensionless]
/// ```
/// where `p_r` is the peak rarefactional pressure in Pa and
/// `f_MHz = f_hz / 1e6`.
///
/// FDA limit: MI ≤ 1.9 (general) or 0.23 (ophthalmic).
///
/// Delegates to the canonical
/// [`crate::acoustics::analysis::calculate_mechanical_index`] so the
/// book-chapter API and the production safety paths share a single contract.
///
/// # Reference
/// FDA, *Marketing Clearance of Diagnostic Ultrasound Systems and
/// Transducers*, Appendix A; IEC 62359 (2017) §7.2.
#[inline]
pub fn mechanical_index(p_neg_pa: f64, f_hz: f64) -> f64 {
    crate::acoustics::analysis::calculate_mechanical_index(p_neg_pa, f_hz)
}

/// Mechanical Index over a pressure field (array variant).
///
/// Applies [`mechanical_index`] element-wise to every sample in `p_field`:
/// ```text
/// MI_i = |p_field[i]| / (1e6 · √(f_MHz))
/// ```
///
/// # Arguments
/// * `p_field` – peak rarefactional pressures in `Pa`, any shape passed as 1-D
/// * `f_hz` – centre frequency in `Hz`
///
/// # Reference
/// FDA, *Marketing Clearance of Diagnostic Ultrasound Systems and
/// Transducers*, Appendix A; IEC 62359 (2017) §7.2.
#[must_use]
#[inline]
pub fn mechanical_index_field(p_field: &[f64], f_hz: f64) -> Vec<f64> {
    p_field.iter().map(|&p| mechanical_index(p, f_hz)).collect()
}

/// Mechanical Index for one pressure threshold over a frequency sweep.
///
/// Applies [`mechanical_index`] element-wise to `f_hz` while holding the peak
/// rarefactional pressure fixed. This is the canonical safety-curve form used
/// by BBB/LIFU parameter-space plots where an inertial-cavitation pressure
/// threshold is visualized across frequency.
#[must_use]
#[inline]
pub fn mechanical_index_frequency_sweep(p_neg_pa: f64, f_hz: &[f64]) -> Vec<f64> {
    f_hz.iter()
        .map(|&f| mechanical_index(p_neg_pa, f))
        .collect()
}

/// Cavitation-risk probability as a logistic function of Mechanical Index.
///
/// ```text
/// P_risk(MI) = 1 / (1 + exp[-s · (MI − MI_thr)])
/// ```
///
/// `MI_thr` is the MI value at 50% risk and `s` controls the transition
/// steepness. The model is a phenomenological safety-envelope map used by the
/// neuromodulation book examples to turn MI into a smooth risk contour; it is
/// not a substitute for a validated cavitation-threshold experiment.
///
/// Non-finite MI values are mapped to zero risk. Non-finite thresholds or
/// non-positive/non-finite slopes make the model invalid, so the returned field
/// is all zeros.
///
/// # Arguments
/// * `mechanical_index` – Mechanical Index samples [-]
/// * `threshold_mi` – MI at 50% cavitation risk [-]
/// * `slope` – logistic slope in reciprocal MI units; must be positive
#[must_use]
pub fn mechanical_index_cavitation_risk(
    mechanical_index: &[f64],
    threshold_mi: f64,
    slope: f64,
) -> Vec<f64> {
    if !(threshold_mi.is_finite() && slope.is_finite() && slope > 0.0) {
        return vec![0.0; mechanical_index.len()];
    }
    mechanical_index
        .iter()
        .map(|&mi| {
            if mi.is_finite() {
                1.0 / (1.0 + (-(slope * (mi - threshold_mi))).exp())
            } else {
                0.0
            }
        })
        .collect()
}

/// Thermal Index for soft tissue (TIS).
///
/// Simplified IEC 62359 formula:
/// ```text
/// TIS = W_stp [mW] / (210 · f_MHz)
/// ```
///
/// # Arguments
/// * `wstp_mw` – spatial-temporal peak power at the focus in `mW`
/// * `f_mhz` – centre frequency in `MHz`
///
/// # Reference
/// IEC 62359 (2017) §8.3.2.
#[inline]
pub fn thermal_index_soft_tissue(wstp_mw: f64, f_mhz: f64) -> f64 {
    if !(wstp_mw.is_finite() && f_mhz.is_finite() && wstp_mw >= 0.0 && f_mhz > 0.0) {
        return 0.0;
    }
    wstp_mw / (IEC_TIS_DIVISOR * f_mhz)
}

/// Thermal Index for bone (TIB).
///
/// Simplified formula:
/// ```text
/// TIB = W [mW] · f_MHz / 40.0
/// ```
///
/// # Reference
/// IEC 62359 (2017) §8.4.
#[inline]
pub fn thermal_index_bone(w_mw: f64, f_mhz: f64) -> f64 {
    if !(w_mw.is_finite() && f_mhz.is_finite() && w_mw >= 0.0 && f_mhz >= 0.0) {
        return 0.0;
    }
    w_mw * f_mhz / IEC_TIB_DIVISOR
}

/// Thermal Index for cranial bone (TIC).
///
/// Frequency-independent (IEC 62359 §8.5): for transcranial exposure the
/// worst-case heating is at the skin–skull interface near the transducer, where
/// the full source power is deposited over the aperture rather than concentrated
/// by focal absorption — so TIC carries no `f` weighting (unlike TIS/TIB).
/// ```text
/// TIC = W_0 [mW] / (40 [mW/cm] · D_eq [cm])
/// ```
/// `W_0` is the total acoustic power at the transducer face and `D_eq` the
/// equivalent aperture diameter `√(4·A_aprt/π)`.
///
/// # Arguments
/// * `w0_mw` – total acoustic power at the transducer face in `mW`
/// * `aperture_diameter_cm` – equivalent aperture diameter `D_eq` in `cm`
///
/// # Reference
/// IEC 62359 (2017) §8.5; AIUM/NEMA UD-3:2012.
#[inline]
pub fn thermal_index_cranial(w0_mw: f64, aperture_diameter_cm: f64) -> f64 {
    if !(w0_mw.is_finite()
        && aperture_diameter_cm.is_finite()
        && w0_mw >= 0.0
        && aperture_diameter_cm > 0.0)
    {
        return 0.0;
    }
    w0_mw / (IEC_TIC_COEFFICIENT_MW_PER_CM * aperture_diameter_cm)
}

/// Rust-owned Chapter 7 focal-temperature and CEM43 dose fixture.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosedLoopCem43Fixture {
    /// Time axis in `s`.
    pub time_s: Vec<f64>,
    /// Fixed-power focal temperature trace [deg C].
    pub fixed_temperature_c: Vec<f64>,
    /// Feedback-controlled focal temperature trace [deg C].
    pub feedback_temperature_c: Vec<f64>,
    /// Under-driven focal temperature trace [deg C].
    pub underdrive_temperature_c: Vec<f64>,
    /// Cumulative CEM43 for the fixed-power trace in equivalent minutes.
    pub fixed_cem43_min: Vec<f64>,
    /// Cumulative CEM43 for the feedback trace in equivalent minutes.
    pub feedback_cem43_min: Vec<f64>,
    /// Cumulative CEM43 for the under-driven trace in equivalent minutes.
    pub underdrive_cem43_min: Vec<f64>,
}

/// Generate the Chapter 7 closed-loop thermal-dose fixture.
///
/// The fixed-power and underdrive traces are deterministic ramp-and-hold
/// references. The feedback trace models controller overshoot followed by
/// damped convergence, with deterministic seed-controlled MR-thermometry jitter.
/// CEM43 integration delegates directly to Asclepius.
pub fn closed_loop_cem43_fixture(
    n_steps: usize,
    dt_s: f64,
    body_temperature_c: f64,
    target_temperature_c: f64,
    seed: u64,
) -> KwaversResult<ClosedLoopCem43Fixture> {
    if n_steps == 0 {
        return Err(KwaversError::InvalidInput(
            "n_steps must be positive".to_string(),
        ));
    }
    if !(dt_s.is_finite() && dt_s > 0.0) {
        return Err(KwaversError::InvalidInput(
            "dt_s must be finite and positive".to_string(),
        ));
    }
    if !(body_temperature_c.is_finite() && target_temperature_c.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "temperatures must be finite".to_string(),
        ));
    }
    if target_temperature_c <= body_temperature_c {
        return Err(KwaversError::InvalidInput(
            "target_temperature_c must exceed body_temperature_c".to_string(),
        ));
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let steps: Vec<f64> = (0..n_steps).map(|i| i as f64).collect();
    let time_s: Vec<f64> = steps.iter().map(|&step| step * dt_s).collect();

    let fixed_temperature_c: Vec<f64> = steps
        .iter()
        .map(|&step| {
            if step < 30.0 {
                body_temperature_c + (target_temperature_c - body_temperature_c) * step / 30.0
            } else {
                target_temperature_c
            }
        })
        .collect();

    let feedback_temperature_c: Vec<f64> = steps
        .iter()
        .map(|&step| {
            if step < 20.0 {
                body_temperature_c
                    + (target_temperature_c + 10.0 - body_temperature_c) * step / 20.0
            } else {
                target_temperature_c
                    + 5.0 * (-0.15 * (step - 20.0)).exp()
                    + 2.0 * sample_standard_normal(&mut rng)
            }
        })
        .collect();

    let underdrive_temperature_c: Vec<f64> = steps
        .iter()
        .map(|&step| {
            if step < 40.0 {
                body_temperature_c + (56.0 - body_temperature_c) * step / 40.0
            } else {
                56.0
            }
        })
        .collect();

    let cumulative = |temperatures_c: &[f64]| -> KwaversResult<Vec<f64>> {
        let observation = TemperatureSamples::new(
            temperatures_c.iter().copied().map(|temperature_c| {
                ThermodynamicTemperature::from_base(temperature_c + KELVIN_OFFSET_C)
            }),
            Time::from_base(dt_s),
        )
        .map_err(|source| {
            KwaversError::InvalidInput(format!("invalid CEM43 fixture step: {source}"))
        })?;
        let mut exposure = vec![EquivalentExposure::zero(); temperatures_c.len()];
        Cem43::canonical()
            .cumulative_into(observation, &mut exposure)
            .map_err(|source| {
                KwaversError::InvalidInput(format!("invalid CEM43 fixture observation: {source}"))
            })?;
        Ok(exposure
            .into_iter()
            .map(|value| value.get().into_base() / SECONDS_PER_MINUTE)
            .collect())
    };
    let fixed_cem43_min = cumulative(&fixed_temperature_c)?;
    let feedback_cem43_min = cumulative(&feedback_temperature_c)?;
    let underdrive_cem43_min = cumulative(&underdrive_temperature_c)?;

    Ok(ClosedLoopCem43Fixture {
        time_s,
        fixed_temperature_c,
        feedback_temperature_c,
        underdrive_temperature_c,
        fixed_cem43_min,
        feedback_cem43_min,
        underdrive_cem43_min,
    })
}

fn sample_standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    let u1 = rng.r#gen::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
    let u2 = rng.r#gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// FDA ISPTA.3 output limit.
///
/// ```text
/// ISPTA.3 ≤ 720 mW/cm²
/// ```
///
/// # Reference
/// FDA, *Marketing Clearance of Diagnostic Ultrasound Systems and
/// Transducers*, Table 3.
#[inline]
pub fn fda_ispta_limit_mw_cm2() -> f64 {
    720.0
}

/// FDA ISPPA.3 output limit.
///
/// ```text
/// ISPPA.3 ≤ 190 W/cm²
/// ```
///
/// # Reference
/// FDA, *Marketing Clearance of Diagnostic Ultrasound Systems and
/// Transducers*, Table 3.
#[inline]
pub fn fda_isppa_limit_w_cm2() -> f64 {
    190.0
}

#[cfg(test)]
mod tests;

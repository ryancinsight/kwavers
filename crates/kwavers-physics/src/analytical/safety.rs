//! Ultrasound dosimetry and safety indices for book chapter ch15.
//!
//! Covers: Mechanical Index (MI), Thermal Index soft tissue (TIS), bone (TIB),
//! and cranial bone (TIC), CEM43 cumulative dose, Arrhenius damage integral, and
//! FDA output limits.

use kwavers_core::constants::fundamental::GAS_CONSTANT;
use kwavers_core::constants::medical::{
    IEC_TIB_DIVISOR, IEC_TIC_COEFFICIENT_MW_PER_CM, IEC_TIS_DIVISOR, THERMAL_DOSE_REFERENCE_TEMP_C,
    THERMAL_DOSE_R_ABOVE_43C, THERMAL_DOSE_R_BELOW_43C,
};
use kwavers_core::constants::numerical::SECONDS_PER_MINUTE;
use kwavers_core::constants::thermodynamic::KELVIN_OFFSET_C;
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
/// * `p_field` – peak rarefactional pressures [Pa], any shape passed as 1-D
/// * `f_hz` – centre frequency [Hz]
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
/// * `wstp_mw` – spatial-temporal peak power at the focus [mW]
/// * `f_mhz` – centre frequency [MHz]
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
/// * `w0_mw` – total acoustic power at the transducer face [mW]
/// * `aperture_diameter_cm` – equivalent aperture diameter `D_eq` [cm]
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

/// CEM43 cumulative equivalent minutes at 43 °C.
///
/// Computed at each time step i as the running sum:
/// ```text
/// CEM43[i] = Σ_{j=0}^{i} (dt/60) · R^(43 − T[j])
/// R = 0.5  if T[j] ≥ 43 °C
/// R = 0.25 if T[j] < 43 °C
/// ```
///
/// # Arguments
/// * `t_celsius` – temperature time series [°C]
/// * `dt_s` – time step [s]
///
/// Returns cumulative CEM43 at each time step [min].
///
/// # Reference
/// Sapareto & Dewey (1984), *Int. J. Radiat. Oncol. Biol. Phys.* 10, 787.
#[must_use]
pub fn cem43_cumulative(t_celsius: &[f64], dt_s: f64) -> Vec<f64> {
    let dt_min = dt_s / SECONDS_PER_MINUTE;
    let mut cem = 0.0_f64;
    t_celsius
        .iter()
        .map(|&t| {
            let r: f64 = if t >= THERMAL_DOSE_REFERENCE_TEMP_C {
                THERMAL_DOSE_R_ABOVE_43C
            } else {
                THERMAL_DOSE_R_BELOW_43C
            };
            cem += dt_min * r.powf(THERMAL_DOSE_REFERENCE_TEMP_C - t);
            cem
        })
        .collect()
}

/// Rust-owned Chapter 7 focal-temperature and CEM43 dose fixture.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosedLoopCem43Fixture {
    /// Time axis [s].
    pub time_s: Vec<f64>,
    /// Fixed-power focal temperature trace [deg C].
    pub fixed_temperature_c: Vec<f64>,
    /// Feedback-controlled focal temperature trace [deg C].
    pub feedback_temperature_c: Vec<f64>,
    /// Under-driven focal temperature trace [deg C].
    pub underdrive_temperature_c: Vec<f64>,
    /// Cumulative CEM43 for the fixed-power trace [min].
    pub fixed_cem43_min: Vec<f64>,
    /// Cumulative CEM43 for the feedback trace [min].
    pub feedback_cem43_min: Vec<f64>,
    /// Cumulative CEM43 for the under-driven trace [min].
    pub underdrive_cem43_min: Vec<f64>,
}

/// Generate the Chapter 7 closed-loop thermal-dose fixture.
///
/// The fixed-power and underdrive traces are deterministic ramp-and-hold
/// references. The feedback trace models controller overshoot followed by
/// damped convergence, with deterministic seed-controlled MR-thermometry jitter.
/// CEM43 integration delegates to [`cem43_cumulative`] so the temperature
/// fixture and dose contract share one implementation.
pub fn closed_loop_cem43_fixture(
    n_steps: usize,
    dt_s: f64,
    body_temperature_c: f64,
    target_temperature_c: f64,
    seed: u64,
) -> Result<ClosedLoopCem43Fixture, &'static str> {
    if n_steps == 0 {
        return Err("n_steps must be positive");
    }
    if !(dt_s.is_finite() && dt_s > 0.0) {
        return Err("dt_s must be finite and positive");
    }
    if !(body_temperature_c.is_finite() && target_temperature_c.is_finite()) {
        return Err("temperatures must be finite");
    }
    if target_temperature_c <= body_temperature_c {
        return Err("target_temperature_c must exceed body_temperature_c");
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

    let fixed_cem43_min = cem43_cumulative(&fixed_temperature_c, dt_s);
    let feedback_cem43_min = cem43_cumulative(&feedback_temperature_c, dt_s);
    let underdrive_cem43_min = cem43_cumulative(&underdrive_temperature_c, dt_s);

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

/// Arrhenius thermal damage integral.
///
/// ```text
/// Ω = A · ∫ exp(−Ea / (R_gas · T_K(t))) dt
/// ```
/// computed by the rectangle rule:
/// ```text
/// Ω ≈ A · Σ_i exp(−Ea / (R_gas · (T[i] + 273.15))) · dt
/// ```
/// `Ω ≥ 1` indicates irreversible damage (63% cell kill at Ω = 1).
///
/// # Arguments
/// * `t_celsius` – temperature time series [°C]
/// * `dt_s` – time step [s]
/// * `a_per_s` – frequency factor A [s⁻¹]
/// * `ea_j_mol` – activation energy Ea [J/mol]
///
/// # Reference
/// Henriques & Moritz (1947), *Am. J. Pathol.* 23, 531;
/// Bhowmick et al. (2002) for muscle tissue parameters.
pub fn arrhenius_damage_integral(t_celsius: &[f64], dt_s: f64, a_per_s: f64, ea_j_mol: f64) -> f64 {
    let r_gas = GAS_CONSTANT;
    t_celsius.iter().fold(0.0_f64, |acc, &t| {
        let t_k = t + KELVIN_OFFSET_C;
        acc + a_per_s * (-ea_j_mol / (r_gas * t_k)).exp() * dt_s
    })
}

/// Cumulative Arrhenius thermal-damage integral over a temperature time series.
///
/// Returns the running sum Ω(t_k) at each discrete time step:
/// ```text
/// Ω(t_k) = A · Σ_{i=0}^{k} exp(−Ea / (R_gas · T_K[i])) · dt
/// ```
/// The output has the same length as `t_celsius`; element `k` is the
/// total damage accumulated from t=0 through t=k·dt.
///
/// `Ω ≥ 1` indicates irreversible damage (Henriques criterion).
///
/// # Arguments
/// * `t_celsius` – temperature time series [°C]
/// * `dt_s` – time step [s]
/// * `a_per_s` – frequency factor A [s⁻¹]
/// * `ea_j_mol` – activation energy Ea [J/mol]
///
/// # Reference
/// Henriques & Moritz (1947), *Am. J. Pathol.* 23, 531.
#[must_use]
pub fn arrhenius_cumulative(t_celsius: &[f64], dt_s: f64, a_per_s: f64, ea_j_mol: f64) -> Vec<f64> {
    let r_gas = GAS_CONSTANT;
    let mut acc = 0.0_f64;
    t_celsius
        .iter()
        .map(|&t| {
            let t_k = t + KELVIN_OFFSET_C;
            acc += a_per_s * (-ea_j_mol / (r_gas * t_k)).exp() * dt_s;
            acc
        })
        .collect()
}

/// Cumulative thermal cell-death probability from the Arrhenius damage integral.
///
/// ```text
/// P_death(t_k) = 1 − exp(−Ω(t_k))
/// ```
/// where `Ω(t_k)` is the cumulative Arrhenius damage integral (see
/// [`arrhenius_cumulative`]). At `Ω = 1` this reproduces the Henriques (1947)
/// 63 % kill criterion (`1 − e^{-1} ≈ 0.632`); the probability saturates toward
/// 1 as thermal exposure accumulates. This is the canonical map from a thermal
/// dose history to a per-voxel kill probability, suitable for combination with a
/// mechanical (cavitation) kill probability via [`combined_kill_probability`].
///
/// # Arguments
/// * `t_celsius` – temperature time series [°C]
/// * `dt_s` – time step [s]
/// * `a_per_s` – frequency factor A [s⁻¹]
/// * `ea_j_mol` – activation energy Ea [J/mol]
///
/// Returns the cumulative kill probability at each time step, in [0, 1).
///
/// # Reference
/// Henriques & Moritz (1947), *Am. J. Pathol.* 23, 531;
/// Pearce (2013), *J. Biomech. Eng.* 135, 121002 (survival-fraction reading).
#[must_use]
pub fn arrhenius_kill_probability(
    t_celsius: &[f64],
    dt_s: f64,
    a_per_s: f64,
    ea_j_mol: f64,
) -> Vec<f64> {
    arrhenius_cumulative(t_celsius, dt_s, a_per_s, ea_j_mol)
        .into_iter()
        .map(|omega| 1.0 - (-omega).exp())
        .collect()
}

/// Per-voxel thermal cell-death probability for a steady temperature held for a
/// fixed duration.
///
/// Treats each element of `t_celsius` as an INDEPENDENT voxel held at a constant
/// temperature `T` for `duration_s`, giving the Arrhenius damage
/// `Ω = A·exp(−Ea/(R·T_K))·duration_s` and kill probability `P = 1 − exp(−Ω)`.
/// This is the spatial-field analogue of [`arrhenius_kill_probability`] (which
/// integrates a temperature *time series*), and mirrors the per-voxel-steady
/// semantics used for CEM43 field dosimetry — one steady temperature per voxel,
/// one exposure time — so a steady-state thermal map can be turned directly into
/// a biologically-effective thermal kill field for [`combined_kill_probability`].
///
/// # Arguments
/// * `t_celsius` – per-voxel steady temperature [°C]
/// * `duration_s` – exposure duration [s]
/// * `a_per_s` – frequency factor A [s⁻¹]
/// * `ea_j_mol` – activation energy Ea [J/mol]
///
/// Returns per-voxel kill probability in [0, 1), same length as `t_celsius`.
///
/// # Reference
/// Henriques & Moritz (1947), *Am. J. Pathol.* 23, 531.
#[must_use]
pub fn arrhenius_steady_kill_probability(
    t_celsius: &[f64],
    duration_s: f64,
    a_per_s: f64,
    ea_j_mol: f64,
) -> Vec<f64> {
    let r_gas = GAS_CONSTANT;
    t_celsius
        .iter()
        .map(|&t| {
            let t_k = t + KELVIN_OFFSET_C;
            let omega = a_per_s * (-ea_j_mol / (r_gas * t_k)).exp() * duration_s;
            1.0 - (-omega).exp()
        })
        .collect()
}

/// Combine independent cell-kill insults into one biologically-effective kill
/// probability.
///
/// For statistically independent mechanical (cavitation/fractionation) and
/// thermal (protein-denaturation) damage mechanisms the *survival* probabilities
/// multiply, so the combined kill probability is
/// ```text
/// P_kill = 1 − (1 − P_mech)·(1 − P_thermal)
/// ```
/// Each input is clamped to [0, 1]; the result lies in [0, 1] and is never less
/// than `max(P_mech, P_thermal)` (adding an insult cannot reduce kill). This is
/// the single source of truth for fusing the mechanical and thermal dose maps of
/// a histotripsy / boiling-histotripsy treatment into one biologically-effective
/// dose field.
///
/// The two slices are combined element-wise over the shorter of the two lengths.
///
/// # Arguments
/// * `p_mech` – per-voxel mechanical (cavitation) kill probability, in [0, 1]
/// * `p_thermal` – per-voxel thermal kill probability, in [0, 1]
///   (e.g. from [`arrhenius_kill_probability`])
///
/// # Reference
/// Independent-insult survival product (Pearce 2013); multi-mechanism
/// histotripsy damage (Vlaisavljevich et al. 2015).
#[must_use]
pub fn combined_kill_probability(p_mech: &[f64], p_thermal: &[f64]) -> Vec<f64> {
    p_mech
        .iter()
        .zip(p_thermal.iter())
        .map(|(&m, &t)| {
            let m = m.clamp(0.0, 1.0);
            let t = t.clamp(0.0, 1.0);
            1.0 - (1.0 - m) * (1.0 - t)
        })
        .collect()
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::medical::THERMAL_DOSE_REFERENCE_TEMP_C;
    use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
    use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;

    #[test]
    fn mi_dimensional_check() {
        // At 1 MPa negative, 1 MHz: MI = 1.0
        let mi = mechanical_index(-MPA_TO_PA, MHZ_TO_HZ);
        assert!((mi - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mi_rejects_nonpositive_frequency() {
        assert_eq!(mechanical_index(-MPA_TO_PA, 0.0), 0.0);
        assert_eq!(mechanical_index(-MPA_TO_PA, -MHZ_TO_HZ), 0.0);
    }

    #[test]
    fn thermal_indices_are_nonnegative_ratios() {
        assert!((thermal_index_soft_tissue(210.0, 1.0) - 1.0).abs() < 1e-12);
        assert!((thermal_index_bone(40.0, 1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn thermal_indices_reject_invalid_domains() {
        assert_eq!(thermal_index_soft_tissue(1.0, 0.0), 0.0);
        assert_eq!(thermal_index_soft_tissue(-1.0, 1.0), 0.0);
        assert_eq!(thermal_index_soft_tissue(f64::NAN, 1.0), 0.0);
        assert_eq!(thermal_index_bone(-1.0, 1.0), 0.0);
        assert_eq!(thermal_index_bone(1.0, -1.0), 0.0);
        assert_eq!(thermal_index_bone(1.0, f64::INFINITY), 0.0);
    }

    #[test]
    fn thermal_index_cranial_matches_iec_definition() {
        // TIC = W_0 / (40·D_eq): W_0 = 80 mW, D_eq = 2 cm ⇒ 80/(40·2) = 1.0.
        assert!((thermal_index_cranial(80.0, 2.0) - 1.0).abs() < 1e-12);
        // Frequency-independent and linear in power: doubling W_0 doubles TIC.
        assert!((thermal_index_cranial(160.0, 2.0) - 2.0).abs() < 1e-12);
        // A larger aperture spreads the same power ⇒ strictly lower TIC.
        assert!(thermal_index_cranial(80.0, 4.0) < thermal_index_cranial(80.0, 2.0));
        // Invalid domains → 0 (aperture must be strictly positive).
        assert_eq!(thermal_index_cranial(-1.0, 2.0), 0.0);
        assert_eq!(thermal_index_cranial(80.0, 0.0), 0.0);
        assert_eq!(thermal_index_cranial(f64::NAN, 2.0), 0.0);
        assert_eq!(thermal_index_cranial(80.0, f64::INFINITY), 0.0);
    }

    #[test]
    fn cem43_flat_43c() {
        // At exactly 43°C, R = 0.5, 0.5^0 = 1 → each step contributes dt/60
        let t = vec![THERMAL_DOSE_REFERENCE_TEMP_C; 6];
        let dt = 10.0; // seconds
        let c = cem43_cumulative(&t, dt);
        let expected: Vec<f64> = (1..=6).map(|i| i as f64 * dt / 60.0).collect();
        for (got, exp) in c.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-10, "got={} exp={}", got, exp);
        }
    }

    #[test]
    fn cem43_below_threshold_smaller() {
        // Below 43°C (R=0.25), increments are smaller than at 43°C
        let t_above = vec![THERMAL_DOSE_REFERENCE_TEMP_C; 3];
        let t_below = vec![BODY_TEMPERATURE_C; 3];
        let dt = 1.0;
        let c_above = cem43_cumulative(&t_above, dt);
        let c_below = cem43_cumulative(&t_below, dt);
        assert!(c_below.last().unwrap() < c_above.last().unwrap());
    }

    #[test]
    fn closed_loop_cem43_fixture_is_seeded_and_uses_cem43_contract() {
        let fixture = closed_loop_cem43_fixture(60, 0.5, 37.0, 60.0, 42)
            .expect("valid closed-loop CEM43 fixture");
        let repeated = closed_loop_cem43_fixture(60, 0.5, 37.0, 60.0, 42)
            .expect("valid closed-loop CEM43 fixture");
        let shifted_seed = closed_loop_cem43_fixture(60, 0.5, 37.0, 60.0, 43)
            .expect("valid closed-loop CEM43 fixture");

        assert_eq!(fixture, repeated);
        assert_ne!(
            fixture.feedback_temperature_c,
            shifted_seed.feedback_temperature_c
        );
        assert_eq!(fixture.time_s.len(), 60);
        assert_eq!(fixture.time_s[0], 0.0);
        assert_eq!(fixture.time_s[1], 0.5);
        assert_eq!(fixture.fixed_temperature_c[0], 37.0);
        assert_eq!(fixture.fixed_temperature_c[30], 60.0);
        assert_eq!(fixture.underdrive_temperature_c[40], 56.0);

        assert_eq!(
            fixture.fixed_cem43_min,
            cem43_cumulative(&fixture.fixed_temperature_c, 0.5)
        );
        assert_eq!(
            fixture.feedback_cem43_min,
            cem43_cumulative(&fixture.feedback_temperature_c, 0.5)
        );
        assert_eq!(
            fixture.underdrive_cem43_min,
            cem43_cumulative(&fixture.underdrive_temperature_c, 0.5)
        );
        for dose in [
            &fixture.fixed_cem43_min,
            &fixture.feedback_cem43_min,
            &fixture.underdrive_cem43_min,
        ] {
            assert!(dose.windows(2).all(|w| w[1] >= w[0]));
        }
    }

    #[test]
    fn closed_loop_cem43_fixture_rejects_invalid_inputs() {
        assert!(closed_loop_cem43_fixture(0, 0.5, 37.0, 60.0, 0).is_err());
        assert!(closed_loop_cem43_fixture(60, 0.0, 37.0, 60.0, 0).is_err());
        assert!(closed_loop_cem43_fixture(60, 0.5, f64::NAN, 60.0, 0).is_err());
        assert!(closed_loop_cem43_fixture(60, 0.5, 37.0, 37.0, 0).is_err());
    }

    #[test]
    fn arrhenius_positive() {
        // 70°C for 1 s with muscle parameters (A ≈ 3.1e98, Ea ≈ 6.28e5 J/mol)
        let t = vec![70.0; 100];
        let omega = arrhenius_damage_integral(&t, 0.01, 3.1e98, 6.28e5);
        assert!(omega > 0.0);
    }

    #[test]
    fn arrhenius_kill_probability_is_one_minus_exp_neg_omega() {
        // Contract: P_death(t_k) = 1 - exp(-Ω(t_k)) element-wise vs the
        // cumulative Arrhenius integral; monotone non-decreasing; bounded [0, 1].
        let a = 3.1e98;
        let ea = 6.28e5;
        let t = vec![70.0; 200];
        let dt = 0.01;
        let omega = arrhenius_cumulative(&t, dt, a, ea);
        let p = arrhenius_kill_probability(&t, dt, a, ea);
        for (&pk, &ok) in p.iter().zip(omega.iter()) {
            assert!((pk - (1.0 - (-ok).exp())).abs() < 1e-12);
        }
        assert!(p.windows(2).all(|w| w[1] >= w[0] - 1e-15), "monotone");
        assert!(p.iter().all(|&x| (0.0..=1.0).contains(&x)), "bounded [0,1]");
        // The Henriques Ω = 1 point gives the 63% kill criterion (1 - e^-1).
        assert!((1.0 - (-1.0_f64).exp() - 0.6321205588).abs() < 1e-9);
    }

    #[test]
    fn arrhenius_kill_probability_zero_at_body_temp() {
        // At 37 °C over a short exposure, thermal kill is negligible.
        let p = arrhenius_kill_probability(&vec![37.0; 50], 0.01, 3.1e98, 6.28e5);
        assert!(*p.last().unwrap() < 1e-6);
    }

    #[test]
    fn arrhenius_steady_kill_matches_time_series_single_step() {
        // A single-element steady field for `duration` must equal a 1-step
        // time-series kill with dt = duration (same Ω).
        let a = 7.39e66;
        let ea = 4.30e5;
        let dur = 30.0;
        let steady = arrhenius_steady_kill_probability(&[57.0], dur, a, ea);
        let series = arrhenius_kill_probability(&[57.0], dur, a, ea);
        assert!((steady[0] - series[0]).abs() < 1e-12);
        // Per-voxel independence: hotter voxel kills more.
        let field = arrhenius_steady_kill_probability(&[45.0, 55.0, 65.0], dur, a, ea);
        assert!(field[0] <= field[1] && field[1] <= field[2]);
        assert!(field.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn combined_kill_is_independent_insult_product() {
        // P_kill = 1 - (1-m)(1-t); verify a few exact values + the bounds.
        let m = [0.0, 0.5, 0.9, 1.0];
        let t = [0.0, 0.5, 0.2, 0.0];
        let c = combined_kill_probability(&m, &t);
        let expect: Vec<f64> = m
            .iter()
            .zip(t.iter())
            .map(|(&a, &b)| 1.0 - (1.0 - a) * (1.0 - b))
            .collect();
        for (got, exp) in c.iter().zip(expect.iter()) {
            assert!((got - exp).abs() < 1e-12, "got={got} exp={exp}");
        }
        // Adding an insult never reduces kill: combined ≥ max(m, t).
        for ((&a, &b), &got) in m.iter().zip(t.iter()).zip(c.iter()) {
            assert!(got >= a.max(b) - 1e-12);
            assert!((0.0..=1.0).contains(&got));
        }
    }

    #[test]
    fn combined_kill_clamps_out_of_range_inputs() {
        // Inputs outside [0,1] are clamped before combination.
        let c = combined_kill_probability(&[1.5, -0.3], &[0.4, 2.0]);
        assert!((c[0] - 1.0).abs() < 1e-12); // m clamps to 1 → kill 1
        assert!((c[1] - 1.0).abs() < 1e-12); // t clamps to 1 → kill 1
    }

    #[test]
    fn fda_limits_positive() {
        assert!(fda_ispta_limit_mw_cm2() > 0.0);
        assert!(fda_isppa_limit_w_cm2() > 0.0);
    }

    #[test]
    fn mechanical_index_field_matches_scalar() {
        // Field variant must produce the same result as the scalar variant per element.
        let pressures = [-MPA_TO_PA, -2.0 * MPA_TO_PA, -0.5 * MPA_TO_PA];
        let f_hz = 3.0 * MHZ_TO_HZ;
        let field = mechanical_index_field(&pressures, f_hz);
        for (&p, &mi_field) in pressures.iter().zip(field.iter()) {
            let mi_scalar = mechanical_index(p, f_hz);
            assert!(
                (mi_field - mi_scalar).abs() < 1e-12,
                "field={mi_field} scalar={mi_scalar}"
            );
        }
    }

    #[test]
    fn mechanical_index_frequency_sweep_matches_scalar() {
        let pressure = -0.45 * MPA_TO_PA;
        let frequencies = [0.25 * MHZ_TO_HZ, MHZ_TO_HZ, 2.25 * MHZ_TO_HZ];
        let sweep = mechanical_index_frequency_sweep(pressure, &frequencies);

        for (&f_hz, &mi_sweep) in frequencies.iter().zip(sweep.iter()) {
            let mi_scalar = mechanical_index(pressure, f_hz);
            assert!(
                (mi_sweep - mi_scalar).abs() < 1e-12,
                "sweep={mi_sweep} scalar={mi_scalar}"
            );
        }
        assert!(sweep[0] > sweep[1]);
        assert!(sweep[1] > sweep[2]);
    }

    #[test]
    fn mechanical_index_cavitation_risk_at_threshold_is_half() {
        let risk = mechanical_index_cavitation_risk(&[0.7], 0.7, 18.0);
        assert!(
            (risk[0] - 0.5).abs() < 1e-12,
            "P_risk(MI_thr)={} != 0.5",
            risk[0]
        );
    }

    #[test]
    fn mechanical_index_cavitation_risk_matches_logistic_formula() {
        let mi = [0.2, 0.7, 1.0, 1.9];
        let threshold = 0.7_f64;
        let slope = 18.0_f64;
        let risk = mechanical_index_cavitation_risk(&mi, threshold, slope);
        for (actual, mi_value) in risk.iter().zip(mi) {
            let expected = 1.0 / (1.0 + (-(slope * (mi_value - threshold))).exp());
            assert!(
                (actual - expected).abs() < 1e-12,
                "P_risk({mi_value})={actual} != {expected}"
            );
        }
    }

    #[test]
    fn mechanical_index_cavitation_risk_is_bounded_and_monotone() {
        let mi: Vec<f64> = (0..=20).map(|i| i as f64 * 0.1).collect();
        let risk = mechanical_index_cavitation_risk(&mi, 0.7, 18.0);
        for (k, &value) in risk.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&value),
                "P_risk[{k}]={value} is outside [0, 1]"
            );
        }
        for k in 1..risk.len() {
            assert!(
                risk[k] >= risk[k - 1],
                "P_risk[{}]={} < P_risk[{}]={}",
                k,
                risk[k],
                k - 1,
                risk[k - 1]
            );
        }
    }

    #[test]
    fn mechanical_index_cavitation_risk_rejects_invalid_domains() {
        assert_eq!(
            mechanical_index_cavitation_risk(&[0.7, f64::NAN], 0.7, 18.0),
            vec![0.5, 0.0]
        );
        assert_eq!(
            mechanical_index_cavitation_risk(&[0.7, 1.0], f64::NAN, 18.0),
            vec![0.0, 0.0]
        );
        assert_eq!(
            mechanical_index_cavitation_risk(&[0.7, 1.0], 0.7, 0.0),
            vec![0.0, 0.0]
        );
        assert_eq!(
            mechanical_index_cavitation_risk(&[0.7, 1.0], 0.7, f64::INFINITY),
            vec![0.0, 0.0]
        );
    }
}

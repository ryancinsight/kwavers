//! Pressure-based calculations (intensity, MI, TI)

use ndarray::{Array3, ArrayView3};

#[inline]
fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

#[inline]
fn nonnegative_finite(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

#[inline]
pub(crate) fn acoustic_impedance(density: f64, sound_speed: f64) -> Option<f64> {
    if positive_finite(density) && positive_finite(sound_speed) {
        let impedance = density * sound_speed;
        positive_finite(impedance).then_some(impedance)
    } else {
        None
    }
}

#[inline]
pub(crate) fn harmonic_peak_intensity(pressure: f64, impedance: f64) -> f64 {
    if pressure.is_finite() {
        pressure.powi(2) / (2.0 * impedance)
    } else {
        0.0
    }
}

fn peak_pulse_average_intensity(pressure_field: &Array3<f64>, impedance: f64) -> f64 {
    pressure_field
        .iter()
        .map(|&pressure| harmonic_peak_intensity(pressure, impedance))
        .fold(0.0, f64::max)
}

/// Calculate harmonic peak intensity field from pressure.
///
/// Theorem: for a harmonic acoustic pressure amplitude `p` propagating through a
/// lossless fluid with positive finite acoustic impedance `Z = rho c`, the
/// cycle-averaged intensity is `I = p^2 / (2Z)`. Undefined impedance domains and
/// nonfinite samples are rejected with zero contribution because this API has no
/// error channel.
#[must_use]
pub fn calculate_intensity(
    pressure_field: ArrayView3<f64>,
    density: f64,
    sound_speed: f64,
) -> Array3<f64> {
    let Some(impedance) = acoustic_impedance(density, sound_speed) else {
        return Array3::zeros(pressure_field.dim());
    };

    pressure_field.mapv(|pressure| harmonic_peak_intensity(pressure, impedance))
}

/// Calculate Mechanical Index (MI)
///
/// MI = P_neg / sqrt(f_c)
/// where P_neg is peak negative pressure in MPa and f_c is center frequency in MHz
/// Undefined pressure or frequency domains return zero because MI is a
/// nonnegative dimensionless exposure index.
#[must_use]
pub fn calculate_mechanical_index(peak_negative_pressure: f64, frequency: f64) -> f64 {
    if !peak_negative_pressure.is_finite() || !positive_finite(frequency) {
        return 0.0;
    }

    let p_neg_mpa = peak_negative_pressure.abs() / 1e6;
    let freq_mhz = frequency / 1e6;

    p_neg_mpa / freq_mhz.sqrt()
}

/// Calculate Thermal Index (TI)
///
/// **Implementation**: Basic TI₀ calculation per IEC 62359:2017 §5.2.1
/// Uses acoustic power and tissue absorption to estimate thermal deposition.
/// Full TIS/TIB/TIC calculations require detailed anatomical models and beam geometry.
/// Current approximation suitable for general safety assessment.
///
/// **References**:
/// - IEC 62359:2017 "Ultrasonics - Field characterization - Test methods for thermal index"
/// - AIUM/NEMA (2004) "Standard for Real-Time Display of Thermal and Mechanical Indices"
#[must_use]
pub fn calculate_thermal_index(acoustic_power: f64, frequency: f64, tissue_absorption: f64) -> f64 {
    const REFERENCE_POWER: f64 = 0.04; // 40 mW reference per IEC 62359:2017

    if !nonnegative_finite(acoustic_power)
        || !positive_finite(frequency)
        || !nonnegative_finite(tissue_absorption)
    {
        return 0.0;
    }

    let freq_mhz = frequency / 1e6;
    let absorption_factor = tissue_absorption * freq_mhz;

    (acoustic_power * absorption_factor) / REFERENCE_POWER
}

/// Calculate derated pressure (accounting for tissue attenuation).
///
/// Contract: `p_d = p 10^(-0.3 z_cm f_MHz / 20)` for finite pressure,
/// nonnegative finite frequency, and nonnegative finite depth. Negative
/// frequency or depth would invert attenuation into gain and is rejected.
#[must_use]
pub fn calculate_derated_pressure(pressure: f64, frequency: f64, depth: f64) -> f64 {
    // FDA derating: 0.3 dB/cm/MHz
    const DERATING_FACTOR: f64 = 0.3; // dB/cm/MHz

    if !pressure.is_finite() || !nonnegative_finite(frequency) || !nonnegative_finite(depth) {
        return 0.0;
    }

    let freq_mhz = frequency / 1e6;
    let depth_cm = depth * 100.0;

    let attenuation_db = DERATING_FACTOR * depth_cm * freq_mhz;
    let attenuation_factor = 10.0_f64.powf(-attenuation_db / 20.0);

    pressure * attenuation_factor
}

/// Calculate spatial peak temporal average intensity (I_SPTA)
#[must_use]
pub fn calculate_ispta(
    pressure_field: &Array3<f64>,
    density: f64,
    sound_speed: f64,
    duty_cycle: f64,
) -> f64 {
    if !duty_cycle.is_finite() || !(0.0..=1.0).contains(&duty_cycle) {
        return 0.0;
    }

    let Some(impedance) = acoustic_impedance(density, sound_speed) else {
        return 0.0;
    };

    peak_pulse_average_intensity(pressure_field, impedance) * duty_cycle
}

/// Calculate spatial peak pulse average intensity (I_SPPA)
#[must_use]
pub fn calculate_isppa(pressure_field: &Array3<f64>, density: f64, sound_speed: f64) -> f64 {
    let Some(impedance) = acoustic_impedance(density, sound_speed) else {
        return 0.0;
    };

    peak_pulse_average_intensity(pressure_field, impedance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use ndarray::Array3;

    // ── calculate_intensity ───────────────────────────────────────────────────

    /// I = p²/(2ρc). At p=1, ρ=1000, c=1500: I = 1/(2×1500000) = 3.333e-7 W/m².
    #[test]
    fn calculate_intensity_matches_acoustic_intensity_formula() {
        let field = Array3::<f64>::from_elem((2, 2, 2), 1.0_f64);
        let intensity = calculate_intensity(field.view(), 1000.0, SOUND_SPEED_WATER_SIM);
        let expected = 1.0 / (2.0 * 1000.0 * SOUND_SPEED_WATER_SIM);
        for &v in intensity.iter() {
            assert!((v - expected).abs() < 1e-20, "I = p²/(2ρc) (got {v:.3e})");
        }
    }

    /// Zero pressure → zero intensity.
    #[test]
    fn calculate_intensity_zero_for_zero_pressure() {
        let field = Array3::<f64>::zeros((2, 2, 2));
        let intensity = calculate_intensity(field.view(), 1000.0, SOUND_SPEED_WATER_SIM);
        for &v in intensity.iter() {
            assert_eq!(v, 0.0);
        }
    }

    /// Invalid impedance and nonfinite pressure samples have no physical
    /// intensity interpretation in this scalar API and contribute zero.
    #[test]
    fn calculate_intensity_rejects_invalid_impedance_and_nonfinite_pressure() {
        let field =
            Array3::<f64>::from_shape_vec((2, 1, 1), vec![f64::NAN, 2.0]).expect("shape matches");

        let invalid_impedance = calculate_intensity(field.view(), -1000.0, SOUND_SPEED_WATER_SIM);
        assert!(invalid_impedance.iter().all(|&value| value == 0.0));

        let intensity = calculate_intensity(field.view(), 1000.0, SOUND_SPEED_WATER_SIM);
        let expected = 2.0_f64.powi(2) / (2.0 * 1000.0 * SOUND_SPEED_WATER_SIM);
        assert_eq!(intensity[[0, 0, 0]], 0.0);
        assert!((intensity[[1, 0, 0]] - expected).abs() < 1e-20);
    }

    // ── calculate_mechanical_index ────────────────────────────────────────────

    /// MI = |P_neg,MPa| / √f_MHz. At P=0.5 MPa, f=1 MHz: MI = 0.5/1 = 0.5.
    #[test]
    fn mechanical_index_matches_formula_at_half_mpa_one_mhz() {
        let mi = calculate_mechanical_index(-0.5e6, 1e6);
        assert!((mi - 0.5).abs() < 1e-12, "MI must be 0.5 (got {mi:.6})");
    }

    /// Zero frequency → MI = 0.0 (guarded branch).
    #[test]
    fn mechanical_index_zero_for_zero_frequency() {
        let mi = calculate_mechanical_index(1e6, 0.0);
        assert_eq!(mi, 0.0);
    }

    /// Nonfinite pressure or invalid frequency cannot define a finite MI.
    #[test]
    fn mechanical_index_rejects_nonfinite_pressure_and_invalid_frequency() {
        assert_eq!(calculate_mechanical_index(f64::NAN, 1e6), 0.0);
        assert_eq!(calculate_mechanical_index(f64::INFINITY, 1e6), 0.0);
        assert_eq!(calculate_mechanical_index(1e6, -1e6), 0.0);
        assert_eq!(calculate_mechanical_index(1e6, f64::NAN), 0.0);
        assert_eq!(calculate_mechanical_index(1e6, f64::INFINITY), 0.0);
    }

    // ── calculate_thermal_index ───────────────────────────────────────────────

    /// TI₀ = P_abs/P_ref. With P=40 mW, absorption factor=1, TI = 1.
    #[test]
    fn thermal_index_matches_reference_power_ratio() {
        let ti = calculate_thermal_index(0.04, 1e6, 1.0);
        assert!((ti - 1.0).abs() < 1e-12, "TI reference ratio (got {ti:.6})");
    }

    /// Negative or nonfinite deposition factors cannot produce a valid
    /// nonnegative thermal exposure ratio.
    #[test]
    fn thermal_index_rejects_negative_or_nonfinite_domains() {
        assert_eq!(calculate_thermal_index(-0.04, 1e6, 1.0), 0.0);
        assert_eq!(calculate_thermal_index(0.04, 0.0, 1.0), 0.0);
        assert_eq!(calculate_thermal_index(0.04, -1e6, 1.0), 0.0);
        assert_eq!(calculate_thermal_index(0.04, 1e6, -1.0), 0.0);
        assert_eq!(calculate_thermal_index(f64::NAN, 1e6, 1.0), 0.0);
        assert_eq!(calculate_thermal_index(0.04, f64::INFINITY, 1.0), 0.0);
    }

    // ── calculate_derated_pressure ────────────────────────────────────────────

    /// At depth=0, attenuation_dB = 0 → factor=1 → derated = original.
    #[test]
    fn derated_pressure_unchanged_at_zero_depth() {
        let p = 1e5_f64;
        let derated = calculate_derated_pressure(p, 1e6, 0.0);
        assert_eq!(
            derated, p,
            "derated pressure at depth=0 must equal original"
        );
    }

    /// At f=1 MHz, depth=10 cm: attenuation_dB = 0.3 × 10 × 1 = 3 dB
    /// → factor = 10^(-3/20) = 0.7079... → derated = p × factor.
    #[test]
    fn derated_pressure_matches_fda_3db_at_10cm_1mhz() {
        let p = 1.0_f64;
        let derated = calculate_derated_pressure(p, 1e6, 0.10); // 10 cm = 0.10 m
        let expected = 10.0_f64.powf(-3.0 / 20.0);
        assert!(
            (derated - expected).abs() < 1e-14,
            "3 dB at 10cm (got {derated:.6})"
        );
    }

    /// Negative frequency or depth would convert attenuation into gain and is
    /// rejected by the physical domain contract.
    #[test]
    fn derated_pressure_rejects_negative_or_nonfinite_domains() {
        assert_eq!(calculate_derated_pressure(f64::NAN, 1e6, 0.10), 0.0);
        assert_eq!(calculate_derated_pressure(1.0, -1e6, 0.10), 0.0);
        assert_eq!(calculate_derated_pressure(1.0, 1e6, -0.10), 0.0);
        assert_eq!(calculate_derated_pressure(1.0, f64::INFINITY, 0.10), 0.0);
    }

    // ── calculate_ispta ───────────────────────────────────────────────────────

    /// ISPTA = max(p²/(2ρc)) × duty_cycle. Uniform field with p=2: I_max=4/(2ρc).
    #[test]
    fn ispta_equals_peak_intensity_times_duty_cycle() {
        let field = Array3::<f64>::from_elem((2, 2, 2), 2.0_f64);
        let rho = DENSITY_WATER_NOMINAL;
        let c = SOUND_SPEED_WATER_SIM;
        let duty = 0.1_f64;
        let ispta = calculate_ispta(&field, rho, c, duty);
        let expected = 2.0_f64.powi(2) / (2.0 * rho * c) * duty;
        assert!(
            (ispta - expected).abs() < 1e-20,
            "ISPTA formula (got {ispta:.3e})"
        );
    }

    /// Duty cycle is a probability-like temporal fraction and must stay in
    /// [0, 1]; invalid impedance also makes intensity undefined.
    #[test]
    fn ispta_rejects_invalid_duty_cycle_and_impedance() {
        let field = Array3::<f64>::from_elem((2, 2, 2), 2.0_f64);
        assert_eq!(
            calculate_ispta(&field, 1000.0, SOUND_SPEED_WATER_SIM, -0.1),
            0.0
        );
        assert_eq!(
            calculate_ispta(&field, 1000.0, SOUND_SPEED_WATER_SIM, 1.1),
            0.0
        );
        assert_eq!(
            calculate_ispta(&field, 1000.0, SOUND_SPEED_WATER_SIM, f64::NAN),
            0.0
        );
        assert_eq!(
            calculate_ispta(&field, 0.0, SOUND_SPEED_WATER_SIM, 0.1),
            0.0
        );
    }

    /// Nonfinite pressure samples do not dominate the spatial peak.
    #[test]
    fn ispta_ignores_nonfinite_pressure_samples() {
        let field =
            Array3::<f64>::from_shape_vec((2, 1, 1), vec![f64::INFINITY, 2.0]).expect("shape");
        let duty = 0.25;
        let ispta = calculate_ispta(&field, 1000.0, SOUND_SPEED_WATER_SIM, duty);
        let expected = 2.0_f64.powi(2) / (2.0 * 1000.0 * SOUND_SPEED_WATER_SIM) * duty;
        assert!((ispta - expected).abs() < 1e-20);
    }

    // ── calculate_isppa ───────────────────────────────────────────────────────

    /// ISPPA = max(p²/(2ρc)). Spike at p=3: ISPPA = 9/(2ρc).
    #[test]
    fn isppa_equals_peak_intensity() {
        let mut field = Array3::<f64>::zeros((4, 4, 4));
        field[[2, 2, 2]] = 3.0;
        let rho = DENSITY_WATER_NOMINAL;
        let c = SOUND_SPEED_WATER_SIM;
        let isppa = calculate_isppa(&field, rho, c);
        let expected = 3.0_f64.powi(2) / (2.0 * rho * c);
        assert!(
            (isppa - expected).abs() < 1e-20,
            "ISPPA formula (got {isppa:.3e})"
        );
    }

    /// Invalid impedance rejects the ISPPA calculation; nonfinite samples do not
    /// dominate a finite spatial peak.
    #[test]
    fn isppa_rejects_invalid_impedance_and_ignores_nonfinite_pressure() {
        let field = Array3::<f64>::from_shape_vec((2, 1, 1), vec![f64::NAN, 3.0]).expect("shape");
        assert_eq!(calculate_isppa(&field, -1000.0, SOUND_SPEED_WATER_SIM), 0.0);

        let isppa = calculate_isppa(&field, 1000.0, SOUND_SPEED_WATER_SIM);
        let expected = 3.0_f64.powi(2) / (2.0 * 1000.0 * SOUND_SPEED_WATER_SIM);
        assert!((isppa - expected).abs() < 1e-20);
    }
}

//! Pressure-based calculations (intensity, MI, TI)

use ndarray::{Array3, ArrayView3};

/// Calculate intensity field from pressure
#[must_use]
pub fn calculate_intensity(
    pressure_field: ArrayView3<f64>,
    density: f64,
    sound_speed: f64,
) -> Array3<f64> {
    let impedance = density * sound_speed;
    pressure_field.mapv(|p| p.powi(2) / (2.0 * impedance))
}

/// Calculate Mechanical Index (MI)
///
/// MI = P_neg / sqrt(f_c)
/// where P_neg is peak negative pressure in MPa and f_c is center frequency in MHz
#[must_use]
pub fn calculate_mechanical_index(peak_negative_pressure: f64, frequency: f64) -> f64 {
    let p_neg_mpa = peak_negative_pressure.abs() / 1e6;
    let freq_mhz = frequency / 1e6;

    if freq_mhz > 0.0 {
        p_neg_mpa / freq_mhz.sqrt()
    } else {
        0.0
    }
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

    let freq_mhz = frequency / 1e6;
    let absorption_factor = tissue_absorption * freq_mhz;

    (acoustic_power * absorption_factor) / REFERENCE_POWER
}

/// Calculate derated pressure (accounting for tissue attenuation)
#[must_use]
pub fn calculate_derated_pressure(pressure: f64, frequency: f64, depth: f64) -> f64 {
    // FDA derating: 0.3 dB/cm/MHz
    const DERATING_FACTOR: f64 = 0.3; // dB/cm/MHz

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
    let impedance = density * sound_speed;

    let mut max_intensity = 0.0;
    for p in pressure_field {
        let intensity = p.powi(2) / (2.0 * impedance);
        max_intensity = f64::max(max_intensity, intensity);
    }

    max_intensity * duty_cycle
}

/// Calculate spatial peak pulse average intensity (I_SPPA)
#[must_use]
pub fn calculate_isppa(pressure_field: &Array3<f64>, density: f64, sound_speed: f64) -> f64 {
    let impedance = density * sound_speed;

    let mut max_intensity = 0.0;
    for p in pressure_field {
        let intensity = p.powi(2) / (2.0 * impedance);
        max_intensity = f64::max(max_intensity, intensity);
    }

    max_intensity
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    // ── calculate_intensity ───────────────────────────────────────────────────

    /// I = p²/(2ρc). At p=1, ρ=1000, c=1500: I = 1/(2×1500000) = 3.333e-7 W/m².
    #[test]
    fn calculate_intensity_matches_acoustic_intensity_formula() {
        let field = Array3::<f64>::from_elem((2, 2, 2), 1.0_f64);
        let intensity = calculate_intensity(field.view(), 1000.0, 1500.0);
        let expected = 1.0 / (2.0 * 1000.0 * 1500.0);
        for &v in intensity.iter() {
            assert!((v - expected).abs() < 1e-20, "I = p²/(2ρc) (got {v:.3e})");
        }
    }

    /// Zero pressure → zero intensity.
    #[test]
    fn calculate_intensity_zero_for_zero_pressure() {
        let field = Array3::<f64>::zeros((2, 2, 2));
        let intensity = calculate_intensity(field.view(), 1000.0, 1500.0);
        for &v in intensity.iter() {
            assert_eq!(v, 0.0);
        }
    }

    // ── calculate_mechanical_index ────────────────────────────────────────────

    /// MI = |P_neg,MPa| / √f_MHz. At P=0.5 MPa, f=1 MHz: MI = 0.5/1 = 0.5.
    #[test]
    fn mechanical_index_matches_formula_at_half_mpa_one_mhz() {
        let mi = calculate_mechanical_index(0.5e6, 1e6);
        assert!((mi - 0.5).abs() < 1e-12, "MI must be 0.5 (got {mi:.6})");
    }

    /// Zero frequency → MI = 0.0 (guarded branch).
    #[test]
    fn mechanical_index_zero_for_zero_frequency() {
        let mi = calculate_mechanical_index(1e6, 0.0);
        assert_eq!(mi, 0.0);
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

    // ── calculate_ispta ───────────────────────────────────────────────────────

    /// ISPTA = max(p²/(2ρc)) × duty_cycle. Uniform field with p=2: I_max=4/(2ρc).
    #[test]
    fn ispta_equals_peak_intensity_times_duty_cycle() {
        let field = Array3::<f64>::from_elem((2, 2, 2), 2.0_f64);
        let rho = 1000.0_f64;
        let c = 1500.0_f64;
        let duty = 0.1_f64;
        let ispta = calculate_ispta(&field, rho, c, duty);
        let expected = 2.0_f64.powi(2) / (2.0 * rho * c) * duty;
        assert!(
            (ispta - expected).abs() < 1e-20,
            "ISPTA formula (got {ispta:.3e})"
        );
    }

    // ── calculate_isppa ───────────────────────────────────────────────────────

    /// ISPPA = max(p²/(2ρc)). Spike at p=3: ISPPA = 9/(2ρc).
    #[test]
    fn isppa_equals_peak_intensity() {
        let mut field = Array3::<f64>::zeros((4, 4, 4));
        field[[2, 2, 2]] = 3.0;
        let rho = 1000.0_f64;
        let c = 1500.0_f64;
        let isppa = calculate_isppa(&field, rho, c);
        let expected = 3.0_f64.powi(2) / (2.0 * rho * c);
        assert!(
            (isppa - expected).abs() < 1e-20,
            "ISPPA formula (got {isppa:.3e})"
        );
    }
}

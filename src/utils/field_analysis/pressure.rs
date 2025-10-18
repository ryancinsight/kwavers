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
/// MI = `P_neg` / `sqrt(f_c)`
/// where `P_neg` is peak negative pressure in `MPa` and `f_c` is center frequency in `MHz`
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

/// Calculate spatial peak temporal average intensity (`I_SPTA`)
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

/// Calculate spatial peak pulse average intensity (`I_SPPA`)
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

//! Unit conversion helpers shared across Rust crates and PyO3 bindings.

use crate::constants::numerical::TWO_PI;
use crate::error::{KwaversError, KwaversResult};

const NP_PER_DB: f64 = std::f64::consts::LN_10 / 20.0;
const DB_PER_NP: f64 = 20.0 / std::f64::consts::LN_10;
const CENTIMETERS_PER_METER: f64 = 100.0;
const REFERENCE_ANGULAR_FREQUENCY_RAD_S: f64 = TWO_PI * 1.0e6;

/// Convert dB/(MHz^y cm) to Np/(rad/s)^y m.
#[must_use]
pub fn db_per_mhz_cm_to_neper_per_rad_s_m(db: f64, power_law_exponent: f64) -> f64 {
    db * (CENTIMETERS_PER_METER * NP_PER_DB)
        / REFERENCE_ANGULAR_FREQUENCY_RAD_S.powf(power_law_exponent)
}

/// Convert Np/(rad/s)^y m to dB/(MHz^y cm).
#[must_use]
pub fn neper_per_rad_s_m_to_db_per_mhz_cm(neper: f64, power_law_exponent: f64) -> f64 {
    neper
        * (DB_PER_NP / CENTIMETERS_PER_METER)
        * REFERENCE_ANGULAR_FREQUENCY_RAD_S.powf(power_law_exponent)
}

/// Convert temporal frequency in hertz and sound speed in metres per second to
/// wavenumber in radians per metre.
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] when `frequency_hz` is negative,
/// non-finite, or when `sound_speed_m_s` is not finite and positive.
pub fn frequency_to_wavenumber(frequency_hz: f64, sound_speed_m_s: f64) -> KwaversResult<f64> {
    if !sound_speed_m_s.is_finite() || sound_speed_m_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "sound_speed_m_s must be finite and > 0, got {sound_speed_m_s}"
        )));
    }
    if !frequency_hz.is_finite() || frequency_hz < 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "frequency_hz must be finite and >= 0, got {frequency_hz}"
        )));
    }
    Ok(TWO_PI * frequency_hz / sound_speed_m_s)
}

/// Convert k-Wave absorption units dB/(MHz^y cm) to Np/m at `frequency_mhz`.
#[must_use]
pub fn alpha_db_per_mhz_cm_to_np_per_m(
    alpha_db_cm: f64,
    frequency_mhz: f64,
    power_law_exponent: f64,
) -> f64 {
    alpha_db_cm * frequency_mhz.powf(power_law_exponent) * CENTIMETERS_PER_METER / DB_PER_NP
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_neper_round_trip_preserves_value() {
        let value = 0.75;
        let exponent = 1.5;
        let neper = db_per_mhz_cm_to_neper_per_rad_s_m(value, exponent);
        let round_trip = neper_per_rad_s_m_to_db_per_mhz_cm(neper, exponent);
        assert!((round_trip - value).abs() < 1.0e-12);
    }

    #[test]
    fn frequency_to_wavenumber_matches_definition() {
        let got = frequency_to_wavenumber(1.0e6, 1500.0).unwrap();
        let expected = TWO_PI * 1.0e6 / 1500.0;
        assert_eq!(got, expected);
    }

    #[test]
    fn alpha_conversion_respects_power_law_frequency_scaling() {
        let alpha_db_cm = 0.5;
        let at_1mhz = alpha_db_per_mhz_cm_to_np_per_m(alpha_db_cm, 1.0, 1.5);
        let at_2mhz = alpha_db_per_mhz_cm_to_np_per_m(alpha_db_cm, 2.0, 1.5);
        let expected_ratio = 2.0_f64.powf(1.5);
        assert!((at_2mhz / at_1mhz - expected_ratio).abs() < 1.0e-12);
    }
}

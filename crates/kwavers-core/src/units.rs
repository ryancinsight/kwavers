//! Unit conversion helpers shared across Rust crates and PyO3 bindings.
//!
//! Power-law absorption conversions (forward, at-frequency, and inverse forms)
//! are owned by `kwavers_medium::absorption::power_law` beside
//! `PowerLawAbsorption` (KW-POWER-LAW-INVERSE-FORMS); this module keeps only
//! the frequency/wavenumber helper.

use crate::constants::numerical::TWO_PI;
use crate::error::{KwaversError, KwaversResult};

/// Convert temporal frequency in hertz and sound speed in metres per second to
/// wavenumber in radians per metre.
///
/// # Errors
/// Returns `KwaversError::InvalidInput` when `frequency_hz` is negative,
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

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        reason = "test assertions: a failed unwrap is the test failure"
    )]

    use super::*;

    #[test]
    fn frequency_to_wavenumber_matches_definition() {
        let got = frequency_to_wavenumber(1.0e6, 1500.0).unwrap();
        let expected = TWO_PI * 1.0e6 / 1500.0;
        assert_eq!(got, expected);
    }
}

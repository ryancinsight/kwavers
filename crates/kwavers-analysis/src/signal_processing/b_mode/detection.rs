//! Envelope detection and log compression — the amplitude-to-display stages of
//! the B-mode pipeline.
//!
//! After beamforming and TGC, the RF line is **envelope-detected** (the
//! magnitude of its analytic signal, removing the carrier) and then
//! **log-compressed** to fit the wide echo dynamic range into a displayable
//! range. These are the standard final stages before scan conversion.
//!
//! # Reference
//! - Szabo, T. L. (2014). *Diagnostic Ultrasound Imaging: Inside Out* (2nd ed.),
//!   §10.4. Academic Press.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::analytic_signal_1d;
use ndarray::Array1;

/// Envelope of an RF line: the magnitude of its analytic (Hilbert) signal.
///
/// Reuses the workspace analytic-signal SSOT (`kwavers_math::fft`), so the
/// carrier is removed exactly once, consistently with beamforming snapshot code.
#[must_use]
pub fn envelope(rf: &Array1<f64>) -> Array1<f64> {
    analytic_signal_1d(rf).mapv(|z| z.norm())
}

/// Log-compress an envelope to a normalized display image in `[0, 1]`.
///
/// Values are taken relative to the envelope peak: the peak maps to `1.0`, a
/// point `dynamic_range_db` below the peak maps to `0.0`, and anything below the
/// floor is clamped to `0.0`:
///
/// ```text
/// out = clamp( (20·log₁₀(env/env_max) + DR) / DR , 0, 1 ).
/// ```
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] when `dynamic_range_db ≤ 0` or the
/// envelope is empty.
pub fn log_compress(env: &Array1<f64>, dynamic_range_db: f64) -> KwaversResult<Array1<f64>> {
    if dynamic_range_db <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "dynamic_range_db must be positive".to_owned(),
        ));
    }
    if env.is_empty() {
        return Err(KwaversError::InvalidInput("envelope is empty".to_owned()));
    }
    let peak = env.iter().fold(0.0_f64, |m, &v| m.max(v));
    if peak <= 0.0 {
        return Ok(Array1::zeros(env.len()));
    }
    Ok(env.mapv(|v| {
        if v <= 0.0 {
            0.0
        } else {
            let db = 20.0 * (v / peak).log10();
            ((db + dynamic_range_db) / dynamic_range_db).clamp(0.0, 1.0)
        }
    }))
}

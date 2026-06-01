//! Beam pattern calculations for acoustic transducers
//!
//! Implements beam width and directivity calculations

use crate::core::constants::numerical::TWO_PI;
/// Beam pattern calculations
#[derive(Debug)]
pub struct BeamPatterns;

impl BeamPatterns {
    /// Calculate beam width at specified dB level
    #[must_use]
    pub fn beam_width(aperture: f64, wavelength: f64, level_db: f64) -> f64 {
        let factor = match level_db {
            l if l >= -3.0 => 0.88,
            l if l >= -6.0 => 1.02,
            _ => 1.22,
        };
        factor * wavelength / aperture
    }

    /// Calculate directivity pattern
    #[must_use]
    pub fn directivity(theta: f64, aperture: f64, wavelength: f64) -> f64 {
        let k = TWO_PI / wavelength;
        let x = k * aperture * theta.sin() / 2.0;
        if x.abs() < 1e-10 {
            1.0
        } else {
            (x.sin() / x).abs()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── beam_width ────────────────────────────────────────────────────────────

    /// At level_db ≥ -3 dB: beam_width = 0.88 · λ/D.
    #[test]
    fn beam_width_at_minus_3db_uses_factor_0_88() {
        let aperture = 0.01_f64; // 10 mm
        let wavelength = 1.5e-3_f64; // 1.5 mm (1 MHz in water)
        let bw = BeamPatterns::beam_width(aperture, wavelength, -3.0);
        let expected = 0.88 * wavelength / aperture;
        assert!(
            (bw - expected).abs() < 1e-15,
            "bw(-3dB)={bw:.6e}, expected={expected:.6e}"
        );
    }

    /// At level_db ≥ -6 dB and < -3 dB: beam_width = 1.02 · λ/D.
    #[test]
    fn beam_width_at_minus_6db_uses_factor_1_02() {
        let aperture = 0.01_f64;
        let wavelength = 1.5e-3_f64;
        let bw = BeamPatterns::beam_width(aperture, wavelength, -6.0);
        let expected = 1.02 * wavelength / aperture;
        assert!(
            (bw - expected).abs() < 1e-15,
            "bw(-6dB)={bw:.6e}, expected={expected:.6e}"
        );
    }

    /// At level_db < -6 dB: beam_width = 1.22 · λ/D (Rayleigh criterion).
    #[test]
    fn beam_width_below_minus_6db_uses_factor_1_22() {
        let aperture = 0.01_f64;
        let wavelength = 1.5e-3_f64;
        let bw = BeamPatterns::beam_width(aperture, wavelength, -12.0);
        let expected = 1.22 * wavelength / aperture;
        assert!(
            (bw - expected).abs() < 1e-15,
            "bw(-12dB)={bw:.6e}, expected={expected:.6e}"
        );
    }

    // ── directivity ───────────────────────────────────────────────────────────

    /// At θ=0 (on-axis): sinc argument x→0 → directivity = 1.0 (normalised on-axis).
    #[test]
    fn directivity_one_at_zero_angle() {
        let d = BeamPatterns::directivity(0.0, 0.01, 1.5e-3);
        assert!(
            (d - 1.0).abs() < 1e-12,
            "on-axis directivity must be 1.0 (got {d})"
        );
    }

    /// First null of sinc: x = π ↔ k·D·sin(θ)/2 = π ↔ sin(θ) = λ/D.
    ///
    /// At the first null, directivity ≈ 0 (< 0.01).
    #[test]
    fn directivity_near_zero_at_first_null() {
        let aperture = 0.01_f64;
        let wavelength = 1.5e-3_f64;
        // sin(θ_null) = λ/D → θ_null = arcsin(λ/D)
        let theta_null = (wavelength / aperture).asin();
        let d = BeamPatterns::directivity(theta_null, aperture, wavelength);
        assert!(
            d.abs() < 0.01,
            "directivity at first null must be near 0 (got {d:.4}), \
             theta_null={theta_null:.4} rad"
        );
    }

    /// Directivity is bounded in [0, 1] for all physically realizable angles.
    #[test]
    fn directivity_bounded_in_unit_interval() {
        let aperture = 0.01_f64;
        let wavelength = 1.5e-3_f64;
        let angles: Vec<f64> = (0..=90).map(|i| i as f64 * PI / 180.0).collect();
        for &theta in &angles {
            let d = BeamPatterns::directivity(theta, aperture, wavelength);
            assert!(
                d >= 0.0 && d <= 1.0 + 1e-12,
                "directivity={d:.4} out of [0,1] at theta={theta:.3} rad"
            );
        }
    }
}

//! Beam pattern calculations for acoustic transducers
//!
//! Implements beam width and directivity calculations


/// Beam pattern calculations
#[derive(Debug)]
pub struct BeamPatterns;

impl BeamPatterns {
    /// Calculate beam width at specified dB level
    pub fn beam_width(aperture: f64, wavelength: f64, level_db: f64) -> f64 {
        let factor = match level_db {
            l if l >= -3.0 => 0.88,
            l if l >= -6.0 => 1.02,
            _ => 1.22,
        };
        factor * wavelength / aperture
    }

    /// Calculate directivity pattern
    pub fn directivity(theta: f64, aperture: f64, wavelength: f64) -> f64 {
        use std::f64::consts::PI;
        let k = 2.0 * PI / wavelength;
        let x = k * aperture * theta.sin() / 2.0;
        if x.abs() < 1e-10 {
            1.0
        } else {
            (x.sin() / x).abs()
        }
    }
}
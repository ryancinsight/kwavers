//! Wavelength range for spectral analysis

use ndarray::Array1;
use crate::core::constants::fundamental::SPEED_OF_LIGHT;

/// Wavelength range for spectral analysis
#[derive(Debug, Clone)]
pub struct SpectralRange {
    /// Minimum wavelength in meters
    pub lambda_min: f64,
    /// Maximum wavelength in meters
    pub lambda_max: f64,
    /// Number of wavelength points
    pub n_points: usize,
}

impl Default for SpectralRange {
    fn default() -> Self {
        Self {
            lambda_min: 200e-9, // 200 nm (UV)
            lambda_max: 800e-9, // 800 nm (near IR)
            n_points: 300,
        }
    }
}

impl SpectralRange {
    /// Generate wavelength array
    #[must_use]
    pub fn wavelengths(&self) -> Array1<f64> {
        Array1::linspace(self.lambda_min, self.lambda_max, self.n_points)
    }

    /// Generate frequency array
    #[must_use]
    pub fn frequencies(&self) -> Array1<f64> {
        self.wavelengths().mapv(|lambda| SPEED_OF_LIGHT / lambda)
    }

    /// Convert wavelength to RGB color
    #[must_use]
    pub fn wavelength_to_rgb(wavelength: f64) -> (f64, f64, f64) {
        let w = wavelength * 1e9; // Convert to nm

        // Pre-computed wavelength-to-RGB lookup table: (wavelength_nm, R, G, B)
        const RGB_TABLE: &[(f64, f64, f64, f64)] = &[
            (380.0, 0.0, 0.0, 0.0), // UV
            (440.0, 0.0, 0.0, 1.0),
            (490.0, 0.0, 1.0, 1.0),
            (510.0, 0.0, 1.0, 0.0),
            (580.0, 1.0, 1.0, 0.0),
            (645.0, 1.0, 0.0, 0.0),
            (780.0, 0.0, 0.0, 0.0), // IR
        ];

        // Find the two closest points in the table
        let (mut r, mut g, mut b) = (0.0, 0.0, 0.0);
        for i in 0..RGB_TABLE.len() - 1 {
            if w >= RGB_TABLE[i].0 && w <= RGB_TABLE[i + 1].0 {
                let t = (w - RGB_TABLE[i].0) / (RGB_TABLE[i + 1].0 - RGB_TABLE[i].0);
                r = RGB_TABLE[i].1 + t * (RGB_TABLE[i + 1].1 - RGB_TABLE[i].1);
                g = RGB_TABLE[i].2 + t * (RGB_TABLE[i + 1].2 - RGB_TABLE[i].2);
                b = RGB_TABLE[i].3 + t * (RGB_TABLE[i + 1].3 - RGB_TABLE[i].3);
                break;
            }
        }

        // Apply intensity correction for eye sensitivity
        let intensity = if w < 420.0 {
            0.3 + 0.7 * (w - 380.0) / 40.0
        } else if w > 700.0 {
            0.3 + 0.7 * (780.0 - w) / 80.0
        } else {
            1.0
        };

        (r * intensity, g * intensity, b * intensity)
    }
}

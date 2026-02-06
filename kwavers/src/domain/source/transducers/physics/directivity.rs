//! Directivity Pattern Module
//!
//! Models the spatial radiation pattern of transducer elements.

use ndarray::Array1;
use std::f64::consts::PI;

/// Directivity pattern of a transducer element
///
/// Based on Rayleigh-Sommerfeld diffraction theory
/// Reference: Kino (1987) Chapter 4 - Radiation and Reception
#[derive(Debug, Clone)]
pub struct DirectivityPattern {
    /// Angle vector (degrees)
    pub angles: Array1<f64>,
    /// Normalized pressure amplitude
    pub amplitude: Array1<f64>,
    /// -3 dB beamwidth (degrees)
    pub beamwidth_3db: f64,
    /// -6 dB beamwidth (degrees)
    pub beamwidth_6db: f64,
    /// First sidelobe level (dB)
    pub sidelobe_level: f64,
}

impl DirectivityPattern {
    /// Calculate directivity for rectangular element
    ///
    /// Uses sinc function model for rectangular aperture
    #[must_use]
    pub fn rectangular_element(width: f64, height: f64, frequency: f64, num_points: usize) -> Self {
        let wavelength = 1540.0 / frequency; // Assume tissue
        let k = 2.0 * PI / wavelength;

        let angles = Array1::linspace(-90.0, 90.0, num_points);
        let mut amplitude = Array1::zeros(num_points);

        for (i, &angle_deg) in angles.iter().enumerate() {
            let angle_rad = angle_deg * PI / 180.0;
            let sin_theta = angle_rad.sin();

            // Directivity function for rectangular aperture
            let arg_x = k * width * sin_theta / 2.0;
            let arg_y = k * height * sin_theta / 2.0;

            let dir_x = if arg_x.abs() < 1e-10 {
                1.0
            } else {
                arg_x.sin() / arg_x
            };

            let dir_y = if arg_y.abs() < 1e-10 {
                1.0
            } else {
                arg_y.sin() / arg_y
            };

            amplitude[i] = (dir_x * dir_y).abs();
        }

        // Find beamwidth
        let (beamwidth_3db, beamwidth_6db) = Self::find_beamwidth(&angles, &amplitude);

        // Find sidelobe level
        let sidelobe_level = Self::find_sidelobe_level(&amplitude);

        Self {
            angles,
            amplitude,
            beamwidth_3db,
            beamwidth_6db,
            sidelobe_level,
        }
    }

    /// Calculate directivity for circular element
    #[must_use]
    pub fn circular_element(diameter: f64, frequency: f64, num_points: usize) -> Self {
        let wavelength = 1540.0 / frequency;
        let k = 2.0 * PI / wavelength;
        let radius = diameter / 2.0;

        let angles = Array1::linspace(-90.0, 90.0, num_points);
        let mut amplitude = Array1::zeros(num_points);

        for (i, &angle_deg) in angles.iter().enumerate() {
            let angle_rad = angle_deg * PI / 180.0;
            let sin_theta = angle_rad.sin();

            let arg = k * radius * sin_theta;

            // Directivity function for circular aperture (Bessel function)
            let dir = if arg.abs() < 1e-10 {
                1.0
            } else {
                2.0 * Self::bessel_j1(arg) / arg
            };

            amplitude[i] = dir.abs();
        }

        let (beamwidth_3db, beamwidth_6db) = Self::find_beamwidth(&angles, &amplitude);
        let sidelobe_level = Self::find_sidelobe_level(&amplitude);

        Self {
            angles,
            amplitude,
            beamwidth_3db,
            beamwidth_6db,
            sidelobe_level,
        }
    }

    /// Find beamwidth at -3dB and -6dB levels
    fn find_beamwidth(angles: &Array1<f64>, amplitude: &Array1<f64>) -> (f64, f64) {
        let threshold_3db = 1.0 / 2.0_f64.sqrt();
        let threshold_6db = 0.5;

        let center_idx = angles.len() / 2;
        let mut width_3db = 180.0;
        let mut width_6db = 180.0;

        // Search from center outward
        for i in center_idx..angles.len() {
            if amplitude[i] < threshold_3db && width_3db == 180.0 {
                width_3db = 2.0 * angles[i].abs();
            }
            if amplitude[i] < threshold_6db && width_6db == 180.0 {
                width_6db = 2.0 * angles[i].abs();
                break;
            }
        }

        (width_3db, width_6db)
    }

    /// Find first sidelobe level in dB
    fn find_sidelobe_level(amplitude: &Array1<f64>) -> f64 {
        let center_idx = amplitude.len() / 2;
        let mut in_null = false;
        let mut max_sidelobe = 0.0;

        for i in center_idx..amplitude.len() {
            if !in_null && amplitude[i] < amplitude[i.saturating_sub(1)] {
                in_null = true;
            }
            if in_null && amplitude[i] > amplitude[i.saturating_sub(1)] {
                max_sidelobe = amplitude[i];
                break;
            }
        }

        if max_sidelobe > 0.0 {
            20.0 * max_sidelobe.log10()
        } else {
            -60.0 // No sidelobe found
        }
    }

    /// Bessel function of first kind, order 1 (approximation)
    fn bessel_j1(x: f64) -> f64 {
        if x.abs() < 3.0 {
            // Series expansion for small x
            let x2 = x * x;
            x / 2.0 * (1.0 - x2 / 8.0 + x2 * x2 / 192.0)
        } else {
            // Asymptotic expansion for large x
            let inv_x = 1.0 / x;
            let phase = x - 3.0 * PI / 4.0;
            (2.0 / (PI * x)).sqrt() * phase.cos() * (1.0 - 0.1875 * inv_x * inv_x)
        }
    }

    /// Calculate beam divergence angle
    #[must_use]
    pub fn divergence_angle(&self) -> f64 {
        self.beamwidth_3db
    }

    /// Calculate directivity index (DI)
    #[must_use]
    pub fn directivity_index(&self) -> f64 {
        // DI = 10 * log10(4π / Ω)
        // where Ω is the beam solid angle
        let beam_solid_angle = 2.0 * PI * (1.0 - (self.beamwidth_3db * PI / 180.0 / 2.0).cos());
        10.0 * (4.0 * PI / beam_solid_angle).log10()
    }
}

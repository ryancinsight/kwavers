//! Emission spectrum data models

use ndarray::{s, Array1};

/// Emission spectrum data
#[derive(Debug, Clone)]
pub struct EmissionSpectrum {
    /// Wavelengths in meters
    pub wavelengths: Array1<f64>,
    /// Spectral intensities in arbitrary units
    pub intensities: Array1<f64>,
    /// Time stamp
    pub time: f64,
    /// Spatial position (i, j, k)
    pub position: Option<(usize, usize, usize)>,
}

impl EmissionSpectrum {
    /// Create new emission spectrum
    #[must_use]
    pub fn new(wavelengths: Array1<f64>, intensities: Array1<f64>, time: f64) -> Self {
        assert_eq!(
            wavelengths.len(),
            intensities.len(),
            "Wavelength and intensity arrays must have same length"
        );
        Self {
            wavelengths,
            intensities,
            time,
            position: None,
        }
    }

    /// Calculate total integrated intensity
    #[must_use]
    pub fn total_intensity(&self) -> f64 {
        // Vectorized trapezoidal integration
        let dlambda = &self.wavelengths.slice(s![1..]) - &self.wavelengths.slice(s![..-1]);
        let avg_intensity =
            0.5 * (&self.intensities.slice(s![1..]) + &self.intensities.slice(s![..-1]));
        (avg_intensity * dlambda).sum()
    }

    /// Find peak wavelength
    #[must_use]
    pub fn peak_wavelength(&self) -> f64 {
        let max_idx = self
            .intensities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map_or(0, |(idx, _)| idx);
        self.wavelengths[max_idx]
    }

    /// Calculate centroid wavelength
    #[must_use]
    pub fn centroid_wavelength(&self) -> f64 {
        let mut sum_lambda_i = 0.0;
        let mut sum_i = 0.0;

        for i in 0..self.wavelengths.len() {
            sum_lambda_i += self.wavelengths[i] * self.intensities[i];
            sum_i += self.intensities[i];
        }

        if sum_i > 0.0 {
            sum_lambda_i / sum_i
        } else {
            0.0
        }
    }

    /// Calculate full width at half maximum (FWHM)
    #[must_use]
    pub fn fwhm(&self) -> f64 {
        let (max_idx, &max_val) = self
            .intensities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        let half_max = max_val / 2.0;

        // Find left half-maximum point
        let mut left_idx = max_idx;
        for i in (0..max_idx).rev() {
            if self.intensities[i] < half_max {
                left_idx = i;
                break;
            }
        }

        // Find right half-maximum point
        let mut right_idx = max_idx;
        for i in max_idx..self.wavelengths.len() {
            if self.intensities[i] < half_max {
                right_idx = i;
                break;
            }
        }

        if right_idx > left_idx {
            self.wavelengths[right_idx] - self.wavelengths[left_idx]
        } else {
            0.0
        }
    }
}

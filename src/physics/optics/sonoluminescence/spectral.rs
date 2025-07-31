//! Spectral analysis for sonoluminescence
//!
//! Tools for analyzing emission spectra and extracting physical parameters

use ndarray::{Array1, s};

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
            lambda_min: 200e-9,  // 200 nm (UV)
            lambda_max: 800e-9,  // 800 nm (near IR)
            n_points: 300,
        }
    }
}

impl SpectralRange {
    /// Generate wavelength array
    pub fn wavelengths(&self) -> Array1<f64> {
        Array1::linspace(self.lambda_min, self.lambda_max, self.n_points)
    }
    
    /// Generate frequency array
    pub fn frequencies(&self) -> Array1<f64> {
        let c = 2.99792458e8; // Speed of light
        self.wavelengths().mapv(|lambda| c / lambda)
    }
    
    /// Convert wavelength to RGB color
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
    pub fn new(wavelengths: Array1<f64>, intensities: Array1<f64>, time: f64) -> Self {
        assert_eq!(wavelengths.len(), intensities.len(), "Wavelength and intensity arrays must have same length");
        Self {
            wavelengths,
            intensities,
            time,
            position: None,
        }
    }
    
    /// Calculate total integrated intensity
    pub fn total_intensity(&self) -> f64 {
        // Vectorized trapezoidal integration
        let dlambda = &self.wavelengths.slice(s![1..]) - &self.wavelengths.slice(s![..-1]);
        let avg_intensity = 0.5 * (&self.intensities.slice(s![1..]) + &self.intensities.slice(s![..-1]));
        (avg_intensity * dlambda).sum()
    }
    
    /// Find peak wavelength
    pub fn peak_wavelength(&self) -> f64 {
        let max_idx = self.intensities.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        self.wavelengths[max_idx]
    }
    
    /// Calculate centroid wavelength
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
    pub fn fwhm(&self) -> f64 {
        let (max_idx, &max_val) = self.intensities.iter()
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

/// Spectral analyzer for sonoluminescence
#[derive(Debug)]
pub struct SpectralAnalyzer {
    /// Spectral range
    pub range: SpectralRange,
    /// Time history of spectra
    pub spectra_history: Vec<EmissionSpectrum>,
    /// Maximum number of spectra to store
    pub max_history: usize,
}

impl SpectralAnalyzer {
    /// Create new spectral analyzer
    pub fn new(range: SpectralRange) -> Self {
        Self {
            range,
            spectra_history: Vec::new(),
            max_history: 1000,
        }
    }
    
    /// Add spectrum to history
    pub fn add_spectrum(&mut self, spectrum: EmissionSpectrum) {
        self.spectra_history.push(spectrum);
        
        // Limit history size
        if self.spectra_history.len() > self.max_history {
            self.spectra_history.remove(0);
        }
    }
    
    /// Get time evolution of peak wavelength
    pub fn peak_wavelength_evolution(&self) -> (Array1<f64>, Array1<f64>) {
        let n = self.spectra_history.len();
        let mut times = Array1::zeros(n);
        let mut peaks = Array1::zeros(n);
        
        for (i, spectrum) in self.spectra_history.iter().enumerate() {
            times[i] = spectrum.time;
            peaks[i] = spectrum.peak_wavelength();
        }
        
        (times, peaks)
    }
    
    /// Get time evolution of total intensity
    pub fn intensity_evolution(&self) -> (Array1<f64>, Array1<f64>) {
        let n = self.spectra_history.len();
        let mut times = Array1::zeros(n);
        let mut intensities = Array1::zeros(n);
        
        for (i, spectrum) in self.spectra_history.iter().enumerate() {
            times[i] = spectrum.time;
            intensities[i] = spectrum.total_intensity();
        }
        
        (times, intensities)
    }
    
    /// Calculate time-averaged spectrum
    pub fn time_averaged_spectrum(&self) -> Option<EmissionSpectrum> {
        if self.spectra_history.is_empty() {
            return None;
        }
        
        let wavelengths = self.spectra_history[0].wavelengths.clone();
        let mut avg_intensities = Array1::zeros(wavelengths.len());
        
        for spectrum in &self.spectra_history {
            avg_intensities = avg_intensities + &spectrum.intensities;
        }
        
        avg_intensities /= self.spectra_history.len() as f64;
        
        Some(EmissionSpectrum::new(
            wavelengths,
            avg_intensities,
            self.spectra_history.last().unwrap().time,
        ))
    }
    
    /// Fit blackbody temperature to spectrum
    pub fn fit_blackbody_temperature(&self, spectrum: &EmissionSpectrum) -> f64 {
        // Use Wien's displacement law on peak wavelength
        let peak = spectrum.peak_wavelength();
        if peak > 0.0 {
            2.897771955e-3 / peak // Wien's constant / peak wavelength
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spectral_range() {
        let range = SpectralRange::default();
        let wavelengths = range.wavelengths();
        
        assert_eq!(wavelengths.len(), range.n_points);
        assert_eq!(wavelengths[0], range.lambda_min);
        assert_eq!(wavelengths[wavelengths.len()-1], range.lambda_max);
    }
    
    #[test]
    fn test_emission_spectrum() {
        let wavelengths = Array1::linspace(400e-9, 700e-9, 100);
        let intensities = wavelengths.mapv(|lambda| {
            // Gaussian peak at 550 nm
            let center: f64 = 550e-9;
            let sigma: f64 = 50e-9;
            (-(lambda - center).powi(2) / (2.0 * sigma.powi(2))).exp()
        });
        
        let spectrum = EmissionSpectrum::new(wavelengths, intensities, 0.0);
        
        let peak = spectrum.peak_wavelength();
        assert!((peak - 550e-9).abs() < 10e-9); // Peak should be near 550 nm
        
        let fwhm = spectrum.fwhm();
        assert!(fwhm > 0.0 && fwhm < 200e-9); // FWHM should be reasonable
    }
    
    #[test]
    fn test_wavelength_to_rgb() {
        // Test some known wavelengths
        let (r, g, b) = SpectralRange::wavelength_to_rgb(700e-9); // Red
        assert!(r > 0.9 && g < 0.1 && b < 0.1);
        
        let (r, g, b) = SpectralRange::wavelength_to_rgb(520e-9); // Green (use 520nm for purer green)
        assert!(r < 0.3 && g > 0.9 && b < 0.1);
        
        let (r, g, b) = SpectralRange::wavelength_to_rgb(450e-9); // Blue
        assert!(r < 0.1 && g < 0.5 && b > 0.9);
    }
}
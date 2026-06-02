//! Types and data structures for harmonic detection

use super::config::HarmonicDetectionConfig;
use ndarray::Array3;

/// Multi-frequency displacement field with harmonic components
#[derive(Debug, Clone)]
pub struct HarmonicDisplacementField {
    /// Fundamental frequency displacement magnitude
    pub fundamental_magnitude: Array3<f64>,
    /// Fundamental frequency displacement phase
    pub fundamental_phase: Array3<f64>,
    /// Harmonic displacement magnitudes (vector of arrays for each harmonic)
    pub harmonic_magnitudes: Vec<Array3<f64>>,
    /// Harmonic displacement phases (vector of arrays for each harmonic)
    pub harmonic_phases: Vec<Array3<f64>>,
    /// Signal-to-noise ratios for each harmonic (dB)
    pub harmonic_snrs: Vec<Array3<f64>>,
    /// Nonlinearity parameter B/A estimates
    pub nonlinearity_parameter: Array3<f64>,
    /// Time vector for the analysis
    pub time: Vec<f64>,
    /// Frequency vector for spectral analysis
    pub frequency: Vec<f64>,
}

impl HarmonicDisplacementField {
    /// Create new harmonic displacement field
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, n_harmonics: usize, n_time_points: usize) -> Self {
        let mut harmonic_magnitudes = Vec::with_capacity(n_harmonics);
        let mut harmonic_phases = Vec::with_capacity(n_harmonics);
        let mut harmonic_snrs = Vec::with_capacity(n_harmonics);

        for _ in 0..n_harmonics {
            harmonic_magnitudes.push(Array3::zeros((nx, ny, nz)));
            harmonic_phases.push(Array3::zeros((nx, ny, nz)));
            harmonic_snrs.push(Array3::zeros((nx, ny, nz)));
        }

        Self {
            fundamental_magnitude: Array3::zeros((nx, ny, nz)),
            fundamental_phase: Array3::zeros((nx, ny, nz)),
            harmonic_magnitudes,
            harmonic_phases,
            harmonic_snrs,
            nonlinearity_parameter: Array3::zeros((nx, ny, nz)),
            time: vec![0.0; n_time_points],
            frequency: vec![0.0; n_time_points / 2 + 1], // FFT frequency bins
        }
    }

    /// Get harmonic ratio (A₂/A₁) for nonlinearity estimation
    #[must_use]
    pub fn harmonic_ratio(&self, harmonic_order: usize) -> Array3<f64> {
        // The stored `harmonic_magnitudes` exclude the fundamental and start at the second harmonic.
        // Therefore: harmonic_order=2 -> index 0, harmonic_order=3 -> index 1, etc.
        if harmonic_order < 2 {
            return Array3::zeros(self.fundamental_magnitude.dim());
        }

        let idx = harmonic_order - 2;
        if idx >= self.harmonic_magnitudes.len() {
            return Array3::zeros(self.fundamental_magnitude.dim());
        }

        &self.harmonic_magnitudes[idx] / &self.fundamental_magnitude
    }

    /// Compute local nonlinearity parameter map from the second-harmonic displacement ratio.
    ///
    /// # Theory
    ///
    /// For a weakly nonlinear viscoelastic solid driven at frequency ω₀, the
    /// second harmonic amplitude A₂ is related to the cubic nonlinearity
    /// parameter Γ (Destrade & Ogden 2010, *Proc R Soc A* 466:3474):
    ///
    /// ```text
    /// A₂/A₁ ≈ Γ · A₁ / (8 G')      (small-strain approximation)
    /// ```
    ///
    /// where G' (Pa) is the storage modulus. Solving for Γ requires knowing G'
    /// and A₁ (absolute amplitude), which are not available without additional
    /// calibration data. This function stores the dimensionless harmonic ratio
    /// A₂/A₁ as a relative nonlinearity proxy pending external calibration.
    ///
    /// To obtain absolute Γ values, multiply by `8 G' / A₁` using G' from the
    /// linear SWE reconstruction and A₁ from displacement field amplitude data.
    ///
    /// # Reference
    ///
    /// Destrade M & Ogden RW (2010). "On the third- and fourth-order constants
    /// of incompressible isotropic elasticity." *J Acoust Soc Am* 128(6):3334–3343.
    pub fn compute_nonlinearity_parameter(&mut self, _config: &HarmonicDetectionConfig) {
        // Store the dimensionless second-harmonic displacement ratio A₂/A₁.
        // This is a relative nonlinearity proxy; absolute Γ requires G' and A₁.
        self.nonlinearity_parameter = self.harmonic_ratio(2);
    }
}

/// Harmonic analysis results for a single spatial point
#[derive(Debug)]
pub(crate) struct PointHarmonics {
    /// Fundamental frequency magnitude
    pub fundamental_magnitude: f64,
    /// Fundamental frequency phase
    pub fundamental_phase: f64,
    /// Harmonic magnitudes (excluding fundamental)
    pub harmonic_magnitudes: Vec<f64>,
    /// Harmonic phases (excluding fundamental)
    pub harmonic_phases: Vec<f64>,
    /// Harmonic SNRs in dB
    pub harmonic_snrs: Vec<f64>,
}

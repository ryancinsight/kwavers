//! Types and data structures for harmonic detection

use leto::Array3;

/// Multi-frequency displacement field with harmonic components
#[derive(Debug, Clone)]
pub struct HarmonicDisplacementField {
    /// Fundamental frequency displacement magnitude
    pub fundamental_magnitude: Array3<f64>,
    /// Principal fundamental phase in radians, in the interval [-π, π].
    pub fundamental_phase: Array3<f64>,
    /// Magnitudes for requested harmonics above the fundamental, starting at A₂
    pub harmonic_magnitudes: Vec<Array3<f64>>,
    /// Principal phases in radians for harmonics above the fundamental, starting at A₂.
    pub harmonic_phases: Vec<Array3<f64>>,
    /// Reported signal-to-noise ratios for harmonics above the fundamental, starting at A₂ (dB).
    ///
    /// These values describe the result and do not filter harmonic output.
    pub harmonic_snrs: Vec<Array3<f64>>,
    /// Nonlinearity parameter B/A estimates
    pub nonlinearity_parameter: Array3<f64>,
    /// Sample times for the complete analyzed record.
    pub time: Vec<f64>,
    /// Non-negative FFT-bin frequencies for the complete analyzed record.
    pub frequency: Vec<f64>,
}

impl HarmonicDisplacementField {
    /// Create a harmonic displacement field.
    ///
    /// `higher_harmonic_count` counts stored harmonics above the separately
    /// represented fundamental frequency.
    #[must_use]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        higher_harmonic_count: usize,
        n_time_points: usize,
    ) -> Self {
        let mut harmonic_magnitudes = Vec::with_capacity(higher_harmonic_count);
        let mut harmonic_phases = Vec::with_capacity(higher_harmonic_count);
        let mut harmonic_snrs = Vec::with_capacity(higher_harmonic_count);

        for _ in 0..higher_harmonic_count {
            harmonic_magnitudes.push(Array3::zeros([nx, ny, nz]));
            harmonic_phases.push(Array3::zeros([nx, ny, nz]));
            harmonic_snrs.push(Array3::zeros([nx, ny, nz]));
        }

        Self {
            fundamental_magnitude: Array3::zeros([nx, ny, nz]),
            fundamental_phase: Array3::zeros([nx, ny, nz]),
            harmonic_magnitudes,
            harmonic_phases,
            harmonic_snrs,
            nonlinearity_parameter: Array3::zeros([nx, ny, nz]),
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
            return Array3::zeros(self.fundamental_magnitude.shape());
        }

        let idx = harmonic_order - 2;
        if idx >= self.harmonic_magnitudes.len() {
            return Array3::zeros(self.fundamental_magnitude.shape());
        }

        {
            let mut result = Array3::zeros(self.fundamental_magnitude.shape());
            let [nx, ny, nz] = result.shape();
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        result[[i, j, k]] = self.harmonic_magnitudes[idx][[i, j, k]]
                            / self.fundamental_magnitude[[i, j, k]];
                    }
                }
            }
            result
        }
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
    ///
    /// # Panics
    ///
    /// Panics when a caller has replaced the public second-harmonic array with
    /// one whose shape differs from the fundamental array.
    pub fn compute_nonlinearity_parameter(&mut self) {
        // Store the dimensionless second-harmonic displacement ratio A₂/A₁.
        // This is a relative nonlinearity proxy; absolute Γ requires G' and A₁.
        if self.nonlinearity_parameter.shape() != self.fundamental_magnitude.shape() {
            self.nonlinearity_parameter = Array3::zeros(self.fundamental_magnitude.shape());
        }
        let Some(second_harmonic) = self.harmonic_magnitudes.first() else {
            self.nonlinearity_parameter.fill(0.0);
            return;
        };
        let [nx, ny, nz] = self.fundamental_magnitude.shape();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    self.nonlinearity_parameter[[i, j, k]] =
                        second_harmonic[[i, j, k]] / self.fundamental_magnitude[[i, j, k]];
                }
            }
        }
    }
}

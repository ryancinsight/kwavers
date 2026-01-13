//! Nonlinear elastic wave field state representation
//!
//! This module defines the wave field structure for nonlinear shear wave elastography,
//! including fundamental frequency components, harmonic components, and field operations.
//!
//! ## Wave Field Components
//!
//! The nonlinear wave field consists of:
//! 1. **Fundamental frequency** - Primary displacement field at f₀
//! 2. **Second harmonic** - Displacement field at 2f₀ (quadratic nonlinearity)
//! 3. **Higher harmonics** - Displacement fields at nf₀ (cascading nonlinearity)
//!
//! ## Theoretical Foundation
//!
//! For nonlinear elastic waves with quadratic nonlinearity, the solution can be
//! expanded as a perturbation series:
//!
//! u(x,t) = u₁(x,t) + ε u₂(x,t) + ε² u₃(x,t) + ...
//!
//! where ε is a small parameter related to the nonlinearity strength β.
//!
//! Each harmonic satisfies:
//! - Fundamental: ∂²u₁/∂t² = c²∇²u₁
//! - Second harmonic: ∂²u₂/∂t² = c²∇²u₂ + β u₁ ∇²u₁
//! - Third harmonic: ∂²u₃/∂t² = c²∇²u₃ + β(u₁∇²u₂ + u₂∇²u₁ + 2∇u₁·∇u₂)
//!
//! ## Literature References
//!
//! - Chen, S., et al. (2013). "Harmonic motion detection in ultrasound elastography."
//!   IEEE Transactions on Medical Imaging, 32(5), 863-874.
//! - Hamilton, M. F., & Blackstock, D. T. (1998). "Nonlinear Acoustics", Academic Press.

use ndarray::Array3;

/// Nonlinear elastic wave field with harmonic components
///
/// # Theorem Reference
/// The total displacement field for nonlinear elastic waves can be decomposed as:
/// u_total(x,t) = Σₙ uₙ(x) exp(i n ω t)
/// where ω is the fundamental angular frequency and uₙ are the harmonic amplitudes.
///
/// For weakly nonlinear waves with quadratic nonlinearity (β parameter):
/// - Fundamental amplitude: u₁ ~ A₀
/// - Second harmonic: u₂ ~ β A₀²
/// - Third harmonic: u₃ ~ β² A₀³
///
/// This structure stores the real-space amplitudes at each harmonic frequency.
///
/// # Fields
/// - `u_fundamental`: Primary displacement field at fundamental frequency f₀
/// - `u_fundamental_prev`: Previous time step for time integration
/// - `u_second`: Second harmonic displacement at 2f₀
/// - `u_harmonics`: Higher harmonic displacements at 3f₀, 4f₀, ...
/// - `time`: Current simulation time
/// - `frequency`: Fundamental frequency in Hz
#[derive(Debug, Clone)]
pub struct NonlinearElasticWaveField {
    /// Fundamental frequency displacement (m)
    pub u_fundamental: Array3<f64>,
    /// Previous time step displacement for time integration (m)
    pub u_fundamental_prev: Array3<f64>,
    /// Second harmonic displacement (m)
    pub u_second: Array3<f64>,
    /// Higher harmonic displacements (m)
    /// Index i corresponds to (i+3)th harmonic
    pub u_harmonics: Vec<Array3<f64>>,
    /// Current time (s)
    pub time: f64,
    /// Fundamental frequency (Hz)
    pub frequency: f64,
}

impl NonlinearElasticWaveField {
    /// Create new nonlinear wave field
    ///
    /// # Arguments
    /// * `nx` - Grid points in x direction
    /// * `ny` - Grid points in y direction
    /// * `nz` - Grid points in z direction
    /// * `n_harmonics` - Total number of harmonics (including fundamental)
    ///
    /// # Returns
    /// Initialized wave field with zero displacement
    ///
    /// # Example
    /// ```ignore
    /// // Create field for 64x64x64 grid with fundamental + 2 harmonics
    /// let field = NonlinearElasticWaveField::new(64, 64, 64, 3);
    /// ```
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, n_harmonics: usize) -> Self {
        Self {
            u_fundamental: Array3::zeros((nx, ny, nz)),
            u_fundamental_prev: Array3::zeros((nx, ny, nz)),
            u_second: Array3::zeros((nx, ny, nz)),
            u_harmonics: vec![Array3::zeros((nx, ny, nz)); n_harmonics.saturating_sub(2)],
            time: 0.0,
            frequency: 50.0, // 50 Hz typical for SWE
        }
    }

    /// Compute total displacement magnitude including all harmonics
    ///
    /// # Theorem Reference
    /// The total displacement magnitude is computed as:
    /// |u_total| = √(|u₁|² + |u₂|² + |u₃|² + ...)
    ///
    /// This represents the RMS displacement from all frequency components.
    /// For weakly nonlinear waves (β << 1), u₁ >> u₂ >> u₃, so the total
    /// magnitude is dominated by the fundamental frequency.
    ///
    /// # Returns
    /// 3D array of total displacement magnitudes in meters
    ///
    /// # Example
    /// ```ignore
    /// let magnitude = field.total_displacement_magnitude();
    /// let max_displacement = magnitude.iter().fold(0.0, |m, &x| m.max(x));
    /// ```
    #[must_use]
    pub fn total_displacement_magnitude(&self) -> Array3<f64> {
        let mut total = &self.u_fundamental * &self.u_fundamental + &self.u_second * &self.u_second;

        for harmonic in &self.u_harmonics {
            total = &total + &(harmonic * harmonic);
        }

        total.mapv(f64::sqrt)
    }

    /// Get the number of harmonics being tracked
    ///
    /// # Returns
    /// Total number of harmonics including fundamental (1 + 1 + n_higher)
    #[must_use]
    pub fn num_harmonics(&self) -> usize {
        2 + self.u_harmonics.len() // fundamental + second + higher
    }

    /// Get the displacement at a specific harmonic
    ///
    /// # Arguments
    /// * `harmonic_index` - Harmonic number (1 = fundamental, 2 = second, 3 = third, ...)
    ///
    /// # Returns
    /// Reference to the displacement array for the specified harmonic
    ///
    /// # Panics
    /// Panics if harmonic_index is 0 or exceeds the number of tracked harmonics
    #[must_use]
    pub fn get_harmonic(&self, harmonic_index: usize) -> &Array3<f64> {
        match harmonic_index {
            1 => &self.u_fundamental,
            2 => &self.u_second,
            n if n <= self.num_harmonics() => &self.u_harmonics[n - 3],
            _ => panic!(
                "Harmonic index {} out of range [1, {}]",
                harmonic_index,
                self.num_harmonics()
            ),
        }
    }

    /// Get mutable reference to displacement at a specific harmonic
    ///
    /// # Arguments
    /// * `harmonic_index` - Harmonic number (1 = fundamental, 2 = second, 3 = third, ...)
    ///
    /// # Returns
    /// Mutable reference to the displacement array for the specified harmonic
    ///
    /// # Panics
    /// Panics if harmonic_index is 0 or exceeds the number of tracked harmonics
    pub fn get_harmonic_mut(&mut self, harmonic_index: usize) -> &mut Array3<f64> {
        match harmonic_index {
            1 => &mut self.u_fundamental,
            2 => &mut self.u_second,
            n if n <= self.num_harmonics() => &mut self.u_harmonics[n - 3],
            _ => panic!(
                "Harmonic index {} out of range [1, {}]",
                harmonic_index,
                self.num_harmonics()
            ),
        }
    }

    /// Compute harmonic content spectrum at a specific point
    ///
    /// # Arguments
    /// * `i` - x grid index
    /// * `j` - y grid index
    /// * `k` - z grid index
    ///
    /// # Returns
    /// Vector of displacement amplitudes [u₁, u₂, u₃, ...] at the specified point
    #[must_use]
    pub fn harmonic_spectrum(&self, i: usize, j: usize, k: usize) -> Vec<f64> {
        let mut spectrum = vec![self.u_fundamental[[i, j, k]], self.u_second[[i, j, k]]];
        for harmonic in &self.u_harmonics {
            spectrum.push(harmonic[[i, j, k]]);
        }
        spectrum
    }

    /// Compute nonlinearity parameter from harmonic ratio
    ///
    /// # Theorem Reference
    /// The nonlinearity parameter β can be estimated from the ratio of harmonic amplitudes:
    /// β ≈ u₂ / (u₁²/u_ref)
    ///
    /// where u_ref is a reference displacement scale. This is derived from the
    /// perturbation expansion of the nonlinear wave equation.
    ///
    /// Reference: Hamilton & Blackstock (1998), "Nonlinear Acoustics", Chapter 3.
    ///
    /// # Arguments
    /// * `i` - x grid index
    /// * `j` - y grid index
    /// * `k` - z grid index
    /// * `u_ref` - Reference displacement scale (m)
    ///
    /// # Returns
    /// Estimated nonlinearity parameter (dimensionless)
    #[must_use]
    pub fn estimate_nonlinearity(&self, i: usize, j: usize, k: usize, u_ref: f64) -> f64 {
        let u1 = self.u_fundamental[[i, j, k]].abs();
        let u2 = self.u_second[[i, j, k]].abs();

        if u1 < f64::EPSILON {
            return 0.0;
        }

        u2 / (u1 * u1 / u_ref)
    }

    /// Reset all fields to zero
    pub fn reset(&mut self) {
        self.u_fundamental.fill(0.0);
        self.u_fundamental_prev.fill(0.0);
        self.u_second.fill(0.0);
        for harmonic in &mut self.u_harmonics {
            harmonic.fill(0.0);
        }
        self.time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wave_field_creation() {
        let field = NonlinearElasticWaveField::new(10, 10, 10, 3);

        assert_eq!(field.u_fundamental.dim(), (10, 10, 10));
        assert_eq!(field.u_second.dim(), (10, 10, 10));
        assert_eq!(field.u_harmonics.len(), 1); // 3 total - 2 = 1 additional
        assert_eq!(field.num_harmonics(), 3);
        assert_eq!(field.time, 0.0);
        assert_eq!(field.frequency, 50.0);
    }

    #[test]
    fn test_total_displacement_magnitude() {
        let field = NonlinearElasticWaveField::new(10, 10, 10, 3);

        let magnitude = field.total_displacement_magnitude();
        assert_eq!(magnitude.dim(), (10, 10, 10));

        // All values should be zero initially
        for &val in magnitude.iter() {
            assert!((val - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_get_harmonic() {
        let mut field = NonlinearElasticWaveField::new(10, 10, 10, 4);

        // Set some values
        field.u_fundamental[[0, 0, 0]] = 1.0;
        field.u_second[[0, 0, 0]] = 0.5;
        field.u_harmonics[0][[0, 0, 0]] = 0.25;

        assert_eq!(field.get_harmonic(1)[[0, 0, 0]], 1.0);
        assert_eq!(field.get_harmonic(2)[[0, 0, 0]], 0.5);
        assert_eq!(field.get_harmonic(3)[[0, 0, 0]], 0.25);
    }

    #[test]
    fn test_harmonic_spectrum() {
        let mut field = NonlinearElasticWaveField::new(10, 10, 10, 3);

        field.u_fundamental[[5, 5, 5]] = 1.0;
        field.u_second[[5, 5, 5]] = 0.1;
        field.u_harmonics[0][[5, 5, 5]] = 0.01;

        let spectrum = field.harmonic_spectrum(5, 5, 5);
        assert_eq!(spectrum.len(), 3);
        assert_eq!(spectrum[0], 1.0);
        assert_eq!(spectrum[1], 0.1);
        assert_eq!(spectrum[2], 0.01);
    }

    #[test]
    fn test_estimate_nonlinearity() {
        let mut field = NonlinearElasticWaveField::new(10, 10, 10, 3);

        field.u_fundamental[[5, 5, 5]] = 1e-3;
        field.u_second[[5, 5, 5]] = 1e-7;

        let u_ref = 1e-3;
        let beta = field.estimate_nonlinearity(5, 5, 5, u_ref);

        // β ≈ u₂ / (u₁²/u_ref) = 1e-7 / ((1e-3)²/1e-3) = 1e-7 / 1e-3 = 1e-4
        assert!((beta - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_reset() {
        let mut field = NonlinearElasticWaveField::new(10, 10, 10, 3);

        // Set some values
        field.u_fundamental.fill(1.0);
        field.u_second.fill(0.5);
        field.time = 1.0;

        // Reset
        field.reset();

        // Check all zeros
        for &val in field.u_fundamental.iter() {
            assert_eq!(val, 0.0);
        }
        for &val in field.u_second.iter() {
            assert_eq!(val, 0.0);
        }
        assert_eq!(field.time, 0.0);
    }

    #[test]
    fn test_get_harmonic_mut() {
        let mut field = NonlinearElasticWaveField::new(10, 10, 10, 3);

        {
            let h1 = field.get_harmonic_mut(1);
            h1[[0, 0, 0]] = 2.0;
        }

        assert_eq!(field.u_fundamental[[0, 0, 0]], 2.0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_get_harmonic_invalid_index() {
        let field = NonlinearElasticWaveField::new(10, 10, 10, 3);
        let _ = field.get_harmonic(10); // Should panic
    }

    #[test]
    fn test_num_harmonics() {
        let field2 = NonlinearElasticWaveField::new(10, 10, 10, 2);
        assert_eq!(field2.num_harmonics(), 2);

        let field5 = NonlinearElasticWaveField::new(10, 10, 10, 5);
        assert_eq!(field5.num_harmonics(), 5);
    }
}

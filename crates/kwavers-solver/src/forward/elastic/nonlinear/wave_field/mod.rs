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

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

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
        2 + (self.u_harmonics.shape()[0] * self.u_harmonics.shape()[1] * self.u_harmonics.shape()[2]) // fundamental + second + higher
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
    /// # Panics
    /// Panics if harmonic_index is 0 or exceeds the number of tracked harmonics
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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

    /// Fallible version of [`get_harmonic`](Self::get_harmonic).
    ///
    /// Returns `Err(InvalidInput)` instead of panicking when the index is out of range.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn try_get_harmonic(&self, harmonic_index: usize) -> KwaversResult<&Array3<f64>> {
        match harmonic_index {
            1 => Ok(&self.u_fundamental),
            2 => Ok(&self.u_second),
            n if n <= self.num_harmonics() => Ok(&self.u_harmonics[n - 3]),
            _ => Err(KwaversError::InvalidInput(format!(
                "Harmonic index {} out of range [1, {}]",
                harmonic_index,
                self.num_harmonics()
            ))),
        }
    }

    /// Fallible version of [`get_harmonic_mut`](Self::get_harmonic_mut).
    ///
    /// Returns `Err(InvalidInput)` instead of panicking when the index is out of range.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn try_get_harmonic_mut(
        &mut self,
        harmonic_index: usize,
    ) -> KwaversResult<&mut Array3<f64>> {
        let n_harmonics = self.num_harmonics();
        match harmonic_index {
            1 => Ok(&mut self.u_fundamental),
            2 => Ok(&mut self.u_second),
            n if n <= n_harmonics => Ok(&mut self.u_harmonics[n - 3]),
            _ => Err(KwaversError::InvalidInput(format!(
                "Harmonic index {} out of range [1, {}]",
                harmonic_index, n_harmonics
            ))),
        }
    }

    /// Compute harmonic content spectrum at a specific point
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
    /// Reference: Hamilton & Blackstock (1998), "Nonlinear Acoustics", Chapter 3.
    ///
    /// # Arguments
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
mod tests;

//! Analytical Solutions for k-Wave Comparison and Validation
//!
//! This module provides exact analytical solutions to the acoustic wave equation
//! for validation of numerical solvers against k-Wave and mathematical ground truth.
//!
//! # Mathematical Specifications
//!
//! All analytical solutions are derived from first principles and validated against
//! published literature. Each solution includes:
//! - Governing equations with references
//! - Domain of validity and assumptions
//! - Error metrics for numerical comparison
//!
//! # References
//!
//! 1. Pierce, A. D. (1989). *Acoustics: An Introduction to Its Physical Principles
//!    and Applications*. Acoustical Society of America.
//! 2. Kinsler, L. E., et al. (2000). *Fundamentals of Acoustics* (4th ed.). Wiley.
//! 3. Goodman, J. W. (2005). *Introduction to Fourier Optics* (3rd ed.).
//!    Roberts and Company Publishers. (Gaussian beam theory)
//! 4. Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for the simulation
//!    and reconstruction of photoacoustic wave fields." *J. Biomed. Opt.*, 15(2), 021314.
//!
//! # Author
//!
//! Ryan Clanton (@ryancinsight)
//! Sprint 217 Session 8 - k-Wave Comparison Framework

use ndarray::{Array3, ArrayView3};
use std::f64::consts::PI;

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;

/// Analytical solution for plane wave propagation.
///
/// # Mathematical Specification
///
/// Plane wave solution to the linear acoustic wave equation:
///
/// ```text
/// p(x, t) = A sin(k·x - ωt + φ)                    (1)
///
/// where:
///   k = ω/c₀ = 2πf/c₀  [rad/m]  Wave number       (2)
///   ω = 2πf             [rad/s]  Angular frequency  (3)
///   A                   [Pa]     Amplitude
///   φ                   [rad]    Phase offset
///   c₀                  [m/s]    Sound speed
/// ```
///
/// **Dispersion Relation**: ω² = c₀²k² (exact, no dispersion)
///
/// **Energy Conservation**: Total energy density constant in lossless medium:
/// ```text
/// E = ½(p²/(ρ₀c₀²) + ρ₀|u|²)                       (4)
/// ```
///
/// # Validation Criteria
///
/// - Phase velocity: |c_numerical - c₀|/c₀ < 0.001 (0.1% error)
/// - Amplitude decay: |A_final/A_initial - 1| < 0.001 (no dissipation)
/// - Dispersion error: L2 norm < 0.01
///
/// # References
///
/// - Pierce (1989), Ch. 1: Plane wave solutions
/// - Treeby & Cox (2010): k-Wave validation cases
#[derive(Debug, Clone)]
pub struct PlaneWave {
    /// Wave amplitude [Pa]
    pub amplitude: f64,
    /// Frequency [Hz]
    pub frequency: f64,
    /// Sound speed [m/s]
    pub sound_speed: f64,
    /// Wave propagation direction (unit vector)
    pub direction: [f64; 3],
    /// Phase offset [rad]
    pub phase: f64,
}

impl PlaneWave {
    /// Create new plane wave analytical solution.
    ///
    /// # Arguments
    ///
    /// * `amplitude` - Pressure amplitude [Pa]
    /// * `frequency` - Wave frequency [Hz]
    /// * `sound_speed` - Medium sound speed [m/s]
    /// * `direction` - Propagation direction (will be normalized)
    /// * `phase` - Initial phase offset [rad]
    ///
    /// # Returns
    ///
    /// Plane wave solution instance
    ///
    /// # Mathematical Validation
    ///
    /// - Validates f > 0, c₀ > 0, |direction| > 0
    /// - Normalizes direction to unit vector
    pub fn new(
        amplitude: f64,
        frequency: f64,
        sound_speed: f64,
        direction: [f64; 3],
        phase: f64,
    ) -> KwaversResult<Self> {
        if frequency <= 0.0 {
            return Err(KwaversError::Validation(
                crate::core::error::validation::ValidationError::ConstraintViolation {
                    message: "Frequency must be positive".to_string(),
                },
            ));
        }
        if sound_speed <= 0.0 {
            return Err(KwaversError::Validation(
                crate::core::error::validation::ValidationError::ConstraintViolation {
                    message: "Sound speed must be positive".to_string(),
                },
            ));
        }

        // Normalize direction vector
        let norm = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2)).sqrt();
        if norm < 1e-10 {
            return Err(KwaversError::Validation(
                crate::core::error::validation::ValidationError::ConstraintViolation {
                    message: "Direction vector must be non-zero".to_string(),
                },
            ));
        }

        let direction = [
            direction[0] / norm,
            direction[1] / norm,
            direction[2] / norm,
        ];

        Ok(Self {
            amplitude,
            frequency,
            sound_speed,
            direction,
            phase,
        })
    }

    /// Evaluate plane wave pressure at given position and time.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// p(x, t) = A sin(k·x - ωt + φ)
    /// k·x = k_x·x + k_y·y + k_z·z
    /// ```
    pub fn pressure(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        let omega = 2.0 * PI * self.frequency;
        let k = omega / self.sound_speed;

        let k_dot_x = k * (self.direction[0] * x + self.direction[1] * y + self.direction[2] * z);
        let phase = k_dot_x - omega * t + self.phase;

        self.amplitude * phase.sin()
    }

    /// Evaluate plane wave on 3D grid at given time.
    ///
    /// # Returns
    ///
    /// 3D array of pressure values [Pa]
    pub fn pressure_field(&self, grid: &Grid, t: f64) -> Array3<f64> {
        let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    field[[i, j, k]] = self.pressure(x, y, z, t);
                }
            }
        }

        field
    }

    /// Wave number k = 2πf/c₀ [rad/m]
    pub fn wave_number(&self) -> f64 {
        2.0 * PI * self.frequency / self.sound_speed
    }

    /// Wavelength λ = c₀/f [m]
    pub fn wavelength(&self) -> f64 {
        self.sound_speed / self.frequency
    }

    /// Angular frequency ω = 2πf [rad/s]
    pub fn angular_frequency(&self) -> f64 {
        2.0 * PI * self.frequency
    }
}

/// Analytical solution for Gaussian beam propagation.
///
/// # Mathematical Specification (Goodman 2005, Ch. 3)
///
/// Paraxial approximation for Gaussian beam:
///
/// ```text
/// p(r, z, t) = A₀(w₀/w(z)) exp(-r²/w(z)²) exp(i(kz - ωt + φ(z)))  (5)
///
/// where:
///   w(z) = w₀√(1 + (z/z_R)²)    Beam width                         (6)
///   z_R = πw₀²/λ                 Rayleigh range                     (7)
///   φ(z) = arctan(z/z_R)         Gouy phase                         (8)
///   r = √(x² + y²)               Radial coordinate                  (9)
/// ```
///
/// **Validity**: Paraxial approximation valid for w₀ >> λ (typically w₀ > 3λ)
///
/// **Focus Properties**:
/// - Minimum beam width at z = 0: w(0) = w₀
/// - Beam width at Rayleigh range: w(z_R) = √2·w₀
/// - Far-field divergence angle: θ ≈ λ/(πw₀)
///
/// # Validation Criteria
///
/// - Beam width at z_R: |w_numerical - √2·w₀|/(√2·w₀) < 0.01 (1%)
/// - Focal intensity: 0.95 < I_numerical/I_analytical < 1.05
/// - Gouy phase: |φ_numerical - φ_analytical| < π/20
///
/// # References
///
/// - Goodman (2005), Ch. 3: Gaussian beam propagation
/// - Siegman, A. E. (1986). *Lasers*. University Science Books.
#[derive(Debug, Clone)]
pub struct GaussianBeam {
    /// Peak amplitude [Pa]
    pub amplitude: f64,
    /// Frequency [Hz]
    pub frequency: f64,
    /// Sound speed [m/s]
    pub sound_speed: f64,
    /// Beam waist radius (1/e² intensity) [m]
    pub waist_radius: f64,
    /// Focal position [m]
    pub focal_z: f64,
}

impl GaussianBeam {
    /// Create new Gaussian beam analytical solution.
    ///
    /// # Paraxial Validity Check
    ///
    /// Ensures w₀ > 3λ for paraxial approximation validity.
    pub fn new(
        amplitude: f64,
        frequency: f64,
        sound_speed: f64,
        waist_radius: f64,
        focal_z: f64,
    ) -> KwaversResult<Self> {
        if frequency <= 0.0 {
            return Err(KwaversError::Validation(
                crate::core::error::validation::ValidationError::ConstraintViolation {
                    message: "Frequency must be positive".to_string(),
                },
            ));
        }
        if sound_speed <= 0.0 {
            return Err(KwaversError::Validation(
                crate::core::error::validation::ValidationError::ConstraintViolation {
                    message: "Sound speed must be positive".to_string(),
                },
            ));
        }
        if waist_radius <= 0.0 {
            return Err(KwaversError::Validation(
                crate::core::error::validation::ValidationError::ConstraintViolation {
                    message: "Waist radius must be positive".to_string(),
                },
            ));
        }

        // Check paraxial approximation validity
        let wavelength = sound_speed / frequency;
        if waist_radius < 3.0 * wavelength {
            return Err(KwaversError::Validation(
                crate::core::error::validation::ValidationError::ConstraintViolation {
                    message: format!(
                        "Paraxial approximation requires w₀ > 3λ. Got w₀={:.3e}m, λ={:.3e}m",
                        waist_radius, wavelength
                    ),
                },
            ));
        }

        Ok(Self {
            amplitude,
            frequency,
            sound_speed,
            waist_radius,
            focal_z,
        })
    }

    /// Rayleigh range z_R = πw₀²/λ [m]
    pub fn rayleigh_range(&self) -> f64 {
        let wavelength = self.sound_speed / self.frequency;
        PI * self.waist_radius.powi(2) / wavelength
    }

    /// Beam width at distance z: w(z) = w₀√(1 + (z/z_R)²) [m]
    pub fn beam_width(&self, z: f64) -> f64 {
        let z_rel = z - self.focal_z;
        let z_r = self.rayleigh_range();
        self.waist_radius * (1.0 + (z_rel / z_r).powi(2)).sqrt()
    }

    /// Gouy phase shift: φ(z) = arctan(z/z_R) [rad]
    pub fn gouy_phase(&self, z: f64) -> f64 {
        let z_rel = z - self.focal_z;
        let z_r = self.rayleigh_range();
        (z_rel / z_r).atan()
    }

    /// Evaluate Gaussian beam pressure (real part) at given position and time.
    ///
    /// # Note
    ///
    /// Returns real part of complex amplitude (sin component for pressure).
    pub fn pressure(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        let r = (x.powi(2) + y.powi(2)).sqrt();
        let z_rel = z - self.focal_z;

        let w_z = self.beam_width(z);
        let amplitude_factor = self.waist_radius / w_z;
        let gaussian_envelope = (-r.powi(2) / w_z.powi(2)).exp();

        let k = 2.0 * PI * self.frequency / self.sound_speed;
        let omega = 2.0 * PI * self.frequency;
        let gouy = self.gouy_phase(z);

        let phase = k * z_rel - omega * t + gouy;

        self.amplitude * amplitude_factor * gaussian_envelope * phase.sin()
    }

    /// Evaluate Gaussian beam on 3D grid at given time.
    pub fn pressure_field(&self, grid: &Grid, t: f64) -> Array3<f64> {
        let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    field[[i, j, k]] = self.pressure(x, y, z, t);
                }
            }
        }

        field
    }
}

/// Analytical solution for spherical wave from point source.
///
/// # Mathematical Specification (Pierce 1989, Ch. 4)
///
/// Spherical wave from point source at origin:
///
/// ```text
/// p(r, t) = (A/r) sin(kr - ωt + φ)    for r > 0              (10)
///
/// where:
///   r = √(x² + y² + z²)                Distance from source   (11)
///   A                                  Source strength [Pa·m]
/// ```
///
/// **Geometric Spreading**: Amplitude decays as 1/r (energy conservation)
///
/// **Energy Conservation**: Integrating over spherical shell:
/// ```text
/// ∫_shell p² r² dΩ = constant         (Surface integral)     (12)
/// ```
///
/// **Singularity**: Solution singular at r=0 (requires regularization in numerics)
///
/// # Validation Criteria
///
/// - Geometric spreading: |(p·r)/A - 1| < 0.01
/// - Wavefront curvature: |∇²p/(k²p) - 1| < 0.02
/// - Energy conservation: |E(r₂)/E(r₁) - (r₁/r₂)²| < 0.01
///
/// # References
///
/// - Pierce (1989), Ch. 4: Spherical waves
/// - Morse, P. M., & Ingard, K. U. (1968). *Theoretical Acoustics*. McGraw-Hill.
#[derive(Debug, Clone)]
pub struct SphericalWave {
    /// Source strength A [Pa·m]
    pub source_strength: f64,
    /// Frequency [Hz]
    pub frequency: f64,
    /// Sound speed [m/s]
    pub sound_speed: f64,
    /// Source position [m]
    pub source_position: [f64; 3],
    /// Phase offset [rad]
    pub phase: f64,
}

impl SphericalWave {
    /// Create new spherical wave analytical solution.
    pub fn new(
        source_strength: f64,
        frequency: f64,
        sound_speed: f64,
        source_position: [f64; 3],
        phase: f64,
    ) -> KwaversResult<Self> {
        if frequency <= 0.0 {
            return Err(KwaversError::Validation(
                crate::core::error::validation::ValidationError::ConstraintViolation {
                    message: "Frequency must be positive".to_string(),
                },
            ));
        }
        if sound_speed <= 0.0 {
            return Err(KwaversError::Validation(
                crate::core::error::validation::ValidationError::ConstraintViolation {
                    message: "Sound speed must be positive".to_string(),
                },
            ));
        }

        Ok(Self {
            source_strength,
            frequency,
            sound_speed,
            source_position,
            phase,
        })
    }

    /// Evaluate spherical wave pressure at given position and time.
    ///
    /// # Singularity Handling
    ///
    /// Returns 0 for r < ε (small threshold to avoid division by zero).
    pub fn pressure(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        let dx = x - self.source_position[0];
        let dy = y - self.source_position[1];
        let dz = z - self.source_position[2];
        let r = (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt();

        // Regularization: avoid singularity at source
        const EPSILON: f64 = 1e-10;
        if r < EPSILON {
            return 0.0;
        }

        let k = 2.0 * PI * self.frequency / self.sound_speed;
        let omega = 2.0 * PI * self.frequency;
        let phase = k * r - omega * t + self.phase;

        (self.source_strength / r) * phase.sin()
    }

    /// Evaluate spherical wave on 3D grid at given time.
    pub fn pressure_field(&self, grid: &Grid, t: f64) -> Array3<f64> {
        let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    field[[i, j, k]] = self.pressure(x, y, z, t);
                }
            }
        }

        field
    }
}

/// Error metrics for comparing numerical and analytical solutions.
///
/// # Mathematical Definitions
///
/// ```text
/// L² error:    ε₂ = √(∫(p_num - p_ana)² dV / ∫p_ana² dV)      (13)
/// L∞ error:    ε∞ = max|p_num - p_ana| / max|p_ana|           (14)
/// Phase error: Δφ = acos(∫p_num·p_ana dV / √(∫p_num²·∫p_ana²)) (15)
/// ```
///
/// # Acceptance Criteria (k-Wave Baseline)
///
/// - L² error < 0.01 (1%)
/// - L∞ error < 0.05 (5%)
/// - Phase error < 0.1 rad (5.7°)
#[derive(Debug, Clone)]
pub struct ErrorMetrics {
    /// L² (RMS) relative error
    pub l2_error: f64,
    /// L∞ (maximum) relative error
    pub linf_error: f64,
    /// Phase error [rad]
    pub phase_error: f64,
    /// Amplitude correlation coefficient
    pub correlation: f64,
}

impl ErrorMetrics {
    /// Compute error metrics between numerical and analytical solutions.
    ///
    /// # Arguments
    ///
    /// * `numerical` - Numerical solution pressure field
    /// * `analytical` - Analytical solution pressure field
    ///
    /// # Returns
    ///
    /// Error metrics structure
    ///
    /// # Mathematical Implementation
    ///
    /// Uses discrete approximations of continuous integrals (13-15).
    pub fn compute(numerical: ArrayView3<f64>, analytical: ArrayView3<f64>) -> Self {
        assert_eq!(
            numerical.dim(),
            analytical.dim(),
            "Arrays must have same dimensions"
        );

        let mut l2_num = 0.0f64;
        let mut l2_ana = 0.0f64;
        let mut linf_error = 0.0f64;
        let mut correlation_num = 0.0f64;

        let ana_max = analytical
            .iter()
            .map(|x| x.abs())
            .fold(0.0f64, |a, b| a.max(b));

        for (n, a) in numerical.iter().zip(analytical.iter()) {
            let diff = n - a;
            l2_num += diff * diff;
            l2_ana += a * a;
            correlation_num += n * a;

            let local_error = diff.abs() / ana_max.max(1e-10);
            linf_error = linf_error.max(local_error);
        }

        let l2_error = if l2_ana > 0.0 {
            (l2_num / l2_ana).sqrt()
        } else {
            0.0
        };

        // Compute correlation (normalized dot product)
        let num_norm = numerical.iter().map(|x| x * x).sum::<f64>().sqrt();
        let ana_norm = analytical.iter().map(|x| x * x).sum::<f64>().sqrt();
        let correlation = if num_norm > 0.0 && ana_norm > 0.0 {
            correlation_num / (num_norm * ana_norm)
        } else {
            0.0
        };

        // Phase error from correlation
        let phase_error = if correlation.abs() <= 1.0 {
            correlation.acos()
        } else {
            0.0 // Perfect correlation (within numerical precision)
        };

        Self {
            l2_error,
            linf_error,
            phase_error,
            correlation,
        }
    }

    /// Check if errors meet acceptance criteria.
    ///
    /// # Acceptance Thresholds (k-Wave Standard)
    ///
    /// - L² < 0.01 (1%)
    /// - L∞ < 0.05 (5%)
    /// - Phase < 0.1 rad (5.7°)
    pub fn meets_acceptance_criteria(&self) -> bool {
        self.l2_error < 0.01 && self.linf_error < 0.05 && self.phase_error < 0.1
    }

    /// Generate validation report string.
    pub fn report(&self) -> String {
        format!(
            "Error Metrics:\n\
             - L² error:    {:.4e} ({})\n\
             - L∞ error:    {:.4e} ({})\n\
             - Phase error: {:.4} rad = {:.2}° ({})\n\
             - Correlation: {:.6} ({})\n\
             Overall: {}",
            self.l2_error,
            if self.l2_error < 0.01 { "✓" } else { "✗" },
            self.linf_error,
            if self.linf_error < 0.05 { "✓" } else { "✗" },
            self.phase_error,
            self.phase_error.to_degrees(),
            if self.phase_error < 0.1 { "✓" } else { "✗" },
            self.correlation,
            if self.correlation > 0.99 {
                "✓"
            } else {
                "✗"
            },
            if self.meets_acceptance_criteria() {
                "PASS ✓"
            } else {
                "FAIL ✗"
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_wave_creation() {
        let wave = PlaneWave::new(1e5, 1e6, 1500.0, [1.0, 0.0, 0.0], 0.0).unwrap();
        assert_eq!(wave.amplitude, 1e5);
        assert_eq!(wave.frequency, 1e6);

        // Wavelength λ = c/f = 1500/1e6 = 1.5 mm
        assert!((wave.wavelength() - 1.5e-3).abs() < 1e-10);
    }

    #[test]
    fn test_plane_wave_direction_normalization() {
        let wave = PlaneWave::new(1e5, 1e6, 1500.0, [3.0, 4.0, 0.0], 0.0).unwrap();
        let norm =
            (wave.direction[0].powi(2) + wave.direction[1].powi(2) + wave.direction[2].powi(2))
                .sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_plane_wave_pressure_temporal_periodicity() {
        let wave = PlaneWave::new(1e5, 1e6, 1500.0, [1.0, 0.0, 0.0], 0.0).unwrap();
        let period = 1.0 / wave.frequency;

        let p1 = wave.pressure(0.0, 0.0, 0.0, 0.0);
        let p2 = wave.pressure(0.0, 0.0, 0.0, period);

        // Use relative tolerance for floating point comparison
        // sin(0) and sin(2π) should be identical, but with floating point arithmetic
        // we need to account for accumulated round-off error in the calculation
        let relative_error = (p1 - p2).abs() / wave.amplitude.max(1e-12);
        assert!(
            relative_error < 1e-12,
            "Temporal periodicity violated: p(t=0)={}, p(t=T)={}, relative_error={}",
            p1,
            p2,
            relative_error
        );
    }

    #[test]
    fn test_gaussian_beam_paraxial_check() {
        // Should fail: w₀ < 3λ
        let result = GaussianBeam::new(1e5, 1e6, 1500.0, 1e-3, 0.0);
        assert!(result.is_err());

        // Should succeed: w₀ > 3λ
        let result = GaussianBeam::new(1e5, 1e6, 1500.0, 5e-3, 0.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gaussian_beam_rayleigh_range() {
        let beam = GaussianBeam::new(1e5, 2e6, 1500.0, 5e-3, 0.0).unwrap();
        let wavelength = beam.sound_speed / beam.frequency;
        let z_r = PI * beam.waist_radius.powi(2) / wavelength;

        assert!((beam.rayleigh_range() - z_r).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_beam_width_at_rayleigh() {
        let beam = GaussianBeam::new(1e5, 2e6, 1500.0, 5e-3, 0.0).unwrap();
        let z_r = beam.rayleigh_range();
        let w_at_zr = beam.beam_width(z_r);

        // At Rayleigh range: w(z_R) = √2·w₀
        assert!((w_at_zr / beam.waist_radius - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_spherical_wave_geometric_spreading() {
        let wave = SphericalWave::new(1e3, 1e6, 1500.0, [0.0, 0.0, 0.0], 0.0).unwrap();

        let r1 = 0.01; // 1 cm
        let r2 = 0.02; // 2 cm
        let t = 0.0;

        let p1 = wave.pressure(r1, 0.0, 0.0, t);
        let p2 = wave.pressure(r2, 0.0, 0.0, t);

        // Amplitude should decay as 1/r (ignoring phase)
        let ratio = (p1 * r1).abs() / (p2 * r2).abs();
        assert!((ratio - 1.0).abs() < 0.1); // 10% tolerance for phase effects
    }

    #[test]
    fn test_error_metrics_perfect_match() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let wave = PlaneWave::new(1e5, 1e6, 1500.0, [1.0, 0.0, 0.0], 0.0).unwrap();

        let field1 = wave.pressure_field(&grid, 0.0);
        let field2 = wave.pressure_field(&grid, 0.0);

        let metrics = ErrorMetrics::compute(field1.view(), field2.view());

        assert!(metrics.l2_error < 1e-10);
        assert!(metrics.linf_error < 1e-10);
        assert!(metrics.phase_error < 1e-10);
        assert!((metrics.correlation - 1.0).abs() < 1e-10);
        assert!(metrics.meets_acceptance_criteria());
    }

    #[test]
    fn test_error_metrics_phase_shifted() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let wave1 = PlaneWave::new(1e5, 1e6, 1500.0, [1.0, 0.0, 0.0], 0.0).unwrap();
        let wave2 = PlaneWave::new(1e5, 1e6, 1500.0, [1.0, 0.0, 0.0], PI / 4.0).unwrap();

        let field1 = wave1.pressure_field(&grid, 0.0);
        let field2 = wave2.pressure_field(&grid, 0.0);

        let metrics = ErrorMetrics::compute(field1.view(), field2.view());

        // Should detect phase difference
        assert!(metrics.phase_error > 0.1);
        assert!(!metrics.meets_acceptance_criteria());
    }
}

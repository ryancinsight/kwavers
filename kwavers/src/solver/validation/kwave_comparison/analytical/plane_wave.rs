use ndarray::Array3;
use std::f64::consts::PI;

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;

/// Analytical solution for plane wave propagation.
///
/// # Mathematical Specification
///
/// ```text
/// p(x, t) = A sin(k·x - ωt + φ)
/// k = ω/c₀ = 2πf/c₀  [rad/m]
/// ω = 2πf             [rad/s]
/// ```
///
/// # References
///
/// - Pierce (1989), Ch. 1: Plane wave solutions
/// - Treeby & Cox (2010): k-Wave validation cases
#[derive(Debug, Clone)]
pub struct PlaneWave {
    pub amplitude: f64,
    pub frequency: f64,
    pub sound_speed: f64,
    pub direction: [f64; 3],
    pub phase: f64,
}

impl PlaneWave {
    /// Create new plane wave. Validates f > 0, c₀ > 0, normalizes direction.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
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
                    message: "Frequency must be positive".to_owned(),
                },
            ));
        }
        if sound_speed <= 0.0 {
            return Err(KwaversError::Validation(
                crate::core::error::validation::ValidationError::ConstraintViolation {
                    message: "Sound speed must be positive".to_owned(),
                },
            ));
        }

        let norm = direction[2].mul_add(direction[2], direction[1].mul_add(direction[1], direction[0].powi(2))).sqrt();
        if norm < 1e-10 {
            return Err(KwaversError::Validation(
                crate::core::error::validation::ValidationError::ConstraintViolation {
                    message: "Direction vector must be non-zero".to_owned(),
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

    /// p(x, t) = A sin(k·x − ωt + φ)
    #[must_use] 
    pub fn pressure(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        let omega = 2.0 * PI * self.frequency;
        let k = omega / self.sound_speed;
        let k_dot_x = k * self.direction[2].mul_add(z, self.direction[0].mul_add(x, self.direction[1] * y));
        self.amplitude * (k_dot_x - omega * t + self.phase).sin()
    }

    /// Evaluate on 3D grid at time t.
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
    #[must_use] 
    pub fn wave_number(&self) -> f64 {
        2.0 * PI * self.frequency / self.sound_speed
    }

    /// Wavelength λ = c₀/f (m)
    #[must_use] 
    pub fn wavelength(&self) -> f64 {
        self.sound_speed / self.frequency
    }

    /// Angular frequency ω = 2πf [rad/s]
    #[must_use] 
    pub fn angular_frequency(&self) -> f64 {
        2.0 * PI * self.frequency
    }
}

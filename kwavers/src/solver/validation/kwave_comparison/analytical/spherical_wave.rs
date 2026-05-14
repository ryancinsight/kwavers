use ndarray::Array3;
use std::f64::consts::PI;

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;

/// Analytical solution for spherical wave from a point source.
///
/// # Mathematical Specification (Pierce 1989, Ch. 4)
///
/// ```text
/// p(r, t) = (A/r) sin(kr - ωt + φ)    for r > 0
/// r = √(x² + y² + z²)
/// ```
///
/// Amplitude decays as 1/r (geometric spreading, energy conservation).
/// Singular at r = 0; regularized by returning 0 for r < ε.
#[derive(Debug, Clone)]
pub struct SphericalWave {
    pub source_strength: f64,
    pub frequency: f64,
    pub sound_speed: f64,
    pub source_position: [f64; 3],
    pub phase: f64,
}

impl SphericalWave {
    /// Create new spherical wave. Validates f > 0 and c₀ > 0.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
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

        Ok(Self {
            source_strength,
            frequency,
            sound_speed,
            source_position,
            phase,
        })
    }

    /// p(r, t) = (A/r) sin(kr − ωt + φ); returns 0 for r < 1e-10.
    #[must_use]
    pub fn pressure(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        let dx = x - self.source_position[0];
        let dy = y - self.source_position[1];
        let dz = z - self.source_position[2];
        let r = dz.mul_add(dz, dy.mul_add(dy, dx.powi(2))).sqrt();

        const EPSILON: f64 = 1e-10;
        if r < EPSILON {
            return 0.0;
        }

        let k = 2.0 * PI * self.frequency / self.sound_speed;
        let omega = 2.0 * PI * self.frequency;
        let phase = k * r - omega * t + self.phase;

        (self.source_strength / r) * phase.sin()
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
}

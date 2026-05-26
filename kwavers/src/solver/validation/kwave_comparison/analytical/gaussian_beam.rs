use ndarray::Array3;
use std::f64::consts::PI;

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::core::constants::numerical::{TWO_PI};

/// Analytical solution for Gaussian beam propagation (paraxial approximation).
///
/// # Mathematical Specification (Goodman 2005, Ch. 3)
///
/// ```text
/// p(r, z, t) = A₀(w₀/w(z)) exp(-r²/w(z)²) exp(i(kz - ωt + φ(z)))
/// w(z) = w₀√(1 + (z/z_R)²)    z_R = πw₀²/λ
/// φ(z) = arctan(z/z_R)         r = √(x² + y²)
/// ```
///
/// Validity: paraxial approximation requires w₀ > 3λ.
#[derive(Debug, Clone)]
pub struct GaussianBeam {
    pub amplitude: f64,
    pub frequency: f64,
    pub sound_speed: f64,
    pub waist_radius: f64,
    pub focal_z: f64,
}

impl GaussianBeam {
    /// Create new Gaussian beam. Validates f > 0, c₀ > 0, w₀ > 3λ.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
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
        if waist_radius <= 0.0 {
            return Err(KwaversError::Validation(
                crate::core::error::validation::ValidationError::ConstraintViolation {
                    message: "Waist radius must be positive".to_owned(),
                },
            ));
        }

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

    /// Rayleigh range z_R = πw₀²/λ (m)
    #[must_use]
    pub fn rayleigh_range(&self) -> f64 {
        let wavelength = self.sound_speed / self.frequency;
        PI * self.waist_radius.powi(2) / wavelength
    }

    /// Beam width w(z) = w₀√(1 + (z/z_R)²) (m)
    #[must_use]
    pub fn beam_width(&self, z: f64) -> f64 {
        let z_rel = z - self.focal_z;
        let z_r = self.rayleigh_range();
        self.waist_radius * (z_rel / z_r).mul_add(z_rel / z_r, 1.0).sqrt()
    }

    /// Gouy phase φ(z) = arctan(z/z_R) (rad)
    #[must_use]
    pub fn gouy_phase(&self, z: f64) -> f64 {
        let z_rel = z - self.focal_z;
        let z_r = self.rayleigh_range();
        (z_rel / z_r).atan()
    }

    /// Evaluate Gaussian beam pressure (real part) at given position and time.
    #[must_use]
    pub fn pressure(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        let r = x.hypot(y);
        let z_rel = z - self.focal_z;
        let w_z = self.beam_width(z);
        let amplitude_factor = self.waist_radius / w_z;
        let gaussian_envelope = (-r.powi(2) / w_z.powi(2)).exp();
        let k = TWO_PI * self.frequency / self.sound_speed;
        let omega = TWO_PI * self.frequency;
        let gouy = self.gouy_phase(z);
        let phase = k * z_rel - omega * t + gouy;
        self.amplitude * amplitude_factor * gaussian_envelope * phase.sin()
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

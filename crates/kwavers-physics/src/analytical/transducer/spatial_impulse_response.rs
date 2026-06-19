//! Transient spatial impulse response (SIR) of a flat circular piston.
//!
//! The spatial impulse response `h(r, z, t)` is the velocity-potential response
//! at a field point to an impulsive uniform normal velocity of the aperture — the
//! transient (broadband) analogue of the Fast Nearfield Method's continuous-wave
//! field ([`crate`] has the CW path; this is the impulse/pulse-echo path). It is
//! the diffraction kernel of the Field II model: the radiated pressure is
//! `p(t) = ρ ∂/∂t [ v(t) ⊛ h(t) ]`, and the pulse-echo response convolves the
//! transmit and receive SIRs with the excitation.
//!
//! # Closed form (flat circular piston, radius `a`)
//!
//! For a field point at axial distance `z > 0` and lateral offset `r ≥ 0` from the
//! axis, with `ρ = c·t` the radius of the expanding spherical wavefront's
//! intersection circle (`ρ² = (ct)² − z²`):
//!
//! ```text
//!            ⎧ 0                                              ct < d_min
//!            ⎪ c                                  d_min ≤ ct < d_plateau   (only r < a)
//! h(r,z,t) = ⎨ (c/π)·arccos[ ((ct)²−z²+r²−a²) / (2 r √((ct)²−z²)) ]
//!            ⎪                                d_plateau ≤ ct < d_max
//!            ⎩ 0                                              ct ≥ d_max
//! ```
//!
//! with `d_min = z` if `r ≤ a` else `√(z²+(r−a)²)`,
//! `d_plateau = √(z²+(a−r)²)`, and `d_max = √(z²+(a+r)²)`.
//! On axis (`r = 0`) this reduces to the rectangular pulse `h = c` over
//! `z/c ≤ t < √(z²+a²)/c`.
//!
//! # References
//! - Stepanishen, P. R. (1971). "Transient radiation from pistons in an infinite
//!   planar baffle." *J. Acoust. Soc. Am.* 49(5B), 1629–1638.
//! - Jensen, J. A. (1999). "A new calculation procedure for spatial impulse
//!   responses in ultrasound." *J. Acoust. Soc. Am.* 105(6), 3266–3274.

use kwavers_core::error::{KwaversError, KwaversResult};
use std::f64::consts::PI;

/// A flat circular piston in an infinite rigid baffle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircularPistonSir {
    radius: f64,
    sound_speed: f64,
}

impl CircularPistonSir {
    /// Create a circular-piston SIR model.
    ///
    /// # Errors
    /// - [`KwaversError::InvalidInput`] if `radius` or `sound_speed` is
    ///   non-finite or `≤ 0`.
    pub fn new(radius: f64, sound_speed: f64) -> KwaversResult<Self> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "CircularPistonSir requires radius > 0, got {radius}"
            )));
        }
        if !sound_speed.is_finite() || sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "CircularPistonSir requires sound_speed > 0, got {sound_speed}"
            )));
        }
        Ok(Self { radius, sound_speed })
    }

    /// First-arrival time [s] at field point `(r, z)` — the nearest aperture point.
    #[must_use]
    pub fn first_arrival_time(&self, r: f64, z: f64) -> f64 {
        let d_min = if r <= self.radius {
            z.abs()
        } else {
            (z * z + (r - self.radius).powi(2)).sqrt()
        };
        d_min / self.sound_speed
    }

    /// Last-arrival time [s] at field point `(r, z)` — the farthest rim point.
    #[must_use]
    pub fn last_arrival_time(&self, r: f64, z: f64) -> f64 {
        (z * z + (r + self.radius).powi(2)).sqrt() / self.sound_speed
    }

    /// Spatial impulse response `h(r, z, t)` [m/s] at a field point.
    ///
    /// `r ≥ 0` is the lateral offset from the axis, `z > 0` the axial distance,
    /// `t` the time [s]. Returns `0` outside the support `[d_min/c, d_max/c]`.
    #[must_use]
    pub fn evaluate(&self, r: f64, z: f64, t: f64) -> f64 {
        let a = self.radius;
        let c = self.sound_speed;
        let z = z.abs();
        let ct = c * t;

        // Wavefront intersection-circle radius ρ with the aperture plane.
        let rho_sq = ct * ct - z * z;
        if rho_sq <= 0.0 {
            return 0.0; // wavefront has not reached the aperture plane
        }

        let d_min = self.first_arrival_time(r, z) * c;
        let d_max = self.last_arrival_time(r, z) * c;
        if ct < d_min || ct >= d_max {
            return 0.0;
        }

        // On-axis: full-circle plateau over the whole support.
        if r == 0.0 {
            return c;
        }

        // Off-axis: plateau (full circle inside the piston) only when r < a and
        // the wavefront circle still lies entirely within the aperture.
        let d_plateau = (z * z + (a - r).powi(2)).sqrt();
        if r < a && ct < d_plateau {
            return c;
        }

        // Partial-arc region: the wavefront circle intersects the piston rim.
        // arg = (ρ² + r² − a²) / (2 r ρ), clamped for FP safety.
        let rho = rho_sq.sqrt();
        let arg = ((rho_sq + r * r - a * a) / (2.0 * r * rho)).clamp(-1.0, 1.0);
        (c / PI) * arg.acos()
    }
}

#[cfg(test)]
mod tests;

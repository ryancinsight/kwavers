//! Acoustic streaming velocity around oscillating microbubbles.
//!
//! ## References
//!
//! - Elder (1959): "Steady flow produced by vibrating cylinders"
//! - Marmottant & Hilgenfeldt (2003): "Controlled vesicle deformation"

use crate::core::error::KwaversResult;

/// Steady acoustic streaming velocity induced by viscous dissipation [m/s].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamingVelocity {
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
}

impl StreamingVelocity {
    #[must_use]
    pub fn new(vx: f64, vy: f64, vz: f64) -> Self {
        Self { vx, vy, vz }
    }

    #[must_use]
    pub fn zero() -> Self {
        Self {
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
        }
    }

    /// Speed.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn speed(&self) -> f64 {
        self.vz
            .mul_add(self.vz, self.vx.mul_add(self.vx, self.vy * self.vy))
            .sqrt()
    }
}

/// Acoustic streaming velocity at a point near an oscillating bubble.
///
/// Simplified model (Elder 1959): v_streaming ∝ (R₀²ω/ν)·(U/c)²·f(r/R₀)
///
/// Returns zero for distances ≤ R₀ (inside bubble).
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn calculate_acoustic_streaming_velocity(
    radius_equilibrium: f64,
    wall_velocity_amplitude: f64,
    frequency: f64,
    distance: f64,
    direction: (f64, f64, f64),
) -> KwaversResult<StreamingVelocity> {
    const KINEMATIC_VISCOSITY: f64 = 1e-6; // Water at 37°C [m²/s]
    const SOUND_SPEED: f64 = 1540.0; // Soft tissue [m/s]

    if distance <= radius_equilibrium {
        return Ok(StreamingVelocity::zero());
    }

    let omega = 2.0 * std::f64::consts::PI * frequency;
    let mach_sq = (wall_velocity_amplitude / SOUND_SPEED).powi(2);
    let re = (radius_equilibrium.powi(2) * omega) / KINEMATIC_VISCOSITY;
    let r_ratio = distance / radius_equilibrium;
    let decay = 1.0 / r_ratio.powi(2);
    let v_magnitude = re * mach_sq * radius_equilibrium * omega * decay;

    let dir_mag = direction
        .2
        .mul_add(
            direction.2,
            direction.0.mul_add(direction.0, direction.1 * direction.1),
        )
        .sqrt();
    if dir_mag < 1e-10 {
        return Ok(StreamingVelocity::zero());
    }

    let nx = direction.0 / dir_mag;
    let ny = direction.1 / dir_mag;
    let nz = direction.2 / dir_mag;

    Ok(StreamingVelocity::new(
        v_magnitude * nx,
        v_magnitude * ny,
        v_magnitude * nz,
    ))
}

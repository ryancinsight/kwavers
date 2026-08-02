//! Acoustic streaming velocity around oscillating microbubbles.
//!
//! ## References
//!
//! - Elder (1959): "Steady flow produced by vibrating cylinders"
//! - Marmottant & Hilgenfeldt (2003): "Controlled vesicle deformation"

use aequitas::systems::si::quantities::{Frequency, Length, Velocity};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;

use super::radiation::Direction3D;

/// Steady acoustic streaming velocity induced by viscous dissipation [m/s].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamingVelocity {
    pub vx: Velocity<f64>,
    pub vy: Velocity<f64>,
    pub vz: Velocity<f64>,
}

impl StreamingVelocity {
    #[must_use]
    pub fn new(vx: Velocity<f64>, vy: Velocity<f64>, vz: Velocity<f64>) -> Self {
        Self { vx, vy, vz }
    }

    #[must_use]
    pub fn zero() -> Self {
        Self {
            vx: Velocity::from_base(0.0),
            vy: Velocity::from_base(0.0),
            vz: Velocity::from_base(0.0),
        }
    }

    /// Speed.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn speed(&self) -> Velocity<f64> {
        let vx = self.vx.into_base();
        let vy = self.vy.into_base();
        let vz = self.vz.into_base();
        Velocity::from_base(vz.mul_add(vz, vx.mul_add(vx, vy * vy)).sqrt())
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
    radius_equilibrium: Length<f64>,
    wall_velocity_amplitude: Velocity<f64>,
    frequency: Frequency<f64>,
    distance: Length<f64>,
    direction: Direction3D,
) -> KwaversResult<StreamingVelocity> {
    use kwavers_core::constants::cavitation::VISCOSITY_WATER;
    use kwavers_core::constants::fundamental::{DENSITY_WATER, SOUND_SPEED_TISSUE};
    // Kinematic viscosity ν = η/ρ for water at 20 °C (SSOT), ≈ 1.004 × 10⁻⁶ m²/s.
    // The previous hardcoded 1e-6 was correctly the 20 °C value but the comment
    // mislabelled it as 37 °C (where ν ≈ 7 × 10⁻⁷ m²/s).
    let kinematic_viscosity = VISCOSITY_WATER / DENSITY_WATER;
    let radius_equilibrium_value = radius_equilibrium.into_base();
    let wall_velocity_amplitude_value = wall_velocity_amplitude.into_base();
    let frequency_value = frequency.into_base();
    let distance_value = distance.into_base();

    if distance_value <= radius_equilibrium_value {
        return Ok(StreamingVelocity::zero());
    }

    let omega = TWO_PI * frequency_value;
    let mach_sq = (wall_velocity_amplitude_value / SOUND_SPEED_TISSUE).powi(2);
    let re = (radius_equilibrium_value.powi(2) * omega) / kinematic_viscosity;
    let r_ratio = distance_value / radius_equilibrium_value;
    let decay = 1.0 / r_ratio.powi(2);
    let v_magnitude = re * mach_sq * radius_equilibrium_value * omega * decay;

    let dir_mag = direction
        .z
        .mul_add(
            direction.z,
            direction.x.mul_add(direction.x, direction.y * direction.y),
        )
        .sqrt();
    if dir_mag < 1e-10 {
        return Ok(StreamingVelocity::zero());
    }

    let nx = direction.x / dir_mag;
    let ny = direction.y / dir_mag;
    let nz = direction.z / dir_mag;

    Ok(StreamingVelocity::new(
        Velocity::from_base(v_magnitude * nx),
        Velocity::from_base(v_magnitude * ny),
        Velocity::from_base(v_magnitude * nz),
    ))
}

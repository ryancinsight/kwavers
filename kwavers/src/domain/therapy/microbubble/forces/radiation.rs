//! Radiation force on oscillating microbubbles (Bjerknes force and Stokes drag).
//!
//! ## References
//!
//! - Leighton (1994): "The Acoustic Bubble"
//! - Blake (1986): "Bjerknes forces in stationary sound fields"

use crate::core::constants::cavitation::VISCOSITY_WATER;
use crate::core::error::KwaversResult;

/// Time-averaged radiation force on an oscillating bubble (N).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadiationForce {
    pub fx: f64,
    pub fy: f64,
    pub fz: f64,
}

impl RadiationForce {
    #[must_use]
    pub fn new(fx: f64, fy: f64, fz: f64) -> Self {
        Self { fx, fy, fz }
    }

    #[must_use]
    pub fn zero() -> Self {
        Self {
            fx: 0.0,
            fy: 0.0,
            fz: 0.0,
        }
    }

    #[must_use]
    pub fn magnitude(&self) -> f64 {
        self.fz
            .mul_add(self.fz, self.fx.mul_add(self.fx, self.fy * self.fy))
            .sqrt()
    }

    #[must_use]
    pub fn normalized(&self) -> Self {
        let mag = self.magnitude();
        if mag > 0.0 {
            Self {
                fx: self.fx / mag,
                fy: self.fy / mag,
                fz: self.fz / mag,
            }
        } else {
            Self::zero()
        }
    }

    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            fx: self.fx + other.fx,
            fy: self.fy + other.fy,
            fz: self.fz + other.fz,
        }
    }

    /// Scale.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            fx: self.fx * factor,
            fy: self.fy * factor,
            fz: self.fz * factor,
        }
    }
}

/// Primary Bjerknes force using instantaneous radius: F = -(4π/3)R³ · ∇P.
///
/// For time-averaged force, average the result over multiple acoustic periods.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn calculate_primary_bjerknes_force(
    radius: f64,
    _radius_equilibrium: f64,
    pressure_gradient: (f64, f64, f64),
) -> KwaversResult<RadiationForce> {
    let volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
    Ok(RadiationForce::new(
        -volume * pressure_gradient.0,
        -volume * pressure_gradient.1,
        -volume * pressure_gradient.2,
    ))
}

/// Primary Bjerknes force using time-averaged radius: F = -(4π/3)⟨R³⟩ · ∇P.
///
/// Pass the cube-root of ⟨R³⟩ as `radius_avg` for correct time-averaging.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn calculate_primary_bjerknes_force_averaged(
    radius_avg: f64,
    _radius_equilibrium: f64,
    pressure_gradient: (f64, f64, f64),
) -> KwaversResult<RadiationForce> {
    let volume_avg = (4.0 / 3.0) * std::f64::consts::PI * radius_avg.powi(3);
    Ok(RadiationForce::new(
        -volume_avg * pressure_gradient.0,
        -volume_avg * pressure_gradient.1,
        -volume_avg * pressure_gradient.2,
    ))
}

/// Stokes drag force: F_drag = −6πμRv.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn calculate_drag_force(
    radius: f64,
    relative_velocity: (f64, f64, f64),
) -> KwaversResult<RadiationForce> {
    // Dynamic viscosity of water at 20 °C from SSOT (≈ 1.002 mPa·s).
    // The previous hardcoded 0.001 was correctly the 20 °C value but the
    // comment mislabelled it as 37 °C (where η ≈ 6.9 × 10⁻⁴ Pa·s).
    let drag_coeff = 6.0 * std::f64::consts::PI * VISCOSITY_WATER * radius;
    Ok(RadiationForce::new(
        -drag_coeff * relative_velocity.0,
        -drag_coeff * relative_velocity.1,
        -drag_coeff * relative_velocity.2,
    ))
}

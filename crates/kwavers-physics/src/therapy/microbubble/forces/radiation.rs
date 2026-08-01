//! Radiation force on oscillating microbubbles (Bjerknes force and Stokes drag).
//!
//! ## References
//!
//! - Leighton (1994): "The Acoustic Bubble"
//! - Blake (1986): "Bjerknes forces in stationary sound fields"

use crate::therapy::microbubble::Velocity3D;
use aequitas::systems::si::quantities::{Dimensionless, Force, Length, PressureGradient};
use kwavers_core::constants::cavitation::VISCOSITY_WATER;
use kwavers_core::error::KwaversResult;

/// Dimensionless Cartesian direction used by vector-valued force models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Direction3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Direction3D {
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// Cartesian pressure gradient with Aequitas dimensions on every component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureGradient3D {
    pub x: PressureGradient<f64>,
    pub y: PressureGradient<f64>,
    pub z: PressureGradient<f64>,
}

impl PressureGradient3D {
    #[must_use]
    pub const fn new(
        x: PressureGradient<f64>,
        y: PressureGradient<f64>,
        z: PressureGradient<f64>,
    ) -> Self {
        Self { x, y, z }
    }
}

/// Time-averaged radiation force on an oscillating bubble (N).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadiationForce {
    pub fx: Force<f64>,
    pub fy: Force<f64>,
    pub fz: Force<f64>,
}

impl RadiationForce {
    #[must_use]
    pub fn new(fx: Force<f64>, fy: Force<f64>, fz: Force<f64>) -> Self {
        Self { fx, fy, fz }
    }

    #[must_use]
    pub fn zero() -> Self {
        Self {
            fx: Force::from_base(0.0),
            fy: Force::from_base(0.0),
            fz: Force::from_base(0.0),
        }
    }

    #[must_use]
    pub fn magnitude(&self) -> Force<f64> {
        let fx = self.fx.into_base();
        let fy = self.fy.into_base();
        let fz = self.fz.into_base();
        Force::from_base(fz.mul_add(fz, fx.mul_add(fx, fy * fy)).sqrt())
    }

    #[must_use]
    pub fn normalized(&self) -> Direction3D {
        let mag = self.magnitude().into_base();
        if mag > 0.0 {
            Direction3D::new(
                self.fx.into_base() / mag,
                self.fy.into_base() / mag,
                self.fz.into_base() / mag,
            )
        } else {
            Direction3D::new(0.0, 0.0, 0.0)
        }
    }

    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            fx: Force::from_base(self.fx.into_base() + other.fx.into_base()),
            fy: Force::from_base(self.fy.into_base() + other.fy.into_base()),
            fz: Force::from_base(self.fz.into_base() + other.fz.into_base()),
        }
    }

    /// Scale.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn scale(&self, factor: Dimensionless<f64>) -> Self {
        Self {
            fx: Force::from_base(self.fx.into_base() * factor.into_base()),
            fy: Force::from_base(self.fy.into_base() * factor.into_base()),
            fz: Force::from_base(self.fz.into_base() * factor.into_base()),
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
    radius: Length<f64>,
    _radius_equilibrium: Length<f64>,
    pressure_gradient: PressureGradient3D,
) -> KwaversResult<RadiationForce> {
    let volume = (4.0 / 3.0) * std::f64::consts::PI * radius.into_base().powi(3);
    Ok(RadiationForce::new(
        Force::from_base(-volume * pressure_gradient.x.into_base()),
        Force::from_base(-volume * pressure_gradient.y.into_base()),
        Force::from_base(-volume * pressure_gradient.z.into_base()),
    ))
}

/// Primary Bjerknes force using time-averaged radius: F = -(4π/3)⟨R³⟩ · ∇P.
///
/// Pass the cube-root of ⟨R³⟩ as `radius_avg` for correct time-averaging.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn calculate_primary_bjerknes_force_averaged(
    radius_avg: Length<f64>,
    _radius_equilibrium: Length<f64>,
    pressure_gradient: PressureGradient3D,
) -> KwaversResult<RadiationForce> {
    let volume_avg = (4.0 / 3.0) * std::f64::consts::PI * radius_avg.into_base().powi(3);
    Ok(RadiationForce::new(
        Force::from_base(-volume_avg * pressure_gradient.x.into_base()),
        Force::from_base(-volume_avg * pressure_gradient.y.into_base()),
        Force::from_base(-volume_avg * pressure_gradient.z.into_base()),
    ))
}

/// Stokes drag force: F_drag = −6πμRv.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn calculate_drag_force(
    radius: Length<f64>,
    relative_velocity: Velocity3D,
) -> KwaversResult<RadiationForce> {
    // Dynamic viscosity of water at 20 °C from SSOT (≈ 1.002 mPa·s).
    // The previous hardcoded 0.001 was correctly the 20 °C value but the
    // comment mislabelled it as 37 °C (where η ≈ 6.9 × 10⁻⁴ Pa·s).
    let drag_coeff = 6.0 * std::f64::consts::PI * VISCOSITY_WATER * radius.into_base();
    Ok(RadiationForce::new(
        Force::from_base(-drag_coeff * relative_velocity.vx.into_base()),
        Force::from_base(-drag_coeff * relative_velocity.vy.into_base()),
        Force::from_base(-drag_coeff * relative_velocity.vz.into_base()),
    ))
}

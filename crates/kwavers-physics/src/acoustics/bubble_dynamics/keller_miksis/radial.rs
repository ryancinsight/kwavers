//! Pressure-driven spherical Keller–Miksis balance.

use aequitas::systems::si::quantities::{
    Acceleration, DynamicViscosity, Length, MassDensity, Pressure, PressureRate, Velocity,
};
use eunomia::FloatElement;

/// First-order compressible radial acceleration for a prescribed pressure jump.
///
/// For signed wall Mach number M, the inertial coefficient is
/// `R(1−M) + 4μ/(ρc)`. Differentiating the viscous traction
/// `−4μU/R` gives the acceleration term moved to the left.
/// Supply `pressure_rate` for the rest of the jump derivative, including
/// `+4μU²/R²` when differentiating viscosity. A zero derivative viscosity
/// selects a pressure derivative already supplied in full.
///
/// Inputs must be finite, with positive radius, density and sound speed,
/// nonnegative derivative viscosity and subsonic wall motion.
/// See Keller & Miksis (1980), doi:10.1121/1.384720.
#[must_use]
#[inline]
pub fn pressure_driven_acceleration<T: FloatElement>(
    radius: Length<T>,
    velocity: Velocity<T>,
    density: MassDensity<T>,
    sound_speed: Velocity<T>,
    pressure_jump: Pressure<T>,
    pressure_rate: PressureRate<T>,
    derivative_viscosity: DynamicViscosity<T>,
) -> Acceleration<T> {
    let r = radius.into_base();
    let u = velocity.into_base();
    let rho = density.into_base();
    let c = sound_speed.into_base();
    let mach = u / c;
    let pressure = (T::ONE + mach) * pressure_jump.into_base() / rho;
    let radiation = r / (rho * c) * pressure_rate.into_base();
    let inertia = T::from_f64(1.5) * (T::ONE - mach / T::from_count(3)) * u * u;
    let denominator =
        r * (T::ONE - mach) + T::from_count(4) * derivative_viscosity.into_base() / (rho * c);
    Acceleration::from_base((pressure + radiation - inertia) / denominator)
}

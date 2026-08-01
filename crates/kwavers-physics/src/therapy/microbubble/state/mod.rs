//! Microbubble State Domain Entity
//!
//! Core domain entity representing the complete state of a single microbubble
//! during therapeutic ultrasound exposure.
//!
//! ## Domain Model
//!
//! A microbubble is characterized by:
//! - **Geometric State**: Current radius, equilibrium radius
//! - **Dynamic State**: Wall velocity, wall acceleration, position, velocity
//! - **Thermodynamic State**: Internal temperature, pressure, gas content
//! - **Shell State**: Marmottant shell properties and current state
//! - **Therapeutic Payload**: Encapsulated drug content and release state
//!
//! ## Mathematical Foundations
//!
//! The microbubble dynamics are governed by the **Keller-Miksis equation**
//! with **Marmottant shell model** extensions:
//!
//! ```text
//! (1 - Ṙ/c)R R̈ + (3/2)(1 - Ṙ/3c)Ṙ² = (1 + Ṙ/c)(P_L/ρ) + (R/ρc)(dP_L/dt)
//! ```
//!
//! ## References
//!
//! - Keller & Miksis (1980): "Bubble oscillations of large amplitude"
//! - Marmottant et al. (2005): "A model for large amplitude oscillations of coated bubbles"
//! - De Jong et al. (2002): "Ultrasound scattering properties of microbubbles"

use aequitas::systems::si::quantities::{
    Acceleration, AmountOfSubstance, DynamicViscosity, Length, Mass, MassDensity, Pressure,
    SurfaceTension, ThermodynamicTemperature, Time, Velocity,
};
use kwavers_core::constants::cavitation::SURFACE_TENSION_WATER;
use kwavers_core::constants::fundamental::{ATMOSPHERIC_PRESSURE, GAS_CONSTANT};
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use std::fmt;

mod physics;
#[cfg(test)]
mod tests;

/// Position in 3D space (Cartesian coordinates) — value object (m).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position3D {
    pub x: Length<f64>,
    pub y: Length<f64>,
    pub z: Length<f64>,
}

impl Position3D {
    #[must_use]
    pub fn new(x: Length<f64>, y: Length<f64>, z: Length<f64>) -> Self {
        Self { x, y, z }
    }

    #[must_use]
    pub fn zero() -> Self {
        Self {
            x: Length::from_base(0.0),
            y: Length::from_base(0.0),
            z: Length::from_base(0.0),
        }
    }

    #[must_use]
    pub fn distance_to(&self, other: &Self) -> Length<f64> {
        let dx = self.x.into_base() - other.x.into_base();
        let dy = self.y.into_base() - other.y.into_base();
        let dz = self.z.into_base() - other.z.into_base();
        Length::from_base(dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt())
    }
}

/// Velocity in 3D space — value object (m/s).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Velocity3D {
    pub vx: Velocity<f64>,
    pub vy: Velocity<f64>,
    pub vz: Velocity<f64>,
}

impl Velocity3D {
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

    #[must_use]
    pub fn magnitude(&self) -> Velocity<f64> {
        let vx = self.vx.into_base();
        let vy = self.vy.into_base();
        let vz = self.vz.into_base();
        Velocity::from_base(vz.mul_add(vz, vx.mul_add(vx, vy * vy)).sqrt())
    }
}

/// Complete microbubble state for therapeutic ultrasound simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct MicrobubbleState {
    pub radius: Length<f64>,
    pub radius_equilibrium: Length<f64>,
    pub wall_velocity: Velocity<f64>,
    pub wall_acceleration: Acceleration<f64>,
    pub position: Position3D,
    pub velocity: Velocity3D,
    pub temperature: ThermodynamicTemperature<f64>,
    pub pressure_internal: Pressure<f64>,
    pub pressure_liquid: Pressure<f64>,
    pub gas_moles: AmountOfSubstance<f64>,
    pub vapor_moles: AmountOfSubstance<f64>,
    pub shell_elasticity: SurfaceTension<f64>,
    pub shell_viscosity: DynamicViscosity<f64>,
    pub shell_radius_buckling: Length<f64>,
    pub shell_radius_rupture: Length<f64>,
    pub surface_tension: SurfaceTension<f64>,
    pub drug_concentration: MassDensity<f64>,
    pub drug_released_total: Mass<f64>,
    pub time: Time<f64>,
    pub has_cavitated: bool,
    pub shell_is_ruptured: bool,
}

impl MicrobubbleState {
    /// New.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn new(
        radius_equilibrium: Length<f64>,
        shell_elasticity: SurfaceTension<f64>,
        shell_viscosity: DynamicViscosity<f64>,
        drug_concentration: MassDensity<f64>,
        position: Position3D,
    ) -> KwaversResult<Self> {
        let radius_equilibrium_value = radius_equilibrium.into_base();
        let shell_elasticity_value = shell_elasticity.into_base();
        let shell_viscosity_value = shell_viscosity.into_base();
        let drug_concentration_value = drug_concentration.into_base();

        if radius_equilibrium_value <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_equilibrium".to_owned(),
                value: radius_equilibrium_value,
                reason: "must be positive".to_owned(),
            }));
        }
        if shell_elasticity_value < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "shell_elasticity".to_owned(),
                value: shell_elasticity_value,
                reason: "must be non-negative".to_owned(),
            }));
        }
        if shell_viscosity_value < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "shell_viscosity".to_owned(),
                value: shell_viscosity_value,
                reason: "must be non-negative".to_owned(),
            }));
        }
        if drug_concentration_value < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "drug_concentration".to_owned(),
                value: drug_concentration_value,
                reason: "must be non-negative".to_owned(),
            }));
        }

        // Use SSOT BODY_TEMPERATURE_K (310.15 K = 37°C)
        let body_temperature = BODY_TEMPERATURE_K;
        // SURFACE_TENSION_WATER = 0.0728 N/m at 20°C (cavitation::SSOT).

        let shell_radius_buckling = radius_equilibrium_value * 0.9;
        let shell_radius_rupture = radius_equilibrium_value * 1.5;
        let volume = (4.0 / 3.0) * std::f64::consts::PI * radius_equilibrium_value.powi(3);
        let gas_moles = (ATMOSPHERIC_PRESSURE * volume) / (GAS_CONSTANT * body_temperature);

        Ok(Self {
            radius: Length::from_base(radius_equilibrium_value),
            radius_equilibrium,
            wall_velocity: Velocity::from_base(0.0),
            wall_acceleration: Acceleration::from_base(0.0),
            position,
            velocity: Velocity3D::zero(),
            temperature: ThermodynamicTemperature::from_base(body_temperature),
            pressure_internal: Pressure::from_base(ATMOSPHERIC_PRESSURE),
            pressure_liquid: Pressure::from_base(ATMOSPHERIC_PRESSURE),
            gas_moles: AmountOfSubstance::from_base(gas_moles),
            vapor_moles: AmountOfSubstance::from_base(0.0),
            shell_elasticity,
            shell_viscosity,
            shell_radius_buckling: Length::from_base(shell_radius_buckling),
            shell_radius_rupture: Length::from_base(shell_radius_rupture),
            surface_tension: SurfaceTension::from_base(SURFACE_TENSION_WATER),
            drug_concentration,
            drug_released_total: Mass::from_base(0.0),
            time: Time::from_base(0.0),
            has_cavitated: false,
            shell_is_ruptured: false,
        })
    }

    /// SonoVue-like microbubble: 1.25 μm radius, phospholipid shell, SF6 gas.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn sono_vue(position: Position3D) -> KwaversResult<Self> {
        Self::new(
            Length::from_base(1.25e-6),
            SurfaceTension::from_base(0.5),
            DynamicViscosity::from_base(0.8e-9),
            MassDensity::from_base(0.0),
            position,
        )
    }

    /// Definity-like microbubble: 1.5 μm radius, lipid bilayer, C3F8 gas.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn definity(position: Position3D) -> KwaversResult<Self> {
        Self::new(
            Length::from_base(1.5e-6),
            SurfaceTension::from_base(1.0),
            DynamicViscosity::from_base(1.2e-9),
            MassDensity::from_base(0.0),
            position,
        )
    }

    /// Drug-loaded therapeutic microbubble with weaker shell for easier rupture.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn drug_loaded(
        radius: Length<f64>,
        drug_concentration: MassDensity<f64>,
        position: Position3D,
    ) -> KwaversResult<Self> {
        Self::new(
            radius,
            SurfaceTension::from_base(0.3),
            DynamicViscosity::from_base(0.5e-9),
            drug_concentration,
            position,
        )
    }
}

impl fmt::Display for MicrobubbleState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Microbubble(R={:.2}μm, R₀={:.2}μm, Ṙ={:.2}m/s, T={:.1}K, pos=({:.3},{:.3},{:.3})m)",
            self.radius.into_base() * 1e6,
            self.radius_equilibrium.into_base() * 1e6,
            self.wall_velocity.into_base(),
            self.temperature.into_base(),
            self.position.x.into_base(),
            self.position.y.into_base(),
            self.position.z.into_base()
        )
    }
}

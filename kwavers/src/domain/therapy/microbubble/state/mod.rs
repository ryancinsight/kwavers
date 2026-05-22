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

use crate::core::constants::fundamental::{ATMOSPHERIC_PRESSURE, GAS_CONSTANT};
use crate::core::constants::thermodynamic::BODY_TEMPERATURE_K;
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use std::fmt;

mod physics;
#[cfg(test)]
mod tests;

/// Position in 3D space (Cartesian coordinates) — value object (m).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Position3D {
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[must_use]
    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt()
    }
}

/// Velocity in 3D space — value object (m/s).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Velocity3D {
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
}

impl Velocity3D {
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

    #[must_use]
    pub fn magnitude(&self) -> f64 {
        self.vz
            .mul_add(self.vz, self.vx.mul_add(self.vx, self.vy * self.vy))
            .sqrt()
    }
}

/// Complete microbubble state for therapeutic ultrasound simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct MicrobubbleState {
    pub radius: f64,
    pub radius_equilibrium: f64,
    pub wall_velocity: f64,
    pub wall_acceleration: f64,
    pub position: Position3D,
    pub velocity: Velocity3D,
    pub temperature: f64,
    pub pressure_internal: f64,
    pub pressure_liquid: f64,
    pub gas_moles: f64,
    pub vapor_moles: f64,
    pub shell_elasticity: f64,
    pub shell_viscosity: f64,
    pub shell_radius_buckling: f64,
    pub shell_radius_rupture: f64,
    pub surface_tension: f64,
    pub drug_concentration: f64,
    pub drug_released_total: f64,
    pub time: f64,
    pub has_cavitated: bool,
    pub shell_is_ruptured: bool,
}

impl MicrobubbleState {
    /// New.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn new(
        radius_equilibrium: f64,
        shell_elasticity: f64,
        shell_viscosity: f64,
        drug_concentration: f64,
        position: Position3D,
    ) -> KwaversResult<Self> {
        if radius_equilibrium <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_equilibrium".to_owned(),
                value: radius_equilibrium,
                reason: "must be positive".to_owned(),
            }));
        }
        if shell_elasticity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "shell_elasticity".to_owned(),
                value: shell_elasticity,
                reason: "must be non-negative".to_owned(),
            }));
        }
        if shell_viscosity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "shell_viscosity".to_owned(),
                value: shell_viscosity,
                reason: "must be non-negative".to_owned(),
            }));
        }
        if drug_concentration < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "drug_concentration".to_owned(),
                value: drug_concentration,
                reason: "must be non-negative".to_owned(),
            }));
        }

        // Use SSOT BODY_TEMPERATURE_K (310.15 K = 37°C)
        let body_temperature = BODY_TEMPERATURE_K;
        const WATER_SURFACE_TENSION: f64 = 0.072;

        let shell_radius_buckling = radius_equilibrium * 0.9;
        let shell_radius_rupture = radius_equilibrium * 1.5;
        let volume = (4.0 / 3.0) * std::f64::consts::PI * radius_equilibrium.powi(3);
        let gas_moles = (ATMOSPHERIC_PRESSURE * volume) / (GAS_CONSTANT * body_temperature);

        Ok(Self {
            radius: radius_equilibrium,
            radius_equilibrium,
            wall_velocity: 0.0,
            wall_acceleration: 0.0,
            position,
            velocity: Velocity3D::zero(),
            temperature: body_temperature,
            pressure_internal: ATMOSPHERIC_PRESSURE,
            pressure_liquid: ATMOSPHERIC_PRESSURE,
            gas_moles,
            vapor_moles: 0.0,
            shell_elasticity,
            shell_viscosity,
            shell_radius_buckling,
            shell_radius_rupture,
            surface_tension: WATER_SURFACE_TENSION,
            drug_concentration,
            drug_released_total: 0.0,
            time: 0.0,
            has_cavitated: false,
            shell_is_ruptured: false,
        })
    }

    /// SonoVue-like microbubble: 1.25 μm radius, phospholipid shell, SF6 gas.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn sono_vue(position: Position3D) -> KwaversResult<Self> {
        Self::new(1.25e-6, 0.5, 0.8e-9, 0.0, position)
    }

    /// Definity-like microbubble: 1.5 μm radius, lipid bilayer, C3F8 gas.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn definity(position: Position3D) -> KwaversResult<Self> {
        Self::new(1.5e-6, 1.0, 1.2e-9, 0.0, position)
    }

    /// Drug-loaded therapeutic microbubble with weaker shell for easier rupture.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn drug_loaded(
        radius_um: f64,
        drug_concentration_kg_m3: f64,
        position: Position3D,
    ) -> KwaversResult<Self> {
        Self::new(
            radius_um * 1e-6,
            0.3,
            0.5e-9,
            drug_concentration_kg_m3,
            position,
        )
    }
}

impl fmt::Display for MicrobubbleState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Microbubble(R={:.2}μm, R₀={:.2}μm, Ṙ={:.2}m/s, T={:.1}K, pos=({:.3},{:.3},{:.3})m)",
            self.radius * 1e6,
            self.radius_equilibrium * 1e6,
            self.wall_velocity,
            self.temperature,
            self.position.x,
            self.position.y,
            self.position.z
        )
    }
}

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
//! where:
//! - R: Current bubble radius [m]
//! - Ṙ: Bubble wall velocity [m/s]
//! - R̈: Bubble wall acceleration [m/s²]
//! - P_L: Liquid pressure at bubble wall [Pa]
//! - ρ: Liquid density [kg/m³]
//! - c: Sound speed in liquid [m/s]
//!
//! ## References
//!
//! - Keller & Miksis (1980): "Bubble oscillations of large amplitude"
//! - Marmottant et al. (2005): "A model for large amplitude oscillations of coated bubbles"
//! - De Jong et al. (2002): "Ultrasound scattering properties of microbubbles"

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use std::fmt;

/// Position in 3D space (Cartesian coordinates)
///
/// Value object representing spatial position in meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position3D {
    pub x: f64, // [m]
    pub y: f64, // [m]
    pub z: f64, // [m]
}

impl Position3D {
    /// Create new position
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Zero position (origin)
    #[must_use]
    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Distance to another position
    #[must_use]
    pub fn distance_to(&self, other: &Position3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Velocity in 3D space
///
/// Value object representing velocity in meters per second.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Velocity3D {
    pub vx: f64, // [m/s]
    pub vy: f64, // [m/s]
    pub vz: f64, // [m/s]
}

impl Velocity3D {
    /// Create new velocity
    #[must_use]
    pub fn new(vx: f64, vy: f64, vz: f64) -> Self {
        Self { vx, vy, vz }
    }

    /// Zero velocity (stationary)
    #[must_use]
    pub fn zero() -> Self {
        Self {
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
        }
    }

    /// Magnitude (speed)
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.vx * self.vx + self.vy * self.vy + self.vz * self.vz).sqrt()
    }
}

/// Complete state of a microbubble during therapeutic ultrasound exposure
///
/// This entity represents the full dynamic state of a single microbubble,
/// including geometric, kinematic, thermodynamic, and therapeutic properties.
///
/// ## Domain Invariants
///
/// 1. **Radius Positivity**: `radius > 0` and `radius_equilibrium > 0`
/// 2. **Physical Bounds**: `temperature > 0` (Kelvin), `pressure_internal > 0`
/// 3. **Temporal Causality**: State evolves forward in time according to physics
/// 4. **Energy Conservation**: Total energy (kinetic + potential + thermal) conserved
///
/// ## Lifecycle States
///
/// ```text
/// Initialized → Oscillating → {Stable, Cavitating, Ruptured}
/// ```
#[derive(Debug, Clone)]
pub struct MicrobubbleState {
    // === Geometric Properties ===
    /// Current bubble radius [m]
    pub radius: f64,

    /// Equilibrium radius (at rest) [m]
    pub radius_equilibrium: f64,

    // === Dynamic Properties ===
    /// Bubble wall velocity (dR/dt) [m/s]
    pub wall_velocity: f64,

    /// Bubble wall acceleration (d²R/dt²) [m/s²]
    pub wall_acceleration: f64,

    /// Spatial position in simulation domain [m]
    pub position: Position3D,

    /// Translation velocity (center of mass) [m/s]
    pub velocity: Velocity3D,

    // === Thermodynamic Properties ===
    /// Internal gas temperature [K]
    pub temperature: f64,

    /// Internal gas pressure [Pa]
    pub pressure_internal: f64,

    /// Liquid pressure at bubble wall [Pa]
    pub pressure_liquid: f64,

    /// Number of gas moles inside bubble [mol]
    pub gas_moles: f64,

    /// Number of vapor moles inside bubble [mol]
    pub vapor_moles: f64,

    // === Shell Properties (Marmottant Model) ===
    /// Shell elasticity modulus [N/m]
    pub shell_elasticity: f64,

    /// Shell viscosity [Pa·s]
    pub shell_viscosity: f64,

    /// Shell buckling radius [m]
    pub shell_radius_buckling: f64,

    /// Shell rupture radius [m]
    pub shell_radius_rupture: f64,

    /// Current surface tension (shell-dependent) [N/m]
    pub surface_tension: f64,

    // === Therapeutic Properties ===
    /// Encapsulated drug concentration [kg/m³]
    pub drug_concentration: f64,

    /// Cumulative drug released [kg]
    pub drug_released_total: f64,

    // === Simulation Metadata ===
    /// Current simulation time [s]
    pub time: f64,

    /// Whether bubble has undergone inertial cavitation
    pub has_cavitated: bool,

    /// Whether shell has ruptured
    pub shell_is_ruptured: bool,
}

impl MicrobubbleState {
    /// Create new microbubble state with typical contrast agent properties
    ///
    /// # Arguments
    ///
    /// - `radius_equilibrium`: Equilibrium radius [m]
    /// - `shell_elasticity`: Shell elastic modulus [N/m]
    /// - `shell_viscosity`: Shell viscosity [Pa·s]
    /// - `drug_concentration`: Initial drug loading [kg/m³]
    /// - `position`: Initial spatial position [m]
    ///
    /// # Returns
    ///
    /// New microbubble state at equilibrium with specified properties
    ///
    /// # Validation
    ///
    /// - All radii must be positive
    /// - Shell properties must be non-negative
    /// - Drug concentration must be non-negative
    pub fn new(
        radius_equilibrium: f64,
        shell_elasticity: f64,
        shell_viscosity: f64,
        drug_concentration: f64,
        position: Position3D,
    ) -> KwaversResult<Self> {
        // Validate inputs
        if radius_equilibrium <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_equilibrium".to_string(),
                value: radius_equilibrium,
                reason: "must be positive".to_string(),
            }));
        }
        if shell_elasticity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "shell_elasticity".to_string(),
                value: shell_elasticity,
                reason: "must be non-negative".to_string(),
            }));
        }
        if shell_viscosity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "shell_viscosity".to_string(),
                value: shell_viscosity,
                reason: "must be non-negative".to_string(),
            }));
        }
        if drug_concentration < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "drug_concentration".to_string(),
                value: drug_concentration,
                reason: "must be non-negative".to_string(),
            }));
        }

        // Standard physiological conditions
        const AMBIENT_PRESSURE: f64 = 101325.0; // 1 atm [Pa]
        const BODY_TEMPERATURE: f64 = 310.0; // 37°C [K]
        const WATER_SURFACE_TENSION: f64 = 0.072; // [N/m]

        // Marmottant model parameters (typical for lipid shells)
        let shell_radius_buckling = radius_equilibrium * 0.9; // 10% compression
        let shell_radius_rupture = radius_equilibrium * 1.5; // 50% expansion

        // Initial gas content (ideal gas law: PV = nRT)
        const R_GAS: f64 = 8.314; // Universal gas constant [J/(mol·K)]
        let volume = (4.0 / 3.0) * std::f64::consts::PI * radius_equilibrium.powi(3);
        let gas_moles = (AMBIENT_PRESSURE * volume) / (R_GAS * BODY_TEMPERATURE);

        Ok(Self {
            // Geometric
            radius: radius_equilibrium,
            radius_equilibrium,

            // Dynamic (initially at rest)
            wall_velocity: 0.0,
            wall_acceleration: 0.0,
            position,
            velocity: Velocity3D::zero(),

            // Thermodynamic
            temperature: BODY_TEMPERATURE,
            pressure_internal: AMBIENT_PRESSURE,
            pressure_liquid: AMBIENT_PRESSURE,
            gas_moles,
            vapor_moles: 0.0,

            // Shell (Marmottant model)
            shell_elasticity,
            shell_viscosity,
            shell_radius_buckling,
            shell_radius_rupture,
            surface_tension: WATER_SURFACE_TENSION,

            // Therapeutic
            drug_concentration,
            drug_released_total: 0.0,

            // Metadata
            time: 0.0,
            has_cavitated: false,
            shell_is_ruptured: false,
        })
    }

    /// Create typical SonoVue-like microbubble (clinical contrast agent)
    ///
    /// SonoVue parameters:
    /// - Mean diameter: 2.5 μm (radius: 1.25 μm)
    /// - Shell: Phospholipid monolayer
    /// - Gas: Sulfur hexafluoride (SF6)
    pub fn sono_vue(position: Position3D) -> KwaversResult<Self> {
        const RADIUS_UM: f64 = 1.25e-6; // 1.25 μm
        const SHELL_ELASTICITY: f64 = 0.5; // [N/m]
        const SHELL_VISCOSITY: f64 = 0.8e-9; // [Pa·s]
        const DRUG_LOADING: f64 = 0.0; // No drug (pure imaging)

        Self::new(
            RADIUS_UM,
            SHELL_ELASTICITY,
            SHELL_VISCOSITY,
            DRUG_LOADING,
            position,
        )
    }

    /// Create Definity-like microbubble
    ///
    /// Definity parameters:
    /// - Mean diameter: 3.0 μm (radius: 1.5 μm)
    /// - Shell: Lipid bilayer
    /// - Gas: Perfluoropropane (C3F8)
    pub fn definity(position: Position3D) -> KwaversResult<Self> {
        const RADIUS_UM: f64 = 1.5e-6; // 1.5 μm
        const SHELL_ELASTICITY: f64 = 1.0; // [N/m]
        const SHELL_VISCOSITY: f64 = 1.2e-9; // [Pa·s]
        const DRUG_LOADING: f64 = 0.0; // No drug (pure imaging)

        Self::new(
            RADIUS_UM,
            SHELL_ELASTICITY,
            SHELL_VISCOSITY,
            DRUG_LOADING,
            position,
        )
    }

    /// Create drug-loaded therapeutic microbubble
    ///
    /// Typical parameters for targeted drug delivery:
    /// - Larger size for payload capacity
    /// - Weaker shell for easier rupture
    /// - High drug loading
    pub fn drug_loaded(
        radius_um: f64,
        drug_concentration_kg_m3: f64,
        position: Position3D,
    ) -> KwaversResult<Self> {
        let radius = radius_um * 1e-6;
        const SHELL_ELASTICITY: f64 = 0.3; // Weaker shell
        const SHELL_VISCOSITY: f64 = 0.5e-9;

        Self::new(
            radius,
            SHELL_ELASTICITY,
            SHELL_VISCOSITY,
            drug_concentration_kg_m3,
            position,
        )
    }

    /// Current bubble volume [m³]
    #[must_use]
    pub fn volume(&self) -> f64 {
        (4.0 / 3.0) * std::f64::consts::PI * self.radius.powi(3)
    }

    /// Current surface area [m²]
    #[must_use]
    pub fn surface_area(&self) -> f64 {
        4.0 * std::f64::consts::PI * self.radius.powi(2)
    }

    /// Compression ratio (R/R₀)
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        self.radius / self.radius_equilibrium
    }

    /// Check if bubble is in compression phase
    #[must_use]
    pub fn is_compressed(&self) -> bool {
        self.radius < self.radius_equilibrium
    }

    /// Check if bubble is in expansion phase
    #[must_use]
    pub fn is_expanded(&self) -> bool {
        self.radius > self.radius_equilibrium
    }

    /// Check if bubble is undergoing inertial cavitation
    ///
    /// Criterion: Compression ratio > 2 (radius doubles)
    #[must_use]
    pub fn is_cavitating(&self) -> bool {
        self.compression_ratio() > 2.0
    }

    /// Kinetic energy of oscillating bubble wall [J]
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        const WATER_DENSITY: f64 = 1000.0; // [kg/m³]
        let mass_effective = 4.0 * std::f64::consts::PI * WATER_DENSITY * self.radius.powi(3);
        0.5 * mass_effective * self.wall_velocity.powi(2)
    }

    /// Potential energy relative to equilibrium [J]
    #[must_use]
    pub fn potential_energy(&self) -> f64 {
        const AMBIENT_PRESSURE: f64 = 101325.0; // [Pa]
        const POLYTROPIC_INDEX: f64 = 1.4; // For air/gas

        let r0 = self.radius_equilibrium;
        let r = self.radius;

        // Potential energy (simplified)
        let p0 = AMBIENT_PRESSURE;
        let gamma = POLYTROPIC_INDEX;

        let v0 = (4.0 / 3.0) * std::f64::consts::PI * r0.powi(3);
        let v = (4.0 / 3.0) * std::f64::consts::PI * r.powi(3);

        // Work done against pressure
        p0 * v0 * ((v0 / v).powf(gamma) - 1.0) / (gamma - 1.0)
    }

    /// Total mechanical energy [J]
    #[must_use]
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    /// Natural resonance frequency (Minnaert frequency) [Hz]
    ///
    /// Formula: f₀ = (1/2πR₀)√(3γP₀/ρ)
    #[must_use]
    pub fn resonance_frequency(&self) -> f64 {
        const AMBIENT_PRESSURE: f64 = 101325.0; // [Pa]
        const WATER_DENSITY: f64 = 1000.0; // [kg/m³]
        const POLYTROPIC_INDEX: f64 = 1.4;

        let numerator = 3.0 * POLYTROPIC_INDEX * AMBIENT_PRESSURE / WATER_DENSITY;
        numerator.sqrt() / (2.0 * std::f64::consts::PI * self.radius_equilibrium)
    }

    /// Encapsulated drug mass [kg]
    #[must_use]
    pub fn drug_mass(&self) -> f64 {
        self.drug_concentration * self.volume()
    }

    /// Remaining drug fraction [0-1]
    #[must_use]
    pub fn drug_remaining_fraction(&self) -> f64 {
        let initial_mass = self.drug_concentration
            * (4.0 / 3.0)
            * std::f64::consts::PI
            * self.radius_equilibrium.powi(3);
        if initial_mass > 0.0 {
            1.0 - (self.drug_released_total / initial_mass)
        } else {
            0.0
        }
    }

    /// Validate domain invariants
    ///
    /// Ensures physical consistency of state.
    pub fn validate(&self) -> KwaversResult<()> {
        if self.radius <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius".to_string(),
                value: self.radius,
                reason: "must be positive".to_string(),
            }));
        }
        if self.radius_equilibrium <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_equilibrium".to_string(),
                value: self.radius_equilibrium,
                reason: "must be positive".to_string(),
            }));
        }
        if self.temperature <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "temperature".to_string(),
                value: self.temperature,
                reason: "must be positive (Kelvin)".to_string(),
            }));
        }
        if self.pressure_internal < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "pressure_internal".to_string(),
                value: self.pressure_internal,
                reason: "must be non-negative".to_string(),
            }));
        }
        if self.drug_concentration < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "drug_concentration".to_string(),
                value: self.drug_concentration,
                reason: "must be non-negative".to_string(),
            }));
        }

        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sono_vue() {
        let pos = Position3D::zero();
        let state = MicrobubbleState::sono_vue(pos).unwrap();

        assert!(state.radius > 0.0);
        assert_eq!(state.radius, state.radius_equilibrium);
        assert_eq!(state.wall_velocity, 0.0);
        assert!(state.temperature > 0.0);
        assert!(state.validate().is_ok());
    }

    #[test]
    fn test_create_definity() {
        let pos = Position3D::zero();
        let state = MicrobubbleState::definity(pos).unwrap();

        assert!(state.radius > 0.0);
        assert!(state.radius > 1.0e-6); // > 1 μm
        assert!(state.validate().is_ok());
    }

    #[test]
    fn test_drug_loaded() {
        let pos = Position3D::zero();
        let drug_conc = 100.0; // kg/m³
        let state = MicrobubbleState::drug_loaded(2.0, drug_conc, pos).unwrap();

        assert_eq!(state.drug_concentration, drug_conc);
        assert!(state.drug_mass() > 0.0);
        assert_eq!(state.drug_remaining_fraction(), 1.0);
        assert!(state.validate().is_ok());
    }

    #[test]
    fn test_validation_negative_radius() {
        let pos = Position3D::zero();
        let result = MicrobubbleState::new(-1.0e-6, 1.0, 1.0, 0.0, pos);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_negative_drug() {
        let pos = Position3D::zero();
        let result = MicrobubbleState::new(1.0e-6, 1.0, 1.0, -10.0, pos);
        assert!(result.is_err());
    }

    #[test]
    fn test_volume_calculation() {
        let pos = Position3D::zero();
        let radius = 1.0e-6; // 1 μm
        let state = MicrobubbleState::new(radius, 1.0, 1.0, 0.0, pos).unwrap();

        let expected_volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
        assert!((state.volume() - expected_volume).abs() < 1e-20);
    }

    #[test]
    fn test_compression_ratio() {
        let pos = Position3D::zero();
        let mut state = MicrobubbleState::new(1.0e-6, 1.0, 1.0, 0.0, pos).unwrap();

        // At equilibrium
        assert_eq!(state.compression_ratio(), 1.0);
        assert!(!state.is_compressed());
        assert!(!state.is_expanded());

        // Compressed
        state.radius = 0.5e-6;
        assert_eq!(state.compression_ratio(), 0.5);
        assert!(state.is_compressed());
        assert!(!state.is_expanded());

        // Expanded
        state.radius = 2.0e-6;
        assert_eq!(state.compression_ratio(), 2.0);
        assert!(!state.is_compressed());
        assert!(state.is_expanded());
    }

    #[test]
    fn test_cavitation_criterion() {
        let pos = Position3D::zero();
        let mut state = MicrobubbleState::new(1.0e-6, 1.0, 1.0, 0.0, pos).unwrap();

        assert!(!state.is_cavitating());

        // Inertial cavitation: R > 2*R0
        state.radius = 2.5e-6;
        assert!(state.is_cavitating());
    }

    #[test]
    fn test_resonance_frequency() {
        let pos = Position3D::zero();
        let state = MicrobubbleState::new(1.0e-6, 1.0, 1.0, 0.0, pos).unwrap();

        let f0 = state.resonance_frequency();
        // For 1 μm bubble, f0 ≈ 3-4 MHz
        assert!(f0 > 2e6 && f0 < 5e6);
    }

    #[test]
    fn test_energy_conservation_zero_at_equilibrium() {
        let pos = Position3D::zero();
        let state = MicrobubbleState::new(1.0e-6, 1.0, 1.0, 0.0, pos).unwrap();

        // At equilibrium with zero velocity, both KE and PE should be zero or very small
        assert!(state.kinetic_energy().abs() < 1e-20);
        assert!(state.potential_energy().abs() < 1e-15);
    }

    #[test]
    fn test_position_distance() {
        let pos1 = Position3D::new(0.0, 0.0, 0.0);
        let pos2 = Position3D::new(3.0, 4.0, 0.0);

        assert_eq!(pos1.distance_to(&pos2), 5.0);
    }

    #[test]
    fn test_velocity_magnitude() {
        let vel = Velocity3D::new(3.0, 4.0, 0.0);
        assert_eq!(vel.magnitude(), 5.0);
    }
}

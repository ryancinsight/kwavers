//! Canonical Boundary Condition Types - Single Source of Truth
//!
//! This module defines all boundary condition types used throughout Kwavers.
//! All other modules MUST import from here - no duplicate definitions allowed.
//!
//! ## Design Principles
//!
//! - **SSOT**: Single Source of Truth for boundary semantics
//! - **Completeness**: Covers all physics domains (acoustic, elastic, EM)
//! - **Composability**: Types can be combined for multi-physics
//! - **Type Safety**: Enum-based dispatch prevents invalid states
//!
//! ## Mathematical Foundations
//!
//! ### General Boundary Condition Form
//!
//! ```text
//! α·u + β·∂u/∂n = g(x, t)
//! ```
//!
//! where:
//! - `u`: Field variable (pressure, displacement, electric field, etc.)
//! - `n`: Outward normal vector
//! - `α, β`: Boundary coefficients
//! - `g`: Boundary data (prescribed value or flux)
//!
//! ### Special Cases
//!
//! - **Dirichlet** (α=1, β=0): Fixed value boundary `u = g`
//! - **Neumann** (α=0, β=1): Fixed flux boundary `∂u/∂n = g`
//! - **Robin** (α≠0, β≠0): Mixed boundary condition
//! - **Periodic**: `u(x_min) = u(x_max)` with phase matching
//! - **Absorbing**: Non-reflecting condition (PML, ABC, Sommerfeld)
//!
//! ## Usage
//!
//! ```rust
//! use kwavers::domain::boundary::types::{BoundaryType, BoundaryFace, AcousticBoundaryType};
//!
//! // General boundary type
//! let bc = BoundaryType::Dirichlet;
//!
//! // Acoustic-specific boundary
//! let acoustic_bc = AcousticBoundaryType::SoundSoft; // pressure release
//!
//! // Apply to specific face
//! let face = BoundaryFace::XMin;
//! ```

use serde::{Deserialize, Serialize};

/// Canonical boundary condition types for all physics domains
///
/// This enum defines the fundamental boundary condition classifications
/// used throughout Kwavers. All solver modules must use these types.
///
/// ## Mathematical Specifications
///
/// Each variant corresponds to a specific mathematical boundary condition:
///
/// - **Dirichlet**: `u = g` (essential/first-kind)
/// - **Neumann**: `∂u/∂n = g` (natural/second-kind)
/// - **Robin**: `α·u + β·∂u/∂n = g` (mixed/third-kind)
/// - **Periodic**: `u(x_min) = u(x_max) · e^(iφ)` (phase-matched periodicity)
/// - **Absorbing**: Non-reflecting boundary (PML, ABC, Sommerfeld)
/// - **Radiation**: Far-field radiation condition (Sommerfeld, Engquist-Majda)
/// - **FreeSurface**: Stress-free boundary for elastic waves
///
/// ## References
///
/// - Kreyszig, E. (2011). "Advanced Engineering Mathematics" (10th ed.).
/// - Gustafsson, B. (2008). "High Order Difference Methods for Time Dependent PDE".
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BoundaryType {
    /// Dirichlet boundary: Fixed value `u = g`
    ///
    /// Essential boundary condition enforcing specific field values.
    /// Used for: rigid walls (acoustics), clamped edges (elasticity),
    /// grounded conductors (electromagnetics).
    Dirichlet,

    /// Neumann boundary: Fixed flux `∂u/∂n = g`
    ///
    /// Natural boundary condition enforcing specific flux/gradient.
    /// Used for: pressure release (acoustics), free edges (elasticity),
    /// insulating boundaries (electromagnetics).
    Neumann,

    /// Robin boundary: Mixed condition `α·u + β·∂u/∂n = g`
    ///
    /// Combines Dirichlet and Neumann through linear combination.
    /// Used for: impedance boundaries, convective heat transfer,
    /// partially reflective surfaces.
    Robin {
        /// Coefficient for field value (dimensionless or matched to flux units)
        alpha: f64,
        /// Coefficient for flux term (dimensionless or matched to value units)
        beta: f64,
    },

    /// Periodic boundary: `u(x_min) = u(x_max) · e^(iφ)`
    ///
    /// Enforces periodicity with optional phase shift for Bloch waves.
    /// Used for: infinite periodic structures, phononic crystals,
    /// periodic lattices, waveguides.
    Periodic {
        /// Phase shift between boundaries (radians)
        phase: f64,
    },

    /// Absorbing boundary: Non-reflecting condition
    ///
    /// Minimizes artificial reflections at computational boundaries.
    /// Implementations: PML, ABC, Sommerfeld, Higdon, Enquist-Majda.
    /// Used for: unbounded domains, open boundaries, anechoic chambers.
    Absorbing,

    /// Radiation boundary: Far-field condition
    ///
    /// Sommerfeld radiation condition: `∂u/∂r + (1/c)·∂u/∂t = 0`
    /// Used for: acoustic radiation, electromagnetic radiation,
    /// far-field scattering problems.
    Radiation,

    /// Free surface: Stress-free boundary (elastic waves)
    ///
    /// Traction-free boundary: `σ·n = 0` (stress tensor dot normal = 0)
    /// Used for: free surfaces in elasticity, fluid-solid interfaces,
    /// seismic problems.
    FreeSurface,

    /// Impedance boundary: `Z·∂u/∂n + u = 0`
    ///
    /// Frequency-dependent impedance matching boundary.
    /// Used for: acoustic impedance matching, material interfaces.
    Impedance {
        /// Acoustic impedance Z = ρc (kg/m²s)
        impedance: f64,
    },
}

/// Boundary face specification for rectangular domains
///
/// Identifies which face of a 3D rectangular domain the boundary
/// condition applies to. For 2D problems, ZMin/ZMax are unused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoundaryFace {
    /// Minimum x-face (x = x_min)
    XMin,
    /// Maximum x-face (x = x_max)
    XMax,
    /// Minimum y-face (y = y_min)
    YMin,
    /// Maximum y-face (y = y_max)
    YMax,
    /// Minimum z-face (z = z_min)
    ZMin,
    /// Maximum z-face (z = z_max)
    ZMax,
}

/// Boundary component specification for vector fields
///
/// Specifies which components of a vector field the boundary condition
/// applies to. Essential for elastic waves (displacement vector) and
/// electromagnetics (E and H field vectors).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoundaryComponent {
    /// All vector components
    All,
    /// X-component only
    X,
    /// Y-component only
    Y,
    /// Z-component only
    Z,
    /// Normal component (n·u)
    Normal,
    /// Tangential components (u - (n·u)n)
    Tangential,
}

/// Acoustic-specific boundary condition types
///
/// Specialized boundary conditions for acoustic wave propagation.
/// These map to general `BoundaryType` but provide acoustic-specific
/// semantics and parameter choices.
///
/// ## Mathematical Specifications
///
/// - **SoundSoft**: `p = 0` (pressure release)
/// - **SoundHard**: `∂p/∂n = 0` (rigid wall, zero normal velocity)
/// - **Impedance**: `Z·(∂p/∂n) + p = 0` where Z = ρc
/// - **Absorbing**: PML or ABC for non-reflecting boundaries
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AcousticBoundaryType {
    /// Sound-soft boundary: `p = 0` (pressure release)
    ///
    /// Corresponds to: Dirichlet with g=0
    /// Physical interpretation: Free surface, air-water interface
    SoundSoft,

    /// Sound-hard boundary: `∂p/∂n = 0` (rigid wall)
    ///
    /// Corresponds to: Neumann with g=0
    /// Physical interpretation: Rigid wall, zero normal velocity
    SoundHard,

    /// Impedance boundary: `Z·(∂p/∂n) + p = 0`
    ///
    /// Corresponds to: Robin with α=1, β=Z
    /// Physical interpretation: Impedance-matched boundary
    ///
    /// ## Parameters
    /// - `impedance`: Acoustic impedance Z = ρc (kg/m²s)
    Impedance {
        /// Acoustic impedance (kg/m²s)
        impedance: f64,
    },

    /// Absorbing boundary (PML or ABC)
    ///
    /// Corresponds to: Absorbing boundary type
    /// Physical interpretation: Anechoic termination
    Absorbing,

    /// Radiation condition (far-field)
    ///
    /// Corresponds to: Radiation boundary type
    /// Physical interpretation: Sommerfeld radiation condition
    Radiation,
}

/// Electromagnetic-specific boundary condition types
///
/// Specialized boundary conditions for Maxwell's equations.
/// These enforce tangential field continuity or discontinuity.
///
/// ## Mathematical Specifications
///
/// - **PEC**: `n × E = 0` (tangential E vanishes)
/// - **PMC**: `n × H = 0` (tangential H vanishes)
/// - **Absorbing**: PML or ABC for wave absorption
/// - **Periodic**: Bloch-periodic boundaries for photonic crystals
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ElectromagneticBoundaryType {
    /// Perfect Electric Conductor: `n × E = 0`
    ///
    /// Tangential electric field vanishes.
    /// Physical interpretation: Metal surface, grounded conductor
    PerfectElectricConductor,

    /// Perfect Magnetic Conductor: `n × H = 0`
    ///
    /// Tangential magnetic field vanishes.
    /// Physical interpretation: Magnetic mirror (idealized)
    PerfectMagneticConductor,

    /// Absorbing boundary (PML or ABC)
    ///
    /// Non-reflecting boundary for open problems.
    /// Physical interpretation: Anechoic chamber, free space
    Absorbing,

    /// Periodic boundary with Bloch phase
    ///
    /// Used for photonic crystals and metamaterials.
    /// Physical interpretation: Infinite periodic structure
    Periodic {
        /// Bloch wave vector k (rad/m)
        k_bloch: [f64; 3],
    },

    /// Impedance boundary for EM waves
    ///
    /// Surface impedance boundary condition.
    /// Physical interpretation: Lossy material interface
    Impedance {
        /// Surface impedance (Ω)
        impedance: f64,
    },
}

/// Elastic-specific boundary condition types
///
/// Specialized boundary conditions for elastic wave propagation.
/// Vector field boundary conditions for displacement or velocity.
///
/// ## Mathematical Specifications
///
/// - **Clamped**: `u = 0` (zero displacement)
/// - **Free**: `σ·n = 0` (zero traction/stress)
/// - **Roller**: `u·n = 0, (σ·n)×n = 0` (tangential slip allowed)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElasticBoundaryType {
    /// Clamped boundary: `u = 0` (zero displacement)
    ///
    /// Corresponds to: Dirichlet with g=0
    /// Physical interpretation: Fixed support
    Clamped,

    /// Free boundary: `σ·n = 0` (zero traction)
    ///
    /// Corresponds to: Neumann (stress-free)
    /// Physical interpretation: Free surface, traction-free
    Free,

    /// Roller boundary: Tangential slip allowed, normal fixed
    ///
    /// Corresponds to: Mixed condition `u·n = 0, (σ·n)×n = 0`
    /// Physical interpretation: Roller support
    Roller,

    /// Absorbing boundary (PML or ABC)
    Absorbing,

    /// Periodic boundary
    Periodic,
}

/// Boundary specification combining type, face, and component
///
/// Complete specification of a boundary condition including:
/// - Which boundary face it applies to
/// - What type of condition (Dirichlet, Neumann, etc.)
/// - Which components (for vector fields)
/// - Time-dependent or spatial boundary data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundarySpec {
    /// Boundary face
    pub face: BoundaryFace,
    /// Boundary condition type
    pub boundary_type: BoundaryType,
    /// Component specification (for vector fields)
    pub component: BoundaryComponent,
    /// Time-dependent flag
    pub time_dependent: bool,
}

impl BoundarySpec {
    /// Create a new boundary specification
    pub fn new(
        face: BoundaryFace,
        boundary_type: BoundaryType,
        component: BoundaryComponent,
    ) -> Self {
        Self {
            face,
            boundary_type,
            component,
            time_dependent: false,
        }
    }

    /// Create a time-dependent boundary specification
    pub fn time_dependent(
        face: BoundaryFace,
        boundary_type: BoundaryType,
        component: BoundaryComponent,
    ) -> Self {
        Self {
            face,
            boundary_type,
            component,
            time_dependent: true,
        }
    }
}

// Conversion traits for convenience

impl From<AcousticBoundaryType> for BoundaryType {
    fn from(acoustic: AcousticBoundaryType) -> Self {
        match acoustic {
            AcousticBoundaryType::SoundSoft => BoundaryType::Dirichlet,
            AcousticBoundaryType::SoundHard => BoundaryType::Neumann,
            AcousticBoundaryType::Impedance { impedance } => BoundaryType::Robin {
                alpha: 1.0,
                beta: impedance,
            },
            AcousticBoundaryType::Absorbing => BoundaryType::Absorbing,
            AcousticBoundaryType::Radiation => BoundaryType::Radiation,
        }
    }
}

impl From<ElectromagneticBoundaryType> for BoundaryType {
    fn from(em: ElectromagneticBoundaryType) -> Self {
        match em {
            ElectromagneticBoundaryType::PerfectElectricConductor => BoundaryType::Dirichlet,
            ElectromagneticBoundaryType::PerfectMagneticConductor => BoundaryType::Neumann,
            ElectromagneticBoundaryType::Absorbing => BoundaryType::Absorbing,
            ElectromagneticBoundaryType::Periodic { k_bloch: _ } => {
                BoundaryType::Periodic { phase: 0.0 }
            }
            ElectromagneticBoundaryType::Impedance { impedance } => {
                BoundaryType::Impedance { impedance }
            }
        }
    }
}

impl From<ElasticBoundaryType> for BoundaryType {
    fn from(elastic: ElasticBoundaryType) -> Self {
        match elastic {
            ElasticBoundaryType::Clamped => BoundaryType::Dirichlet,
            ElasticBoundaryType::Free => BoundaryType::FreeSurface,
            ElasticBoundaryType::Roller => BoundaryType::Robin {
                alpha: 1.0,
                beta: 0.0,
            },
            ElasticBoundaryType::Absorbing => BoundaryType::Absorbing,
            ElasticBoundaryType::Periodic => BoundaryType::Periodic { phase: 0.0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_type_creation() {
        let dirichlet = BoundaryType::Dirichlet;
        assert_eq!(dirichlet, BoundaryType::Dirichlet);

        let robin = BoundaryType::Robin {
            alpha: 1.0,
            beta: 2.0,
        };
        match robin {
            BoundaryType::Robin { alpha, beta } => {
                assert_eq!(alpha, 1.0);
                assert_eq!(beta, 2.0);
            }
            _ => panic!("Expected Robin boundary"),
        }
    }

    #[test]
    fn test_boundary_face() {
        let face = BoundaryFace::XMin;
        assert_eq!(face, BoundaryFace::XMin);
    }

    #[test]
    fn test_acoustic_to_general_conversion() {
        let sound_soft = AcousticBoundaryType::SoundSoft;
        let general: BoundaryType = sound_soft.into();
        assert_eq!(general, BoundaryType::Dirichlet);

        let impedance = AcousticBoundaryType::Impedance { impedance: 1500.0 };
        let general: BoundaryType = impedance.into();
        match general {
            BoundaryType::Robin { alpha, beta } => {
                assert_eq!(alpha, 1.0);
                assert_eq!(beta, 1500.0);
            }
            _ => panic!("Expected Robin boundary"),
        }
    }

    #[test]
    fn test_electromagnetic_conversion() {
        let pec = ElectromagneticBoundaryType::PerfectElectricConductor;
        let general: BoundaryType = pec.into();
        assert_eq!(general, BoundaryType::Dirichlet);

        let pmc = ElectromagneticBoundaryType::PerfectMagneticConductor;
        let general: BoundaryType = pmc.into();
        assert_eq!(general, BoundaryType::Neumann);
    }

    #[test]
    fn test_elastic_conversion() {
        let clamped = ElasticBoundaryType::Clamped;
        let general: BoundaryType = clamped.into();
        assert_eq!(general, BoundaryType::Dirichlet);

        let free = ElasticBoundaryType::Free;
        let general: BoundaryType = free.into();
        assert_eq!(general, BoundaryType::FreeSurface);
    }

    #[test]
    fn test_boundary_spec() {
        let spec = BoundarySpec::new(
            BoundaryFace::XMin,
            BoundaryType::Dirichlet,
            BoundaryComponent::All,
        );
        assert_eq!(spec.face, BoundaryFace::XMin);
        assert_eq!(spec.boundary_type, BoundaryType::Dirichlet);
        assert_eq!(spec.component, BoundaryComponent::All);
        assert!(!spec.time_dependent);
    }

    #[test]
    fn test_time_dependent_spec() {
        let spec = BoundarySpec::time_dependent(
            BoundaryFace::YMax,
            BoundaryType::Neumann,
            BoundaryComponent::Normal,
        );
        assert!(spec.time_dependent);
    }
}

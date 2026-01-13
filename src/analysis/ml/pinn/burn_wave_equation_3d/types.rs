//! Domain types for 3D wave equation PINN
//!
//! This module defines core domain types and boundary conditions for the 3D wave equation
//! Physics-Informed Neural Network solver. All types are pure domain logic with zero external
//! dependencies beyond standard library.
//!
//! ## Boundary Conditions
//!
//! Supported boundary condition types:
//! - **Dirichlet**: u = 0 on boundary (sound-hard surface)
//! - **Neumann**: ∂u/∂n = 0 on boundary (sound-soft surface)
//! - **Absorbing**: Radiation boundary condition (non-reflecting)
//! - **Periodic**: For infinite/repeating domains
//!
//! ## Interface Conditions
//!
//! For multi-region heterogeneous media, interface conditions enforce:
//! - Continuity of pressure: u₁ = u₂
//! - Continuity of normal velocity: ρ₁⁻¹ ∂u₁/∂n = ρ₂⁻¹ ∂u₂/∂n

use super::geometry::Geometry3D;

/// Boundary conditions for 3D wave equation domains
///
/// # Mathematical Specifications
///
/// **Dirichlet**: u(x,t) = 0 for x ∈ ∂Ω (sound-hard boundary)
///
/// **Neumann**: ∂u/∂n(x,t) = 0 for x ∈ ∂Ω (sound-soft boundary)
///
/// **Absorbing**: ∂u/∂t + c·∂u/∂n = 0 for x ∈ ∂Ω (radiation condition)
///
/// **Periodic**: u(x₁,y,z,t) = u(x₂,y,z,t) for periodic boundaries
///
/// # Examples
///
/// ```rust,ignore
/// use kwavers::analysis::ml::pinn::burn_wave_equation_3d::BoundaryCondition3D;
///
/// // Sound-hard boundary (rigid wall)
/// let bc_hard = BoundaryCondition3D::Dirichlet;
///
/// // Sound-soft boundary (pressure release)
/// let bc_soft = BoundaryCondition3D::Neumann;
///
/// // Non-reflecting boundary (open domain)
/// let bc_absorbing = BoundaryCondition3D::Absorbing;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryCondition3D {
    /// Dirichlet boundary condition: u = 0 on boundary
    ///
    /// Physically represents a sound-hard surface (rigid wall) where
    /// particle displacement is zero.
    Dirichlet,

    /// Neumann boundary condition: ∂u/∂n = 0 on boundary
    ///
    /// Physically represents a sound-soft surface (pressure release)
    /// where normal derivative is zero.
    Neumann,

    /// Absorbing boundary condition for radiation problems
    ///
    /// Implements a first-order absorbing boundary condition:
    /// ∂u/∂t + c·∂u/∂n = 0
    ///
    /// This approximates a non-reflecting boundary for outgoing waves.
    Absorbing,

    /// Periodic boundary condition for infinite domains
    ///
    /// Enforces u(x₁,y,z,t) = u(x₂,y,z,t) at opposite boundaries,
    /// useful for simulating infinite or repeating domains.
    Periodic,
}

/// Interface conditions for multi-region heterogeneous media
///
/// # Mathematical Specifications
///
/// For an acoustic interface between regions with properties (ρ₁, c₁) and (ρ₂, c₂):
///
/// **Continuity of pressure**: u₁ = u₂ at interface
///
/// **Continuity of normal velocity**: (1/ρ₁)·∂u₁/∂n = (1/ρ₂)·∂u₂/∂n at interface
///
/// These conditions ensure physical consistency when waves propagate across
/// material boundaries with different acoustic impedances.
///
/// # Examples
///
/// ```rust,ignore
/// use kwavers::analysis::ml::pinn::burn_wave_equation_3d::{
///     InterfaceCondition3D, Geometry3D,
/// };
///
/// // Interface between water (region 0) and tissue (region 1)
/// let interface_geom = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.49, 0.51);
/// let interface = InterfaceCondition3D::AcousticInterface {
///     region1: 0,
///     region2: 1,
///     interface_geometry: Box::new(interface_geom),
/// };
/// ```
#[derive(Debug, Clone)]
pub enum InterfaceCondition3D {
    /// Acoustic interface with continuity of pressure and normal velocity
    ///
    /// # Fields
    ///
    /// - `region1`: Index of first region
    /// - `region2`: Index of second region
    /// - `interface_geometry`: Geometric description of interface surface
    ///
    /// # Physical Meaning
    ///
    /// At the interface, the following must hold:
    /// 1. Pressure continuity: p₁ = p₂
    /// 2. Normal velocity continuity: vₙ₁ = vₙ₂
    ///
    /// For the wave equation u with ρ = density, c = wave speed:
    /// - u₁ = u₂
    /// - (1/ρ₁c₁²)·∂u₁/∂t = (1/ρ₂c₂²)·∂u₂/∂t
    AcousticInterface {
        /// Region 1 identifier
        region1: usize,
        /// Region 2 identifier
        region2: usize,
        /// Geometric description of the interface
        interface_geometry: Box<Geometry3D>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_condition_variants() {
        let bc_dirichlet = BoundaryCondition3D::Dirichlet;
        let bc_neumann = BoundaryCondition3D::Neumann;
        let bc_absorbing = BoundaryCondition3D::Absorbing;
        let bc_periodic = BoundaryCondition3D::Periodic;

        // Verify all variants can be created
        assert!(matches!(bc_dirichlet, BoundaryCondition3D::Dirichlet));
        assert!(matches!(bc_neumann, BoundaryCondition3D::Neumann));
        assert!(matches!(bc_absorbing, BoundaryCondition3D::Absorbing));
        assert!(matches!(bc_periodic, BoundaryCondition3D::Periodic));
    }

    #[test]
    fn test_interface_condition_creation() {
        let interface_geom = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let interface = InterfaceCondition3D::AcousticInterface {
            region1: 0,
            region2: 1,
            interface_geometry: Box::new(interface_geom),
        };

        match interface {
            InterfaceCondition3D::AcousticInterface {
                region1,
                region2,
                interface_geometry: _,
            } => {
                assert_eq!(region1, 0);
                assert_eq!(region2, 1);
            }
        }
    }

    #[test]
    fn test_type_sizes() {
        use std::mem::size_of;

        // Ensure types have reasonable memory footprint
        assert!(size_of::<BoundaryCondition3D>() <= 8);
        // InterfaceCondition3D contains a Box, so size should be pointer-sized
        assert!(size_of::<InterfaceCondition3D>() <= 64);
    }

    #[test]
    fn test_boundary_condition_debug() {
        let bc = BoundaryCondition3D::Dirichlet;
        let debug_str = format!("{:?}", bc);
        assert!(debug_str.contains("Dirichlet"));
    }

    #[test]
    fn test_interface_condition_clone() {
        let interface_geom = Geometry3D::spherical(0.5, 0.5, 0.5, 0.25);
        let interface1 = InterfaceCondition3D::AcousticInterface {
            region1: 0,
            region2: 1,
            interface_geometry: Box::new(interface_geom),
        };

        let interface2 = interface1.clone();

        match (interface1, interface2) {
            (
                InterfaceCondition3D::AcousticInterface {
                    region1: r1a,
                    region2: r2a,
                    interface_geometry: _,
                },
                InterfaceCondition3D::AcousticInterface {
                    region1: r1b,
                    region2: r2b,
                    interface_geometry: _,
                },
            ) => {
                assert_eq!(r1a, r1b);
                assert_eq!(r2a, r2b);
            }
        }
    }

    #[test]
    fn test_type_default_traits() {
        // Verify BoundaryCondition3D implements required traits
        let bc = BoundaryCondition3D::Dirichlet;
        let bc_clone = bc.clone();
        assert_eq!(bc, bc_clone);

        // Verify InterfaceCondition3D implements Clone and Debug
        let interface_geom = Geometry3D::cylindrical(0.5, 0.5, 0.0, 1.0, 0.3);
        let interface = InterfaceCondition3D::AcousticInterface {
            region1: 0,
            region2: 1,
            interface_geometry: Box::new(interface_geom),
        };
        let _interface_clone = interface.clone();
        let _debug_str = format!("{:?}", interface);
    }
}

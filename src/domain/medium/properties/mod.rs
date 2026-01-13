//! Canonical material property data structures for multi-physics simulations
//!
//! # Domain Single Source of Truth (SSOT)
//!
//! This module defines the **canonical data structures** for all material properties
//! used throughout the kwavers framework. These structs complement the trait-based
//! architecture in `domain/medium/traits` by providing concrete, composable data types.
//!
//! ## Architecture Principle: Trait + Struct Duality
//!
//! - **Traits** (`AcousticProperties`, `ElasticProperties`, etc.): Define behavioral contracts
//!   for spatial variation and computation
//! - **Structs** (`AcousticPropertyData`, `ElasticPropertyData`, etc.): Define canonical
//!   data representation for storage, serialization, and composition
//!
//! ## Design Rules
//!
//! 1. **SSOT Enforcement**: No material property structs outside `domain/medium`
//! 2. **Derived Quantities**: Computed on-demand via methods, never stored redundantly
//! 3. **Physics Decoupling**: Each property domain is independent and composable
//! 4. **Validation**: All constructors enforce physical constraints and invariants
//!
//! ## Mathematical Foundations
//!
//! Each property struct is grounded in fundamental physics:
//! - **Acoustic**: Wave equation, impedance matching, absorption models
//! - **Elastic**: Stress-strain relations, Lamé parameters, wave speeds
//! - **Electromagnetic**: Maxwell equations, constitutive relations
//! - **Optical**: Radiative transfer equation, scattering, absorption
//! - **Strength**: Yield criteria, fatigue models, fracture mechanics
//! - **Thermal**: Heat equation, Fourier's law, bio-heat transfer
//!
//! # Deep Vertical Hierarchy
//!
//! This module follows Clean Architecture and SRP principles through vertical splitting:
//! - `acoustic.rs` - Acoustic wave properties (302 lines)
//! - `elastic.rs` - Elastic solid properties (392 lines)
//! - `electromagnetic.rs` - EM wave properties (199 lines)
//! - `optical.rs` - Light propagation properties (377 lines)
//! - `strength.rs` - Mechanical strength properties (157 lines)
//! - `thermal.rs` - Thermal properties (218 lines)
//! - `composite.rs` - Multi-physics composition (267 lines)
//!
//! Each submodule is focused, testable, and independently maintainable.
//!
//! # Examples
//!
//! ```
//! use kwavers::domain::medium::properties::*;
//!
//! // Create acoustic properties for water
//! let water = AcousticPropertyData::water();
//! assert_eq!(water.impedance(), 998.0 * 1481.0);
//!
//! // Create elastic properties from engineering parameters
//! let steel = ElasticPropertyData::from_engineering(7850.0, 200e9, 0.3);
//! let cp = steel.p_wave_speed();
//! assert!((5000.0..7000.0).contains(&cp));
//!
//! // Compose multi-physics material
//! let tissue = MaterialProperties::builder()
//!     .acoustic(AcousticPropertyData::soft_tissue())
//!     .thermal(ThermalPropertyData::soft_tissue())
//!     .electromagnetic(ElectromagneticPropertyData::tissue())
//!     .build();
//! ```

// Submodule declarations
mod acoustic;
mod composite;
mod elastic;
mod electromagnetic;
mod optical;
mod strength;
mod thermal;

// Re-export all public types to maintain API compatibility
pub use acoustic::AcousticPropertyData;
pub use composite::{MaterialProperties, MaterialPropertiesBuilder};
pub use elastic::ElasticPropertyData;
pub use electromagnetic::ElectromagneticPropertyData;
pub use optical::OpticalPropertyData;
pub use strength::StrengthPropertyData;
pub use thermal::ThermalPropertyData;

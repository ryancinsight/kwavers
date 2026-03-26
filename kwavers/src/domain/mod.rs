//! Domain Layer - Core Business Logic & Specifications
//!
//! The **Domain Layer** defines the fundamental concepts, rules, and specifications that govern
//! ultrasound physics simulations. It represents the **ubiquitous language** of wave propagation,
//! tissue properties, and imaging physics.
//!
//! ## 🏗️ Architecture: DDD Bounded Contexts
//!
//! The domain is organized into **bounded contexts** - self-contained business domains with
//! their own models, language, and rules:
//!
//! ### 1. **Spatial Domain** (`geometry/`)
//! **Ubiquitous Language**: Computational domain, spatial discretization, geometric boundaries
//! - Defines **where** physics occurs (Ω ⊂ ℝⁿ)
//! - Provides domain representations (rectangular, spherical, custom)
//! - Enables grid generation, collocation sampling, boundary classification
//!
//! ### 2. **Material Properties** (`medium/`)
//! **Ubiquitous Language**: Tissue, acoustic properties, elastic moduli, attenuation
//! - Defines **what** the medium is made of (material properties)
//! - Enforces physical constraints (ρ > 0, c > 0, μ ≥ 0)
//! - Provides heterogeneous tissue models and property interpolation
//!
//! ### 3. **Wave Physics** (MOVED TO `physics/foundations/`)
//! **Note**: Physics specifications have been consolidated into the `physics/` layer.
//! - Wave equation traits are now in `physics::foundations`
//! - Use `physics::foundations::{WaveEquation, AcousticWaveEquation, ElasticWaveEquation}`
//! - Domain layer now contains only entities, not physics specifications
//!
//! ### 4. **Computational Primitives** (`tensor/`, `grid/`)
//! **Ubiquitous Language**: Arrays, grids, discretization, numerical representation
//! - Defines **how** data is stored and accessed
//! - Provides unified CPU/GPU tensor abstractions
//! - Ensures zero-copy interoperability between frameworks
//!
//! ### 5. **Imaging & Sensing** (`imaging/`, `sensor/`, `signal/`)
//! **Ubiquitous Language**: Transducers, beamforming, image reconstruction, signal processing
//! - Defines **how** we acquire and process ultrasound data
//! - Models transducer arrays, beam patterns, and detection physics
//!
//! ### 6. **Therapeutic Delivery** (`therapy/`, `source/`)
//! **Ubiquitous Language**: HIFU, ablation zones, treatment planning, safety margins
//! - Defines **how** therapeutic ultrasound delivers energy
//! - Models acoustic sources, focusing, and bioeffects
//!
//! ## 🎯 Design Principles
//!
//! ### Single Source of Truth (SSOT)
//! - Domain concepts defined **once** in domain layer
//! - Physics, solvers, and applications **depend** on domain abstractions
//! - No domain logic duplication across layers
//!
//! ### Mathematical Correctness First
//! - Type-safe enforcement of physical invariants
//! - Validation functions prevent invalid states
//! - Domain rules enforced at compile-time where possible
//!
//! ### Solver Agnosticism
//! - Domain abstractions work with any numerical method
//! - Forward solvers, PINNs, and analytical methods share same physics specs
//! - Validation frameworks work across solver implementations
//!
//! ## 🔄 Dependency Flow
//!
//! ```text
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │   Application   │◄───┤    Simulation   │◄───┤     Solver      │
//! │   (use cases)   │    │   (orchestrate) │    │  (numerics)     │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!         │                       │                       │
//!         └───────────────────────┼───────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │                    DOMAIN LAYER                        │
//! │           (business rules, physics specs)              │
//! └─────────────────────────────────────────────────────────┘
//!                                 │
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │                     CORE LAYER                         │
//! │               (fundamental types, errors)              │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## 📚 Key Domain Concepts
//!
//! - **Wave Equations**: Mathematical PDEs governing acoustic/elastic wave propagation
//! - **Material Properties**: Density, sound speed, attenuation, elastic moduli
//! - **Geometric Domains**: Spatial regions where physics is defined
//! - **Boundary Conditions**: Constraints at domain boundaries
//! - **Sources**: Time-varying forcing functions
//! - **Sensors**: Detection and measurement of wave fields
//!
//! ## 🧪 Validation & Testing
//!
//! Domain invariants are tested through:
//! - **Unit tests**: Individual domain object behavior
//! - **Integration tests**: Domain object interactions
//! - **Property-based tests**: Mathematical correctness (Proptest)
//! - **Validation functions**: Runtime invariant checking

pub mod boundary;
pub mod field;
pub mod geometry;
pub mod grid;
pub mod imaging;
pub mod medium;
pub mod mesh;
pub mod plugin;
pub mod sensor;
pub mod signal;

pub mod source;
pub mod tensor;
pub mod therapy;

// Re-export key domain types for convenience
pub use geometry::{Dimension, GeometricDomain, PointLocation, RectangularDomain, SphericalDomain};

pub use tensor::{
    convert, Backend as TensorBackend, DType, NdArrayTensor, Shape as TensorShape, TensorMut,
    TensorView,
};

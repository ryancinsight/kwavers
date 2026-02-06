//! PINN Adapter Layer
//!
//! This module provides adapters that bridge domain layer concepts to PINN-specific
//! representations, maintaining SSOT principles and clean architecture.
//!
//! ## Architecture Pattern
//!
//! The adapter layer follows the **Adapter Pattern** to convert between incompatible interfaces
//! while preserving the domain layer as the single source of truth:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                    PINN Layer (Analysis)                 │
//! │  - Physics-informed neural network training              │
//! │  - PDE residual computation                              │
//! │  - Boundary condition enforcement                        │
//! └──────────────────────────────────────────────────────────┘
//!                              ▲
//!                              │ uses
//!                              │
//! ┌──────────────────────────────────────────────────────────┐
//! │                    Adapter Layer                         │
//! │  - Type conversion (domain → PINN format)                │
//! │  - Property extraction                                   │
//! │  - Boundary condition mapping                            │
//! │  - Zero business logic duplication                       │
//! └──────────────────────────────────────────────────────────┘
//!                              ▲
//!                              │ depends on
//!                              │
//! ┌──────────────────────────────────────────────────────────┐
//! │                    Domain Layer (SSOT)                   │
//! │  - Source: Wave generation primitives                    │
//! │  - Signal: Time-varying amplitudes                       │
//! │  - Medium: Material properties                           │
//! │  - Boundary: Spatial constraints                         │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Design Principles
//!
//! 1. **SSOT Enforcement**: All domain concepts defined once in domain layer
//! 2. **Thin Adapters**: Minimal logic, primarily type conversion
//! 3. **Unidirectional Dependencies**: PINN → Adapter → Domain (never reverse)
//! 4. **Zero Duplication**: No domain types redefined in PINN layer
//! 5. **Explicit Conversion**: All adaptations explicit and traceable
//!
//! ## Module Organization
//!
//! - `source`: Adapts `domain::source::Source` to PINN source specifications
//! - `medium`: Adapts `domain::medium::Medium` to PINN material parameters
//! - `boundary`: Adapts `domain::boundary::*` to PINN boundary conditions
//!
//! ## Usage Example
//!
//! ```ignore
//! use kwavers::domain::source::PointSource;
//! use kwavers::domain::signal::waveform::SineWave;
//! use kwavers::solver::inverse::pinn::ml::adapters::source::PinnAcousticSource;
//! use std::sync::Arc;
//!
//! // 1. Define domain source (SSOT)
//! let signal = Arc::new(SineWave::new(1e6, 1.0, 0.0));
//! let domain_source = PointSource::new((0.0, 0.0, 0.0), signal, SourceField::Pressure);
//!
//! // 2. Adapt to PINN format
//! let pinn_source = PinnAcousticSource::from_domain_source(&domain_source, 0.0)?;
//!
//! // 3. Use in PINN physics domain
//! let acoustic_domain = AcousticWaveDomain::new(...);
//! // pinn_source provides boundary conditions derived from domain_source
//! ```
//!
//! ## Anti-Patterns Prevented
//!
//! ❌ **Don't**: Redefine domain concepts in PINN layer
//! ```ignore
//! // BAD: Duplicates domain::source::Source
//! pub struct PinnSource {
//!     pub position: (f64, f64, f64),
//!     // ... duplicate fields
//! }
//! ```
//!
//! ✅ **Do**: Adapt domain concepts with thin wrappers
//! ```ignore
//! // GOOD: References domain source as SSOT
//! pub struct PinnAcousticSource {
//!     // Extracted properties for PINN physics
//! }
//!
//! impl PinnAcousticSource {
//!     pub fn from_domain_source(source: &dyn Source) -> Self {
//!         // Extract only what PINN needs
//!     }
//! }
//! ```

pub mod electromagnetic;
pub mod source;

// Future adapters (to be implemented as needed)
// pub mod medium;
// pub mod boundary;

pub use source::{
    adapt_sources, AdapterError, FocalProperties, PinnAcousticSource, PinnSourceClass,
};

pub use electromagnetic::{adapt_em_sources, EMAdapterError, PinnEMSource};

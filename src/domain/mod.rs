//! Domain Layer
//!
//! This module contains the fundamental domain configurations and types that are shared
//! across the entire system. It sits at the bottom of the dependency graph (above core).
//!
//! # Architecture
//!
//! The domain layer provides three critical abstractions:
//!
//! 1. **Geometry** (`geometry/`): Spatial domain representations (rectangular, spherical, etc.)
//!    shared by all solvers for grid generation, collocation sampling, and boundary handling.
//!
//! 2. **Physics** (`physics/`): Abstract trait specifications for wave equations and physical
//!    laws, independent of numerical method (forward solvers, PINNs, analytical solutions).
//!
//! 3. **Tensor** (`tensor/`): Unified tensor abstraction supporting both ndarray (CPU) and
//!    Burn (GPU/autodiff) backends with zero-copy interoperability.
//!
//! These abstractions enable:
//! - Forward numerical solvers and inverse PINN solvers to share geometry and physics specs
//! - Minimal conversion overhead between CPU and GPU tensor representations
//! - Type-safe enforcement of physical constraints across solver types

pub mod boundary;
pub mod field;
pub mod geometry;
pub mod grid;
pub mod imaging;
pub mod medium;
pub mod mesh;
pub mod physics;
pub mod plugin;
pub mod sensor;
pub mod signal;
pub mod source;
pub mod tensor;
pub mod therapy;

// Re-export key types for convenience
pub use geometry::{Dimension, GeometricDomain, PointLocation, RectangularDomain, SphericalDomain};
pub use physics::{
    AcousticWaveEquation, BoundaryCondition as PhysicsBoundaryCondition, Domain as PhysicsDomain,
    ElasticWaveEquation, SourceTerm, SpatialDimension, TimeIntegration, WaveEquation,
};
pub use tensor::{
    convert, Backend as TensorBackend, DType, NdArrayTensor, Shape as TensorShape, TensorMut,
    TensorView,
};

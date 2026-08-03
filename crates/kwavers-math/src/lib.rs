//! Pure mathematical abstractions and primitives.
//!
//! This module contains foundational mathematical components that have no domain-specific
//! dependencies. These are the lowest-level computational building blocks used throughout
//! the system.
//!
//! # Architecture
//!
//! Math sits at the foundation of the dependency hierarchy:
//! - **No dependencies** on domain-specific modules (imaging, therapy, analysis)
//! - **Depended upon by**: solvers, physics, domain layers
//!
//! # Single Source of Truth (SSOT)
//!
//! Canonical implementations live in `leto-ops` and are re-exported here:
//!
//! | Domain concept       | leto-ops SSOT                        |
//! |----------------------|---------------------------------------|
//! | Sparse matrices      | `application::sparse`                 |
//! | Vector norms         | `application::linalg::norms`           |
//! | Iterative solvers    | `application::linalg::iterative`       |
//! | Preconditioners      | `application::linalg::iterative::preconditioners` |
//!
//! kwavers-math retains only domain-specific math: FFT, geometry, inverse problems,
//! numerics (FDTD/spectral), and signal processing. See
//! [`crate::linear_algebra`] for the remaining linear-algebra submodules.
//!
//! # Modules
//!
//! - `fft`: Fast Fourier Transform operations and k-space utilities
//! - `geometry`: Geometric primitives and spatial computations
//! - `linear_algebra`: Linear algebra operations
//! - `numerics`: Numerical methods and algorithms
//!
//! # Design Principles
//!
//! 1. **Pure Functions**: Mathematical operations should be deterministic and side-effect free
//! 2. **Type Safety**: Use newtypes and const generics to encode mathematical invariants
//! 3. **Zero-Cost Abstractions**: Leverage Rust's type system without runtime overhead
//! 4. **Composability**: Small, focused functions that compose into complex operations
//! 5. **SSOT Compliance**: No math duplicated from leto-ops; all canonical implementations
//!    flow through the re-export layer above.

pub mod fft;
pub mod geometry;
pub mod inverse_problems;
pub mod linear_algebra;
pub mod numerics;
pub mod optimization;
mod parallel;
pub mod signal;
pub mod simd;
pub mod special;
pub mod statistics;

// ============================================================================
// EXPLICIT RE-EXPORTS (Core Mathematical API)
// ============================================================================

/// FFT operations for signal processing
pub use fft::{Fft1d, Fft2d, Fft3d, KSpaceCalculator};

/// Geometric primitives and mask generation
///
/// Re-exports core geometry functions for creating spatial masks and regions.
/// These functions match MATLAB k-Wave toolbox ergonomics.
pub use geometry::{
    make_ball,   // 3D spherical mask (MATLAB: makeBall)
    make_circle, // 2D circle outline (MATLAB: makeCircle)
    make_disc,   // 2D circular mask (MATLAB: makeDisc)
    make_line,   // Linear mask between two points (MATLAB: makeLine)
    make_sphere, // Alias for make_ball (MATLAB: makeSphere)
};

/// Sparse linear algebra operations (SSOT: leto-ops)
pub use leto_ops::application::sparse::*;

/// Vector norms (SSOT: leto-ops)
pub use leto_ops::application::linalg::norms::*;

/// Iterative linear solvers (SSOT: leto-ops)
pub use leto_ops::application::linalg::iterative::*;

/// Preconditioners for iterative solvers (SSOT: leto-ops)
pub use leto_ops::application::linalg::iterative::preconditioners::*;

/// SIMD acceleration interfaces
pub use simd::{
    FdtdSimdOps, FftSimdOps, InterpolationSimdOps, MathSimdLevel, SimdConfig, SimdPerformance,
};

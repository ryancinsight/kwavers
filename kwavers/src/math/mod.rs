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
//! # Modules
//!
//! - [`fft`]: Fast Fourier Transform operations and k-space utilities
//! - [`geometry`]: Geometric primitives and spatial computations
//! - [`linear_algebra`]: Linear algebra operations including sparse matrix support
//! - [`numerics`]: Numerical methods and algorithms
//!
//! # Design Principles
//!
//! 1. **Pure Functions**: Mathematical operations should be deterministic and side-effect free
//! 2. **Type Safety**: Use newtypes and const generics to encode mathematical invariants
//! 3. **Zero-Cost Abstractions**: Leverage Rust's type system without runtime overhead
//! 4. **Composability**: Small, focused functions that compose into complex operations

pub mod fft;
pub mod geometry;
pub mod inverse_problems;
pub mod linear_algebra;
pub mod numerics;
pub mod simd;
pub mod simd_safe;

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
    make_disc,   // 2D circular mask (MATLAB: makeDisc)
    make_line,   // Linear mask between two points (MATLAB: makeLine)
    make_sphere, // Alias for make_ball (MATLAB: makeSphere)
};

/// Sparse linear algebra operations
pub use linear_algebra::sparse;

/// SIMD acceleration interfaces
pub use simd::{
    FdtdSimdOps, FftSimdOps, InterpolationSimdOps, SimdConfig, SimdLevel, SimdPerformance,
};

/// Safe SIMD operations with runtime feature detection
pub use simd_safe::SimdOps as SafeSimdOps;

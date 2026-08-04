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
//! - `fft`: Fast Fourier Transform operations and k-space utilities
//! - `geometry`: Geometric primitives and spatial computations
//! - `inverse_problems`: Regularization and inverse problem solvers
//! - `linear_algebra`: Linear algebra operations (norms, decompositions via leto-ops)
//! - `numerics`: Numerical methods and algorithms
//!
//! # Re-exports from leto-ops (SSOT)
//!
//! The following symbols are re-exported from `leto-ops` for backward compatibility
//! and to maintain a single source of truth:
//!
//! - Optimization: `leto_ops::application::optimization::{minimize, LbfgsConfig, LbfgsMemory, LbfgsResult}`
//! - Signal processing: `leto_ops::application::signal::{hann, hamming, blackman, tukey, wrap_to_pi}`
//! - Statistics: `leto_ops::application::statistics::{pearson, rmse, nrmse, psnr, normalized_rmse, percentile_range, ...}`
//! - Special functions: `leto_ops::application::special::{erf, j0, j1, jn, sinc}`
//! - Special Legendre: `leto_ops::application::special_legendre::{legendre_poly, legendre_poly_and_deriv}`
//! - Linear algebra norms: `leto_ops::application::linalg::{norm, norm_l1, norm_l2, norm_max, l2_normalize, dot_product, cross_product}`
//! - Complex linear algebra: `leto_ops::application::linalg::{complex_solve, complex_inv}`
//! - Eigendecomposition: `leto_ops::application::linalg::{eigenvalues, hermitian_eigen_jacobi, hermitian_eigen_qr, symmetric_eigen_jacobi}`
//! - Iterative solvers: `leto_ops::application::linalg::{LsqrSolver, LsqrConfig, LsqrResult}`
//! - Sparse matrices: `leto_ops::application::sparse::{CsrMatrix, CscMatrix, CooMatrix, ...}`

pub mod fft;
pub mod geometry;
pub mod inverse_problems;
pub mod linear_algebra;
pub mod numerics;
mod parallel;
pub mod simd;
pub mod simd_safe;

// ============================================================================
// RE-EXPORTS FROM leto-ops (SSOT)
// ============================================================================

// Optimization (identical to leto-ops::application::optimization)
pub use leto_ops::application::optimization::{minimize, LbfgsConfig, LbfgsMemory, LbfgsResult};

// Signal processing (window functions and phase wrapping from leto-ops::application::signal)
pub use leto_ops::application::signal::{blackman, hamming, hann, tukey, wrap_to_pi};

// Statistics (from leto-ops::application::statistics)
pub use leto_ops::application::statistics::{
    normalized_rmse, nrmse, pearson, percentile_range, phase_error_degrees_for_correlation,
    phase_shift_correlation_curve, psnr, rmse, validation_psnr_from_relative_rmse,
};

// Special functions (from leto-ops::application::special)
pub use leto_ops::application::special::{erf, j0, j1, jn, sinc};

// Special Legendre polynomials (from leto-ops::application::special_legendre)
pub use leto_ops::application::special_legendre::{legendre_poly, legendre_poly_and_deriv};

// Linear algebra norms (from leto-ops::application::linalg)
pub use leto_ops::application::linalg::{
    norm, norm_l1, norm_l2, norm_max, l2_normalize, l2_normalize_into,
};

// Complex linear algebra (from leto-ops::application::linalg)
pub use leto_ops::application::linalg::{complex_inv, complex_solve};

// Eigendecomposition (from leto-ops::application::linalg)
pub use leto_ops::application::linalg::{
    eigenvalues, hermitian_eigen_jacobi, hermitian_eigen_qr, symmetric_eigen_jacobi,
};

// Iterative solvers (from leto-ops::application::linalg)
pub use leto_ops::application::linalg::{LsqrConfig, LsqrResult, LsqrSolver};

// Sparse matrices (from leto-ops::application::sparse)
pub use leto_ops::application::sparse::{CsrMatrix, CscMatrix, CooMatrix};

// FFT operations for signal processing
pub use fft::{Fft1d, Fft2d, Fft3d, KSpaceCalculator};

// Geometric primitives and mask generation
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

// SIMD acceleration interfaces
pub use simd::{
    FdtdSimdOps, FftSimdOps, InterpolationSimdOps, MathSimdLevel, SimdConfig, SimdPerformance,
};

// Safe SIMD operations with runtime feature detection
pub use simd_safe::SimdOps;

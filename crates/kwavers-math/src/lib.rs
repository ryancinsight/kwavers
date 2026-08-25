#![doc = include_str!("../README.md")]

pub mod apodization;
pub mod fft;
pub mod geometry;
pub mod inverse_problems;
pub mod linear_algebra;
pub mod numerics;
mod parallel;
pub mod simd_safe;

// Matrix-free linear operators and LSQR solver (Athena-backed via `linear_algebra::sparse`)
pub use linear_algebra::sparse::{
    solve_lsqr_matfree, LsqrConfig, MatFreeOperator, MatFreeResult,
};

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
    l2_normalize, l2_normalize_into, norm, norm_l1, norm_l2, norm_max,
};

// Complex linear algebra (from leto-ops::application::linalg)
pub use leto_ops::application::linalg::{complex_inv, complex_solve};

// Apodization types for transducer arrays
pub use apodization::ApodizationType;

// Eigendecomposition (from leto-ops::application::linalg)
pub use leto_ops::application::linalg::{
    eigenvalues, hermitian_eigen_jacobi, hermitian_eigen_qr, symmetric_eigen_jacobi,
};

// Iterative solvers — LSQR now via `kwavers-math::LsqrConfig` (Athena-backed).
// The deleted `leto_ops::LsqrSolver`/`LsqrResult` are no longer re-exported;
// callers use `solve_lsqr_matfree` with `LsqrConfig` + `MatFreeOperator`.

// Sparse matrices (from leto-ops::application::sparse)
// CsrMatrix, CscMatrix, CooMatrix re-exported only where needed via fully-qualified paths

// Domain-specific sparse matrix type for BEM/FEM
pub use linear_algebra::sparse::CompressedSparseRowMatrix;

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

// Safe SIMD operations with runtime feature detection
pub use simd_safe::SimdOps;

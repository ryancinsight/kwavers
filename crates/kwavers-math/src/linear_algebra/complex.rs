//! Complex linear algebra - SSOT: leto_ops application linalg complex_linalg.
//!
//! Re-exported here as the kwavers vocabulary so higher layers depend on one
//! import path while the implementation lives in leto-ops.

use eunomia::Complex64;
use kwavers_core::error::KwaversResult;
use leto::{Array1, Array2};

type C64 = Complex64;

/// Complex linear algebra operations for beamforming.
///
/// Thin wrapper around leto-ops complex linear algebra, preserving the
/// kwavers error contract (KwaversError) while delegating to the SSOT
/// implementation in leto-ops.
#[derive(Debug)]
pub struct ComplexLinearAlgebra;

impl ComplexLinearAlgebra {
    /// Solve a complex linear system Ax = b using Gaussian elimination with
    /// partial pivoting (delegates to leto_ops::complex_solve).
    pub fn solve_linear_system_complex(
        a: &Array2<C64>,
        b: &Array1<C64>,
    ) -> KwaversResult<Array1<C64>> {
        Ok(leto_ops::complex_solve(a, b)?)
    }

    /// Compute the inverse of a complex matrix (delegates to leto_ops::complex_inv).
    pub fn matrix_inverse_complex(a: &Array2<C64>) -> KwaversResult<Array2<C64>> {
        Ok(leto_ops::complex_inv(a)?)
    }
}

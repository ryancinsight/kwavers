//! Private BC applicators: Dirichlet, Neumann, Robin, Radiation.

use super::BemBoundaryManager;
use kwavers_core::error::KwaversResult;
use kwavers_math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::Array1;
use num_complex::Complex64;

impl BemBoundaryManager {
    /// Apply Dirichlet BCs (p = g): zero the row in H/G and store g in RHS.
    pub(super) fn apply_dirichlet(
        &self,
        h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        boundary_values: &mut Array1<Complex64>,
        node_values: &[(usize, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, bc_value) in node_values {
            h_matrix.zero_row(node_idx);
            g_matrix.zero_row(node_idx);
            boundary_values[node_idx] = bc_value;
        }
        Ok(())
    }

    /// Apply Neumann BCs (∂p/∂n = g): store the derivative value in RHS.
    pub(super) fn apply_neumann(
        &self,
        _h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        _g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        boundary_values: &mut Array1<Complex64>,
        node_derivatives: &[(usize, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, deriv_value) in node_derivatives {
            boundary_values[node_idx] = deriv_value;
        }
        Ok(())
    }

    /// Apply Robin BCs (∂p/∂n + αp = g): modify H diagonal, store g in RHS.
    pub(super) fn apply_robin(
        &self,
        h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        _g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        boundary_values: &mut Array1<Complex64>,
        node_conditions: &[(usize, f64, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, alpha, g_value) in node_conditions {
            h_matrix.set_diagonal(node_idx, Complex64::new(alpha, 0.0));
            boundary_values[node_idx] = g_value;
        }
        Ok(())
    }

    /// Apply radiation BCs (Sommerfeld ∂p/∂n − ikp ≈ 0): add −ik to H diagonal.
    pub(super) fn apply_radiation(
        &self,
        h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        _g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        wavenumber: f64,
        nodes: &[usize],
    ) -> KwaversResult<()> {
        let radiation_term = Complex64::new(0.0, -wavenumber);
        for &node_idx in nodes {
            h_matrix.set_diagonal(node_idx, radiation_term);
        }
        Ok(())
    }
}

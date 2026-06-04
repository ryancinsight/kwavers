//! Shared finite-difference normal-gradient helper used by both the Neumann
//! and Robin transmission branches.

use ndarray::Array3;

use super::SchwarzBoundary;

impl SchwarzBoundary {
    /// Compute normal gradient `∂u/∂n` using centered finite differences.
    ///
    /// # Mathematical Form
    ///
    /// Centered difference (interior points):
    /// ```text
    /// ∂u/∂x ≈ (u[i+1,j,k] - u[i-1,j,k]) / (2Δx)
    /// ```
    ///
    /// Forward difference (left boundary):
    /// ```text
    /// ∂u/∂x ≈ (u[i+1,j,k] - u[i,j,k]) / Δx
    /// ```
    ///
    /// Backward difference (right boundary):
    /// ```text
    /// ∂u/∂x ≈ (u[i,j,k] - u[i-1,j,k]) / Δx
    /// ```
    ///
    /// # Notes
    ///
    /// - Currently implements x-direction gradient (assumes x-normal interface).
    /// - For general interfaces, would need to project gradient onto normal vector.
    /// - Accuracy: O(Δx²) for centered difference, O(Δx) at boundaries.
    pub(super) fn compute_normal_gradient(
        field: &Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> f64 {
        let (nx, _ny, _nz) = field.dim();

        if i > 0 && i < nx - 1 {
            (field[[i + 1, j, k]] - field[[i - 1, j, k]]) / 2.0
        } else if i == 0 {
            field[[i + 1, j, k]] - field[[i, j, k]]
        } else {
            field[[i, j, k]] - field[[i - 1, j, k]]
        }
    }
}

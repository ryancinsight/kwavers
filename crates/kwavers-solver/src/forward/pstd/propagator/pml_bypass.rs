//! Reusable x-plane preservation for Dirichlet-PML bypass rows.

use kwavers_core::error::{KwaversError, KwaversResult};
use crate::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use ndarray::{s, Array3, ArrayViewMut3};

impl PSTDSolver {
    /// Resize the reusable Dirichlet-PML bypass scratch plane.
    ///
    /// # Contract
    /// The scratch buffer stores exactly `rows × ny × nz` values. Each saved
    /// row is a yz-plane, so one buffer is sufficient for velocity and density
    /// components when components are processed sequentially.
    pub(crate) fn resize_pml_bypass_scratch(&mut self) {
        let shape = (
            self.dirichlet_pml_bypass_x.len(),
            self.grid.ny,
            self.grid.nz,
        );
        if self.pml_bypass_plane_scratch.dim() != shape {
            self.pml_bypass_plane_scratch = Array3::zeros(shape);
        }
    }

    /// Apply a field mutation while preserving selected x-planes.
    ///
    /// # Contract
    /// For field `u` and bypass row set `B`, the postcondition is:
    ///
    /// ```text
    /// ∀ i ∈ B, ∀ j,k: u_after[i,j,k] = u_before[i,j,k]
    /// ∀ i ∉ B: u_after[i,j,k] = f(u_before)[i,j,k]
    /// ```
    ///
    /// The restore step runs even when `apply` returns an error, preserving the
    /// bypass invariant for recoverable PML failures.
    pub(crate) fn apply_x_plane_pml_bypass(
        field: &mut Array3<f64>,
        rows: &[usize],
        scratch: &mut Array3<f64>,
        apply: impl FnOnce(ArrayViewMut3<'_, f64>) -> KwaversResult<()>,
    ) -> KwaversResult<()> {
        Self::validate_x_plane_scratch(field, rows, scratch)?;
        Self::save_x_planes(field, rows, scratch);

        let result = apply(field.view_mut());

        Self::restore_x_planes(field, rows, scratch);
        result
    }

    fn validate_x_plane_scratch(
        field: &Array3<f64>,
        rows: &[usize],
        scratch: &Array3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        let expected = (rows.len(), ny, nz);
        if scratch.dim() != expected {
            return Err(KwaversError::InvalidInput(format!(
                "PML bypass scratch shape mismatch: expected {:?}, got {:?}",
                expected,
                scratch.dim()
            )));
        }

        if let Some(&row) = rows.iter().find(|&&row| row >= nx) {
            return Err(KwaversError::InvalidInput(format!(
                "PML bypass row {row} outside x dimension {nx}"
            )));
        }

        Ok(())
    }

    fn save_x_planes(field: &Array3<f64>, rows: &[usize], scratch: &mut Array3<f64>) {
        for (idx, &row) in rows.iter().enumerate() {
            scratch
                .slice_mut(s![idx, .., ..])
                .assign(&field.slice(s![row, .., ..]));
        }
    }

    fn restore_x_planes(field: &mut Array3<f64>, rows: &[usize], scratch: &Array3<f64>) {
        for (idx, &row) in rows.iter().enumerate() {
            field
                .slice_mut(s![row, .., ..])
                .assign(&scratch.slice(s![idx, .., ..]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x_plane_bypass_restores_selected_rows() {
        let mut field = Array3::from_shape_fn((4, 2, 3), |(i, j, k)| (100 * i + 10 * j + k) as f64);
        let original = field.clone();
        let rows = [0, 3];
        let mut scratch = Array3::zeros((rows.len(), 2, 3));

        PSTDSolver::apply_x_plane_pml_bypass(&mut field, &rows, &mut scratch, |mut view| {
            view.fill(-1.0);
            Ok(())
        })
        .expect("bypass mutation succeeds");

        for &row in &rows {
            assert_eq!(
                field.slice(s![row, .., ..]),
                original.slice(s![row, .., ..])
            );
        }
        assert_eq!(field[[1, 0, 0]], -1.0);
        assert_eq!(field[[2, 1, 2]], -1.0);
    }

    #[test]
    fn x_plane_bypass_restores_selected_rows_after_error() {
        let mut field = Array3::from_shape_fn((3, 2, 2), |(i, j, k)| (100 * i + 10 * j + k) as f64);
        let original = field.clone();
        let rows = [1];
        let mut scratch = Array3::zeros((rows.len(), 2, 2));

        let result =
            PSTDSolver::apply_x_plane_pml_bypass(&mut field, &rows, &mut scratch, |mut view| {
                view.fill(-1.0);
                Err(KwaversError::InvalidInput("synthetic apply failure".into()))
            });

        assert!(matches!(result, Err(KwaversError::InvalidInput(_))));
        assert_eq!(field.slice(s![1, .., ..]), original.slice(s![1, .., ..]));
        assert_eq!(field[[0, 0, 0]], -1.0);
        assert_eq!(field[[2, 1, 1]], -1.0);
    }

    #[test]
    fn x_plane_bypass_rejects_invalid_workspace_shape() {
        let mut field = Array3::zeros((3, 2, 2));
        let rows = [0, 2];
        let mut scratch = Array3::zeros((1, 2, 2));

        let result =
            PSTDSolver::apply_x_plane_pml_bypass(&mut field, &rows, &mut scratch, |_| Ok(()));

        let Err(KwaversError::InvalidInput(message)) = result else {
            panic!("expected invalid scratch shape error");
        };
        assert!(message.contains("scratch shape mismatch"));
    }
}

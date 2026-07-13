//! Reusable x-plane preservation for Dirichlet-PML bypass rows.

use crate::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3 as LetoArray3;
use leto::{Array3, ArrayViewMut3};

impl PSTDSolver {
    /// Resize the reusable Dirichlet-PML bypass scratch plane.
    ///
    /// # Contract
    /// The scratch buffer stores exactly `rows × ny × nz` values. Each saved
    /// row is a yz-plane, so one buffer is sufficient for velocity and density
    /// components when components are processed sequentially.
    pub(crate) fn resize_pml_bypass_scratch(&mut self) {
        let shape = (
            (self.dirichlet_pml_bypass_x.len()),
            self.grid.ny,
            self.grid.nz,
        );
        if self.pml_bypass_plane_scratch.shape() != [shape.0, shape.1, shape.2] {
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
    #[cfg(test)]
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

    pub(crate) fn apply_x_plane_pml_bypass_leto(
        field: &mut LetoArray3<f64>,
        rows: &[usize],
        scratch: &mut Array3<f64>,
        apply: impl FnOnce(ArrayViewMut3<'_, f64>) -> KwaversResult<()>,
    ) -> KwaversResult<()> {
        {
            let field_view_ro = field.view();
            Self::validate_x_plane_scratch_view(&field_view_ro, rows, scratch)?;
            Self::save_x_planes_view(&field_view_ro, rows, scratch);
        }

        let result = apply(field.view_mut());

        let mut field_view = field.view_mut();
        Self::restore_x_planes_view(&mut field_view, rows, scratch);
        result
    }

    #[cfg(test)]
    fn validate_x_plane_scratch(
        field: &Array3<f64>,
        rows: &[usize],
        scratch: &Array3<f64>,
    ) -> KwaversResult<()> {
        Self::validate_x_plane_scratch_view(&field.view(), rows, scratch)
    }

    fn validate_x_plane_scratch_view(
        field: &leto::ArrayView3<'_, f64>,
        rows: &[usize],
        scratch: &Array3<f64>,
    ) -> KwaversResult<()> {
        let [nx, ny, nz] = field.shape();
        let expected = [(rows.len()), ny, nz];
        if scratch.shape() != expected {
            return Err(KwaversError::InvalidInput(format!(
                "PML bypass scratch shape mismatch: expected {:?}, got {:?}",
                expected,
                scratch.shape()
            )));
        }

        if let Some(&row) = rows.iter().find(|&&row| row >= nx) {
            return Err(KwaversError::InvalidInput(format!(
                "PML bypass row {row} outside x dimension {nx}"
            )));
        }

        Ok(())
    }

    #[cfg(test)]
    fn save_x_planes(field: &Array3<f64>, rows: &[usize], scratch: &mut Array3<f64>) {
        Self::save_x_planes_view(&field.view(), rows, scratch);
    }

    fn save_x_planes_view(
        field: &leto::ArrayView3<'_, f64>,
        rows: &[usize],
        scratch: &mut Array3<f64>,
    ) {
        for (idx, &row) in rows.iter().enumerate() {
            scratch
                .slice_with_mut::<2>(&s![idx, .., ..])
                .unwrap()
                .assign(
                    &field
                        .slice_with::<2>(&s![row, .., ..])
                        .expect("invariant: row within field x bounds"),
                );
        }
    }

    #[cfg(test)]
    fn restore_x_planes(field: &mut Array3<f64>, rows: &[usize], scratch: &Array3<f64>) {
        let mut view = field.view_mut();
        Self::restore_x_planes_view(&mut view, rows, scratch);
    }

    fn restore_x_planes_view(
        field: &mut ArrayViewMut3<'_, f64>,
        rows: &[usize],
        scratch: &Array3<f64>,
    ) {
        for (idx, &row) in rows.iter().enumerate() {
            field
                .reborrow()
                .slice_with_mut::<2>(&s![row, .., ..])
                .unwrap()
                .assign(
                    &scratch
                        .slice_with::<2>(&s![idx, .., ..])
                        .expect("invariant: idx within scratch bounds"),
                );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x_plane_bypass_restores_selected_rows() {
        let mut field = Array3::from_shape_fn((4, 2, 3), |[i, j, k]| (100 * i + 10 * j + k) as f64);
        let original = field.clone();
        let rows = [0, 3];
        let mut scratch = Array3::zeros(((rows.len()), 2, 3));

        PSTDSolver::apply_x_plane_pml_bypass(&mut field, &rows, &mut scratch, |mut view| {
            view.fill(-1.0);
            Ok(())
        })
        .expect("bypass mutation succeeds");

        for &row in &rows {
            assert_eq!(
                field
                    .slice_with::<2>(&s![row, .., ..])
                    .expect("invariant: row within field x-extent")
                    .to_contiguous(),
                original
                    .slice_with::<2>(&s![row, .., ..])
                    .expect("invariant: row within field x-extent")
                    .to_contiguous()
            );
        }
        assert_eq!(field[[1, 0, 0]], -1.0);
        assert_eq!(field[[2, 1, 2]], -1.0);
    }

    #[test]
    fn x_plane_bypass_restores_selected_rows_after_error() {
        let mut field = Array3::from_shape_fn((3, 2, 2), |[i, j, k]| (100 * i + 10 * j + k) as f64);
        let original = field.clone();
        let rows = [1];
        let mut scratch = Array3::zeros(((rows.len()), 2, 2));

        let result =
            PSTDSolver::apply_x_plane_pml_bypass(&mut field, &rows, &mut scratch, |mut view| {
                view.fill(-1.0);
                Err(KwaversError::InvalidInput("synthetic apply failure".into()))
            });

        assert!(matches!(result, Err(KwaversError::InvalidInput(_))));
        assert_eq!(
            field
                .slice_with::<2>(&s![1, .., ..])
                .expect("invariant: row within field x-extent")
                .to_contiguous(),
            original
                .slice_with::<2>(&s![1, .., ..])
                .expect("invariant: row within field x-extent")
                .to_contiguous()
        );
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

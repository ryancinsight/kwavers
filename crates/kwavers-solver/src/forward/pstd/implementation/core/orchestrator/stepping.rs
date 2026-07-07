use super::PSTDSolver;
use kwavers_boundary::Boundary;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use ndarray::ArrayViewMut3;

fn apply_acoustic_to_ndarray_view(
    boundary: &mut dyn Boundary,
    mut field: ArrayViewMut3<'_, f64>,
    grid: &Grid,
    time_index: usize,
) -> KwaversResult<()> {
    let (nx, ny, nz) = field.dim();
    let mut leto_field = leto::Array3::from_shape_fn([nx, ny, nz], |[i, j, k]| field[[i, j, k]]);
    boundary.apply_acoustic(leto_field.view_mut(), grid, time_index)?;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                field[[i, j, k]] = leto_field[[i, j, k]];
            }
        }
    }
    Ok(())
}

impl PSTDSolver {
    /// Apply boundary.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(crate) fn apply_boundary(&mut self, time_index: usize) -> KwaversResult<()> {
        let Some(boundary) = &mut self.boundary else {
            return Ok(());
        };
        apply_acoustic_to_ndarray_view(boundary.as_mut(), self.fields.p.view_mut(), &self.grid, time_index)?;
        Ok(())
    }
}

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::physics::acoustics::mechanics::acoustic_wave::SpatialOrder;
use crate::simulation::backends::acoustic::{AcousticSolverBackend, FdtdBackend};
use ndarray::Array3;
use super::AcousticWaveSolver;

impl AcousticWaveSolver {
    /// Create new acoustic wave solver with automatic backend selection
    ///
    /// Analyzes problem characteristics and selects the optimal numerical backend.
    /// Currently defaults to FDTD for robustness.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(grid: &Grid, medium: &dyn Medium) -> KwaversResult<Self> {
        let backend = Self::create_fdtd_backend(grid, medium)?;
        let dims = (grid.nx, grid.ny, grid.nz);

        Ok(Self {
            backend,
            grid: grid.clone(),
            accumulated_p_squared: Array3::zeros(dims),
        })
    }

    /// Create FDTD backend with 2nd-order spatial accuracy and PML boundaries.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn create_fdtd_backend(
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Box<dyn AcousticSolverBackend>> {
        let backend = FdtdBackend::new(grid, medium, SpatialOrder::Second)?;
        Ok(Box::new(backend))
    }
}

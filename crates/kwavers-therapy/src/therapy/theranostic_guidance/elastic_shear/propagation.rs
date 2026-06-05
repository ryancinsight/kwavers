//! Single-pass elastic PSTD propagation for trace collection.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_solver::forward::pstd::extensions::{
    ElasticPstdMedium, ElasticPstdOrchestrator, ElasticPstdVelocitySource,
};
use ndarray::{Array2, Array3};

const ELASTIC_CFL: f64 = 0.30;

pub(super) fn propagate_traces(
    grid: &Grid,
    medium: ElasticPstdMedium,
    dt_s: f64,
    time_steps: usize,
    source: &ElasticPstdVelocitySource,
    receiver_mask: &Array3<bool>,
) -> KwaversResult<Array2<f64>> {
    let mut orchestrator = ElasticPstdOrchestrator::new(grid, medium, dt_s)?;
    let pml = (grid.nx / 8, grid.ny / 8, 0);
    orchestrator.set_split_field_pml(pml, ELASTIC_CFL * grid.dx / dt_s, 1.0e-4);
    let data = orchestrator.propagate(time_steps, Some(source), Some(receiver_mask))?;
    data.vz.ok_or_else(|| {
        KwaversError::InvalidInput("elastic shear receiver mask produced no traces".to_owned())
    })
}

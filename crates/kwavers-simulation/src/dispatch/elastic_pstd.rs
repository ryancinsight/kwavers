//! Elastic PSTD solver dispatch.

use kwavers_core::error::KwaversResult;
use crate::types::{SimulationRunRequest, SimulationRunResult};
use kwavers_solver::forward::pstd::extensions::{ElasticPstdMedium, ElasticPstdOrchestrator};

/// Run an elastic pseudo-spectral time-domain simulation.
pub fn run(req: &SimulationRunRequest<'_>) -> KwaversResult<SimulationRunResult> {
    let lame_lambda = req.medium.lame_lambda_array();
    let lame_mu = req.medium.lame_mu_array();
    let density = req.medium.density_array().to_owned();

    let pstd_medium = ElasticPstdMedium {
        lame_lambda,
        lame_mu,
        density,
    };
    let mut orch = ElasticPstdOrchestrator::new(req.grid, pstd_medium, req.dt)?;

    let sensor_mask = req.sensor_mask.clone();
    let recorded = orch.propagate(
        req.time_steps,
        req.elastic_velocity_source.as_ref(),
        sensor_mask.as_ref(),
    )?;

    let sensor_data = recorded
        .vz
        .clone()
        .unwrap_or_else(|| ndarray::Array2::zeros((1, 0)));

    Ok(SimulationRunResult {
        sensor_data,
        stats: None,
        ux_data: recorded.vx,
        uy_data: recorded.vy,
        uz_data: recorded.vz,
        ix_data: None,
        iy_data: None,
        iz_data: None,
        i_avg_x: None,
        i_avg_y: None,
        i_avg_z: None,
        velocity_stats: None,
        full_grid_stats: None,
        thermal_temperature: None,
        thermal_dose: None,
    })
}

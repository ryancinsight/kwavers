//! Elastic PSTD solver dispatch.

use crate::types::{SimulationRunRequest, SimulationRunResult};
use kwavers_core::error::KwaversResult;
use kwavers_solver::forward::pstd::extensions::{ElasticPstdMedium, ElasticPstdOrchestrator};

/// Convert an ndarray Array3<f64> to a leto Array3<f64>.
fn to_leto3(arr: ndarray::Array3<f64>) -> leto::Array3<f64> {
    let shape = arr.dim();
    // Convert ndarray to flat vec - use iter() for safe contiguous extraction
    let flat: Vec<f64> = arr.iter().copied().collect();
    leto::Array3::from_shape_vec([shape.0, shape.1, shape.2], flat)
        .expect("elastic_pstd: ndarray→leto shape mismatch")
}

/// Convert a leto Array3<bool> to a sensor-mask Array3<bool>.
fn sensor_mask_to_leto(mask: &ndarray::Array3<bool>) -> leto::Array3<bool> {
    let shape = mask.dim();
    leto::Array3::from_shape_vec([shape.0, shape.1, shape.2], mask.iter().copied().collect())
        .expect("elastic_pstd: sensor mask conversion")
}

/// Convert a leto Array2<f64> to ndarray::Array2<f64>.
fn leto_to_ndarray2(arr: leto::Array2<f64>) -> ndarray::Array2<f64> {
    let [rows, cols] = arr.shape();
    let flat = arr.into_vec();
    ndarray::Array2::from_shape_vec((rows, cols), flat)
        .expect("elastic_pstd: leto→ndarray Array2 shape mismatch")
}

/// Run an elastic pseudo-spectral time-domain simulation.
pub fn run(req: &SimulationRunRequest<'_>) -> KwaversResult<SimulationRunResult> {
    let lame_lambda = to_leto3(req.medium.lame_lambda_array());
    let lame_mu = to_leto3(req.medium.lame_mu_array());
    let density = to_leto3(req.medium.density_array().to_owned());

    // Maximum P-wave speed c_p = sqrt((λ+2μ)/ρ), used to size the PML σ_max.
    let c_max = lame_lambda
        .iter()
        .zip(lame_mu.iter())
        .zip(density.iter())
        .map(|((&l, &m), &r)| {
            if r > 0.0 {
                ((l + 2.0 * m) / r).sqrt()
            } else {
                0.0
            }
        })
        .fold(0.0_f64, f64::max);

    let pstd_medium = ElasticPstdMedium {
        lame_lambda,
        lame_mu,
        density,
    };
    let mut orch = ElasticPstdOrchestrator::new(req.grid, pstd_medium, req.dt)?;

    // Honor the requested PML so transient / IVP runs absorb outgoing (and the
    // IVP's counter-propagating) waves instead of wrapping around the periodic
    // FFT grid. Uses the exponential PML in the standard leapfrog path (which
    // reads the IVP-seeded real-space stress); split-field PML maintains its
    // own sub-fields and would bypass the seed.
    if let Some(pml) = req.pml {
        let (nx, ny, nz) = req.grid.dimensions();
        let (tx, ty, tz) = pml.size_xyz.unwrap_or_else(|| {
            let s = pml.size.unwrap_or(20);
            (s, s, if nz == 1 { 0 } else { s })
        });
        let _ = (nx, ny);
        if (tx | ty | tz) != 0 && c_max > 0.0 {
            orch.set_pml((tx, ty, tz), c_max, 1e-4);
        }
    }

    // Initial-value problem: seed the initial stress from an initial
    // displacement field (e.g. an SH plane-wave packet) when provided.
    if let (Some(axis), Some(u0)) = (req.elastic_ivp_axis, req.grid_source.p0.as_ref()) {
        orch.seed_initial_displacement(u0, axis);
    }

    // Convert sensor mask from ndarray to leto for the orchestrator.
    let sensor_mask_leto = req.sensor_mask.as_ref().map(sensor_mask_to_leto);
    let recorded = orch.propagate(
        req.time_steps,
        req.elastic_velocity_source.as_ref(),
        sensor_mask_leto.as_ref(),
    )?;

    let sensor_data = recorded
        .vz
        .clone()
        .map(leto_to_ndarray2)
        .unwrap_or_else(|| ndarray::Array2::zeros((1, 0)));

    Ok(SimulationRunResult {
        sensor_data,
        stats: None,
        ux_data: recorded.vx.map(leto_to_ndarray2),
        uy_data: recorded.vy.map(leto_to_ndarray2),
        uz_data: recorded.vz.map(leto_to_ndarray2),
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

//! Rayleigh-Sommerfeld angular-spectrum solver dispatch.

use kwavers_math::fft::Complex64;
use ndarray::{Array1, Array2, Array3};

use crate::dispatch::shared::{next_pow2, trim_initial_recorder_sample};
use crate::types::{SimulationRunRequest, SimulationRunResult};
use kwavers_core::error::KwaversResult;
use kwavers_receiver::recorder::pressure_statistics::SampledStatistics;
use kwavers_solver::analytical::transducer::{FNMConfig, FastNearfieldSolver};
use kwavers_transducer::transducers::rectangular::RectangularTransducer;

/// Run a Rayleigh-Sommerfeld angular-spectrum simulation.
pub fn run(req: &SimulationRunRequest<'_>) -> KwaversResult<SimulationRunResult> {
    let transducer = req.transducers_for_rs.first().ok_or_else(|| {
        kwavers_core::error::KwaversError::InvalidInput(
            "RayleighSommerfeld requires at least one transducer".into(),
        )
    })?;

    let c0 = req.medium.max_sound_speed();
    let rho0 = req
        .medium
        .density(req.grid.nx / 2, req.grid.ny / 2, req.grid.nz / 2);
    let frequency = req.helmholtz.and_then(|h| h.frequency).unwrap_or_else(|| {
        if transducer.frequency() > 0.0 {
            transducer.frequency()
        } else {
            1.0 / req.dt
        }
    });

    let rect_transducer = RectangularTransducer {
        width: transducer.aperture_width(),
        height: transducer.element_length(),
        frequency,
        elements: (transducer.number_elements(), 1),
    };
    let aperture_width = rect_transducer.width;
    let n_elem_x = transducer.number_elements();

    let as_x = next_pow2(2 * n_elem_x).max(64);
    let as_y = next_pow2(2).max(64);

    let fnm_config = FNMConfig {
        dx: req.grid.dx,
        dy: req.grid.dy,
        angular_spectrum_size: (as_x, as_y),
        ..FNMConfig::default()
    };

    let mut solver = FastNearfieldSolver::new(fnm_config)
        .map_err(kwavers_core::error::KwaversError::InvalidInput)?;
    solver.set_transducer(rect_transducer);
    solver.set_medium(c0, rho0);

    let velocity = Array2::<Complex64>::from_elem((n_elem_x, 1), Complex64::new(1.0, 0.0));

    let sensor_mask = req
        .sensor_mask
        .clone()
        .unwrap_or_else(|| Array3::from_elem((req.grid.nx, req.grid.ny, req.grid.nz), false));
    let sensor_indices: Vec<(usize, usize, usize)> = sensor_mask
        .indexed_iter()
        .filter(|(_, &active)| active)
        .map(|((i, j, k), _)| (i, j, k))
        .collect();

    let n_sensors = sensor_indices.len().max(1);
    let z_eval = if let Some(&(_, _, k)) = sensor_indices.first() {
        req.grid.origin[2] + k as f64 * req.grid.dz
    } else {
        req.grid.origin[2] + (req.grid.nz as f64 - 1.0) * req.grid.dz
    };

    solver
        .precompute_factors(z_eval)
        .map_err(kwavers_core::error::KwaversError::InvalidInput)?;
    let pressure_field = solver
        .compute_field(&velocity, z_eval)
        .map_err(kwavers_core::error::KwaversError::InvalidInput)?;

    let mut sensor_data = Array2::<f64>::zeros((n_sensors, 1));
    for (idx, &(i, _j, _k)) in sensor_indices.iter().enumerate() {
        let x_phys = req.grid.origin[0] + i as f64 * req.grid.dx;
        let elem_width = aperture_width / n_elem_x as f64;
        let col = ((x_phys / elem_width) as usize).min(n_elem_x.saturating_sub(1));
        sensor_data[[idx, 0]] = pressure_field[[col, 0]].norm();
    }

    let sensor_data = trim_initial_recorder_sample(sensor_data, 1, req.record_start_index);
    let stats = if !sensor_indices.is_empty() {
        Some(SampledStatistics {
            p_max: Array1::from_vec(sensor_data.rows().into_iter().map(|row| row[0]).collect()),
            p_min: Array1::from_vec(sensor_data.rows().into_iter().map(|row| row[0]).collect()),
            p_rms: Array1::from_vec(sensor_data.rows().into_iter().map(|row| row[0]).collect()),
            p_final: Array1::from_vec(sensor_data.rows().into_iter().map(|row| row[0]).collect()),
        })
    } else {
        None
    };

    Ok(SimulationRunResult {
        sensor_data,
        stats,
        ux_data: None,
        uy_data: None,
        uz_data: None,
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

use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::domain::sensor::recorder::pressure_statistics::SampledStatistics;
use kwavers::domain::source::transducers::rectangular::RectangularTransducer;
use kwavers::domain::source::GridSource;
use kwavers::solver::analytical::transducer::{FNMConfig, FastNearfieldSolver};
use ndarray::Array1;
use ndarray::Array2;
use num_complex::Complex64;

use crate::medium_py::MediumInner;
use crate::sensor_py::Sensor;
use crate::simulation_result_py::SimulationRunResult;
use crate::transducer_array_py::TransducerArray2D;

use super::super::Simulation;

impl Simulation {
    /// Run the Rayleigh-Sommerfeld (Fast Nearfield Method) solver.
    ///
    /// The RS solver computes the pressure field radiated by a rectangular
    /// transducer using angular-spectrum propagation — an O(n log n) algorithm
    /// based on McGough (2004) and Kelly & McGough (2006).
    ///
    /// # Transducer requirements
    ///
    /// The solver requires at least one [`TransducerArray2D`] attached to the
    /// `Simulation`.  The first transducer in the list is used as the source;
    /// its aperture dimensions and element count are converted to a
    /// [`RectangularTransducer`] for the FNM engine.
    ///
    /// # Frequency-domain semantics
    ///
    /// Like Helmholtz and BEM, the RS solver produces a single steady-state
    /// snapshot.  The wavenumber is derived from `helmholtz_frequency` when
    /// set, or falls back to the transducer's own operating frequency:
    /// `k = 2π · f / c`.
    ///
    /// # Output mapping
    ///
    /// The solver computes a 2-D pressure field on a plane at the sensor
    /// z-coordinate (using the first sensor's z-position).  Each sensor's
    /// (x, y) grid position is then mapped to the nearest computed pixel.
    ///
    /// # Limitations
    ///
    /// - Only the first transducer is used; additional transducers are ignored.
    /// - Grid-source (`p0`, masks) is ignored — RS is source-driven, not
    ///   initial-condition-driven.
    /// - All sensors must share the same z-coordinate (single-plane evaluation).
    /// - Angular-spectrum padding and k-space extent are auto-derived from
    ///   the transducer geometry and grid spacing.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_rs_impl(
        grid: &KwaversGrid,
        medium: &MediumInner,
        _time_steps: usize,
        dt: f64,
        helmholtz_frequency: Option<f64>,
        _grid_source: GridSource,
        sensor: Option<&Sensor>,
        transducer_sensor: Option<&TransducerArray2D>,
        record_start_index: usize,
        transducers: &[TransducerArray2D],
    ) -> KwaversResult<SimulationRunResult> {
        // ── Require a transducer ───────────────────────────────────────────
        let transducer = transducers.first().ok_or_else(|| {
            kwavers::core::error::KwaversError::InvalidInput(
                "RayleighSommerfeld solver requires at least one TransducerArray2D".into(),
            )
        })?;

        let c0 = medium.as_medium().max_sound_speed();
        let rho0 = medium
            .as_medium()
            .density(grid.nx / 2, grid.ny / 2, grid.nz / 2);

        // Derive frequency: prefer user-set `helmholtz_frequency`, then
        // transducer operating frequency, then dt proxy.
        let frequency = if let Some(freq) = helmholtz_frequency {
            freq
        } else if transducer.inner.frequency() > 0.0 {
            transducer.inner.frequency()
        } else {
            1.0 / dt
        };

        // ── Convert TransducerArray2D → RectangularTransducer ────────────
        let rect_transducer = RectangularTransducer {
            width: transducer.inner.aperture_width(),
            height: transducer.inner.element_length(),
            frequency,
            elements: (transducer.inner.number_elements(), 1),
        };
        let aperture_width = rect_transducer.width;

        // ── Build FNM solver ──────────────────────────────────────────────
        let (n_elem_x, _n_elem_y) = rect_transducer.elements;
        // Angular-spectrum size: next power-of-two above 2× the element count
        let as_x = next_pow2(2 * n_elem_x).max(64);
        let as_y = next_pow2(2).max(64); // single row in y → pad to 64

        let fnm_config = FNMConfig {
            dx: grid.dx,
            dy: grid.dy,
            angular_spectrum_size: (as_x, as_y),
            ..FNMConfig::default()
        };

        let mut solver = FastNearfieldSolver::new(fnm_config)
            .map_err(|e| kwavers::core::error::KwaversError::InvalidInput(e))?;
        solver.set_transducer(rect_transducer);
        solver.set_medium(c0, rho0);

        // ── Velocity distribution (unit amplitude, piston) ────────────────
        let velocity = Array2::<Complex64>::from_elem(
            (n_elem_x, 1),
            Complex64::new(transducer.amplitude, 0.0),
        );

        // ── Determine evaluation z-plane ───────────────────────────────────
        let sensor_mask = Self::create_sensor_mask(grid, sensor, transducer_sensor);
        let sensor_indices: Vec<(usize, usize, usize)> = sensor_mask
            .indexed_iter()
            .filter(|(_, &active)| active)
            .map(|((i, j, k), _)| (i, j, k))
            .collect();

        let n_sensors = sensor_indices.len().max(1);
        let z_eval = if let Some(&(_, _, k)) = sensor_indices.first() {
            grid.origin[2] + k as f64 * grid.dz
        } else {
            // No sensors → evaluate at the far end of the domain
            grid.origin[2] + (grid.nz as f64 - 1.0) * grid.dz
        };

        // Precompute angular-spectrum factors for this z-plane
        solver
            .precompute_factors(z_eval)
            .map_err(|e| kwavers::core::error::KwaversError::InvalidInput(e))?;

        // Compute 2-D pressure field on the evaluation plane
        let pressure_field = solver
            .compute_field(&velocity, z_eval)
            .map_err(|e| kwavers::core::error::KwaversError::InvalidInput(e))?;

        // ── Map sensors to nearest pixel ──────────────────────────────────
        let n_cols = 1_usize; // single steady-state snapshot
        let mut sensor_data = Array2::<f64>::zeros((n_sensors, n_cols));

        for (idx, &(i, _j, _k)) in sensor_indices.iter().enumerate() {
            // Map grid index to physical x and then to nearest FNM pixel column.
            // The FNM field has dimensions (n_elem_x, 1) — each column
            // corresponds to an element position on the aperture.
            let x_phys = grid.origin[0] + i as f64 * grid.dx;
            let elem_width = aperture_width / n_elem_x as f64;

            // Find the nearest element column
            let col = ((x_phys / elem_width) as usize).min(n_elem_x.saturating_sub(1));
            sensor_data[[idx, 0]] = pressure_field[[col, 0]].norm();
        }

        let sensor_data =
            Self::trim_initial_recorder_sample(sensor_data, 1, record_start_index);

        let stats = if !sensor_indices.is_empty() {
            Some(SampledStatistics {
                p_max: sensor_data
                    .rows()
                    .into_iter()
                    .map(|row| row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
                    .collect::<Array1<f64>>(),
                p_min: sensor_data
                    .rows()
                    .into_iter()
                    .map(|row| row.iter().fold(f64::INFINITY, |a, &b| a.min(b)))
                    .collect::<Array1<f64>>(),
                p_rms: sensor_data
                    .rows()
                    .into_iter()
                    .map(|row| (row[0] * row[0]).sqrt())
                    .collect::<Array1<f64>>(),
                p_final: sensor_data
                    .rows()
                    .into_iter()
                    .map(|row| row[0])
                    .collect::<Array1<f64>>(),
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
}

/// Compute the next power of two ≥ `n`.
fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

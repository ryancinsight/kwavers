//! `SimulationResult` pyclass and supporting types for simulation output.
//!
//! Re-exports the `SimulationRunResult` type and `extract_full_grid_stats`
//! helper from the kwavers simulation runner so old solver files can still
//! import them from here while they are gradually migrated.

use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::simulation::SimulationRunResult as KwaversSimulationRunResult;
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyAny;

// ── Re-exports from kwavers::simulation (keeps old solver files compiling) ──
pub use kwavers::simulation::{extract_full_grid_stats, SimulationRunResult};

// ============================================================================
// Simulation Result
// ============================================================================

/// Results from acoustic simulation.
///
/// Contains sensor recordings and metadata.
#[pyclass]
pub struct SimulationResult {
    /// 1D sensor data (single sensor) [Pa]
    pub(crate) sensor_data_1d: Option<Py<PyArray1<f64>>>,
    /// 2D sensor data (multi-sensor, shape: n_sensors x n_timesteps) [Pa]
    pub(crate) sensor_data_2d: Option<Py<PyArray2<f64>>>,
    /// Time vector [s]
    #[pyo3(get)]
    pub time: Py<PyArray1<f64>>,
    /// Grid shape (nx, ny, nz)
    #[pyo3(get)]
    pub shape: (usize, usize, usize),
    /// Sensor data shape as `(num_sensors, time_steps)`.
    #[pyo3(get)]
    pub sensor_data_shape: (usize, usize),
    /// Number of time steps
    #[pyo3(get)]
    pub time_steps: usize,
    /// Time step [s]
    #[pyo3(get)]
    pub dt: f64,
    /// Final simulation time [s]
    #[pyo3(get)]
    pub final_time: f64,
    /// Maximum pressure at each sensor position over all time steps [Pa] (None if not recorded)
    #[pyo3(get)]
    pub p_max: Option<Py<PyArray1<f64>>>,
    /// Minimum pressure at each sensor position over all time steps [Pa] (None if not recorded)
    #[pyo3(get)]
    pub p_min: Option<Py<PyArray1<f64>>>,
    /// RMS pressure at each sensor position over all time steps [Pa] (None if not recorded)
    #[pyo3(get)]
    pub p_rms: Option<Py<PyArray1<f64>>>,
    /// Final pressure at each sensor position [Pa] (None if not recorded)
    #[pyo3(get)]
    pub p_final: Option<Py<PyArray1<f64>>>,

    /// Full-grid peak compressional pressure [Pa] over all time steps —
    /// shape `(nx, ny, nz)`. None unless a `p_*` recording mode was set.
    #[pyo3(get)]
    pub p_max_field: Option<Py<PyArray3<f64>>>,
    /// Full-grid peak rarefactional pressure [Pa] (most-negative
    /// pressure per voxel) over all time steps. Shape `(nx, ny, nz)`.
    /// This is the canonical cavitation-kernel field: feed it through
    /// the Maxwell-2013 erf-CDF to obtain per-voxel intrinsic-threshold
    /// cavitation probability.
    #[pyo3(get)]
    pub p_min_field: Option<Py<PyArray3<f64>>>,
    /// Full-grid RMS pressure [Pa]. Shape `(nx, ny, nz)`.
    #[pyo3(get)]
    pub p_rms_field: Option<Py<PyArray3<f64>>>,
    /// Full-grid final-time pressure snapshot [Pa]. Shape `(nx, ny, nz)`.
    #[pyo3(get)]
    pub p_final_field: Option<Py<PyArray3<f64>>>,

    // ── Particle velocity time series ────────────────────────────────────────
    /// Staggered ux time series: `(n_sensors, time_steps)` [m/s] (None if not requested)
    #[pyo3(get)]
    pub ux: Option<Py<PyArray2<f64>>>,
    /// Staggered uy time series: `(n_sensors, time_steps)` [m/s] (None if not requested)
    #[pyo3(get)]
    pub uy: Option<Py<PyArray2<f64>>>,
    /// Staggered uz time series: `(n_sensors, time_steps)` [m/s] (None if not requested)
    #[pyo3(get)]
    pub uz: Option<Py<PyArray2<f64>>>,
    /// Acoustic x-intensity time series: `p * ux` [W/m^2] (None if not requested)
    #[pyo3(get)]
    pub ix: Option<Py<PyArray2<f64>>>,
    /// Acoustic y-intensity time series: `p * uy` [W/m^2] (None if not requested)
    #[pyo3(get)]
    pub iy: Option<Py<PyArray2<f64>>>,
    /// Acoustic z-intensity time series: `p * uz` [W/m^2] (None if not requested)
    #[pyo3(get)]
    pub iz: Option<Py<PyArray2<f64>>>,
    /// Time-averaged x-intensity at each sensor [W/m^2] (None if not requested)
    #[pyo3(get)]
    pub i_avg_x: Option<Py<PyArray1<f64>>>,
    /// Time-averaged y-intensity at each sensor [W/m^2] (None if not requested)
    #[pyo3(get)]
    pub i_avg_y: Option<Py<PyArray1<f64>>>,
    /// Time-averaged z-intensity at each sensor [W/m^2] (None if not requested)
    #[pyo3(get)]
    pub i_avg_z: Option<Py<PyArray1<f64>>>,

    // ── Velocity statistics ──────────────────────────────────────────────────
    /// Maximum ux at each sensor position over all time steps [m/s] (None if not requested)
    #[pyo3(get)]
    pub ux_max: Option<Py<PyArray1<f64>>>,
    /// Minimum ux at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    pub ux_min: Option<Py<PyArray1<f64>>>,
    /// RMS ux at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    pub ux_rms: Option<Py<PyArray1<f64>>>,
    /// Maximum uy at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    pub uy_max: Option<Py<PyArray1<f64>>>,
    /// Minimum uy at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    pub uy_min: Option<Py<PyArray1<f64>>>,
    /// RMS uy at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    pub uy_rms: Option<Py<PyArray1<f64>>>,
    /// Maximum uz at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    pub uz_max: Option<Py<PyArray1<f64>>>,
    /// Minimum uz at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    pub uz_min: Option<Py<PyArray1<f64>>>,
    /// RMS uz at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    pub uz_rms: Option<Py<PyArray1<f64>>>,
    /// Final temperature field (nx, ny, nz) [°C]. `None` for acoustic-only runs.
    #[pyo3(get)]
    pub thermal_temperature: Option<Py<PyArray3<f64>>>,
    /// CEM43 thermal dose field (nx, ny, nz) [min]. `None` when not requested.
    #[pyo3(get)]
    pub thermal_dose: Option<Py<PyArray3<f64>>>,
}

#[pymethods]
impl SimulationResult {
    /// Get the sensor data as a numpy array.
    /// Returns a 1D array for single-sensor simulations, 2D (n_sensors, n_timesteps) for multi-sensor.
    #[getter]
    fn sensor_data<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        if let Some(ref data_2d) = self.sensor_data_2d {
            data_2d.clone_ref(py).into_any()
        } else if let Some(ref data_1d) = self.sensor_data_1d {
            data_1d.clone_ref(py).into_any()
        } else {
            py.None()
        }
    }

    /// Number of sensor points.
    #[getter]
    fn num_sensors(&self) -> usize {
        self.sensor_data_shape.0
    }

    /// String representation.
    fn __repr__(&self) -> String {
        let data_desc = if self.sensor_data_2d.is_some() {
            "multi-sensor 2D"
        } else {
            "single-sensor 1D"
        };
        format!(
            "SimulationResult(data={}, shape={:?}, time_steps={}, dt={:.2e}, final_time={:.2e})",
            data_desc, self.shape, self.time_steps, self.dt, self.final_time
        )
    }
}

// ============================================================================
// Builder: convert kwavers SimulationRunResult → SimulationResult
// ============================================================================

/// Build a [`SimulationResult`] from the kwavers runner output.
pub(crate) fn build_simulation_result(
    py: Python<'_>,
    result: &KwaversSimulationRunResult,
    grid: &KwaversGrid,
    time_steps: usize,
    dt: f64,
) -> PyResult<SimulationResult> {
    let ny = grid.ny.max(1);
    let nz = grid.nz.max(1);
    let nx = grid.nx;

    let time_vec = ndarray::Array1::linspace(0.0, dt * (time_steps as f64 - 1.0), time_steps);
    let time_array = PyArray1::from_owned_array(py, time_vec).unbind();

    let final_time = dt * (time_steps as f64 - 1.0);

    // Sensor data
    let n_sensors = result.sensor_data.nrows();
    let (sensor_data_1d, sensor_data_2d) = if n_sensors == 1 {
        let row = result.sensor_data.row(0).to_owned();
        (Some(PyArray1::from_owned_array(py, row).unbind()), None)
    } else {
        (None, Some(PyArray2::from_owned_array(py, result.sensor_data.clone()).unbind()))
    };

    // Sampled statistics
    let (p_max, p_min, p_rms, p_final) = if let Some(ref stats) = result.stats {
        (
            Some(PyArray1::from_owned_array(py, stats.p_max.clone()).unbind()),
            Some(PyArray1::from_owned_array(py, stats.p_min.clone()).unbind()),
            Some(PyArray1::from_owned_array(py, stats.p_rms.clone()).unbind()),
            Some(PyArray1::from_owned_array(py, stats.p_final.clone()).unbind()),
        )
    } else {
        (None, None, None, None)
    };

    // Full-grid statistics
    let (p_max_field, p_min_field, p_rms_field, p_final_field) =
        if let Some((ref pmax, ref pmin, ref prms, ref pfinal)) = result.full_grid_stats {
            (
                Some(PyArray3::from_owned_array(py, pmax.clone()).unbind()),
                Some(PyArray3::from_owned_array(py, pmin.clone()).unbind()),
                Some(PyArray3::from_owned_array(py, prms.clone()).unbind()),
                Some(PyArray3::from_owned_array(py, pfinal.clone()).unbind()),
            )
        } else {
            (None, None, None, None)
        };

    // Velocity time series
    let ux = result.ux_data.as_ref().map(|d| PyArray2::from_owned_array(py, d.clone()).unbind());
    let uy = result.uy_data.as_ref().map(|d| PyArray2::from_owned_array(py, d.clone()).unbind());
    let uz = result.uz_data.as_ref().map(|d| PyArray2::from_owned_array(py, d.clone()).unbind());
    let ix = result.ix_data.as_ref().map(|d| PyArray2::from_owned_array(py, d.clone()).unbind());
    let iy = result.iy_data.as_ref().map(|d| PyArray2::from_owned_array(py, d.clone()).unbind());
    let iz = result.iz_data.as_ref().map(|d| PyArray2::from_owned_array(py, d.clone()).unbind());
    let i_avg_x = result.i_avg_x.as_ref().map(|d| PyArray1::from_owned_array(py, d.clone()).unbind());
    let i_avg_y = result.i_avg_y.as_ref().map(|d| PyArray1::from_owned_array(py, d.clone()).unbind());
    let i_avg_z = result.i_avg_z.as_ref().map(|d| PyArray1::from_owned_array(py, d.clone()).unbind());

    // Velocity statistics
    let (ux_max, ux_min, ux_rms, uy_max, uy_min, uy_rms, uz_max, uz_min, uz_rms) =
        if let Some(ref vstats) = result.velocity_stats {
            (
                Some(PyArray1::from_owned_array(py, vstats.ux_max.clone()).unbind()),
                Some(PyArray1::from_owned_array(py, vstats.ux_min.clone()).unbind()),
                Some(PyArray1::from_owned_array(py, vstats.ux_rms.clone()).unbind()),
                Some(PyArray1::from_owned_array(py, vstats.uy_max.clone()).unbind()),
                Some(PyArray1::from_owned_array(py, vstats.uy_min.clone()).unbind()),
                Some(PyArray1::from_owned_array(py, vstats.uy_rms.clone()).unbind()),
                Some(PyArray1::from_owned_array(py, vstats.uz_max.clone()).unbind()),
                Some(PyArray1::from_owned_array(py, vstats.uz_min.clone()).unbind()),
                Some(PyArray1::from_owned_array(py, vstats.uz_rms.clone()).unbind()),
            )
        } else {
            (None, None, None, None, None, None, None, None, None)
        };

    // Thermal fields
    let thermal_temperature = result
        .thermal_temperature
        .as_ref()
        .map(|d| PyArray3::from_owned_array(py, d.clone()).unbind());
    let thermal_dose = result
        .thermal_dose
        .as_ref()
        .map(|d| PyArray3::from_owned_array(py, d.clone()).unbind());

    Ok(SimulationResult {
        sensor_data_1d,
        sensor_data_2d,
        time: time_array,
        shape: (nx, ny, nz),
        sensor_data_shape: (n_sensors, result.sensor_data.ncols()),
        time_steps,
        dt,
        final_time,
        p_max,
        p_min,
        p_rms,
        p_final,
        p_max_field,
        p_min_field,
        p_rms_field,
        p_final_field,
        ux,
        uy,
        uz,
        ix,
        iy,
        iz,
        i_avg_x,
        i_avg_y,
        i_avg_z,
        ux_max,
        ux_min,
        ux_rms,
        uy_max,
        uy_min,
        uy_rms,
        uz_max,
        uz_min,
        uz_rms,
        thermal_temperature,
        thermal_dose,
    })
}

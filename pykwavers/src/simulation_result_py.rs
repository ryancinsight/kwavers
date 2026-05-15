//! `SimulationResult` pyclass and supporting types for simulation output.

use kwavers::domain::sensor::recorder::pressure_statistics::SampledStatistics;
use kwavers::domain::sensor::recorder::simple::SensorRecorder;
use kwavers::domain::sensor::recorder::velocity_statistics::SampledVelocityStats;
use ndarray::{Array3};
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::PyAny;

// ============================================================================
// Internal run result bundle
// ============================================================================

/// Bundle returned by every `run_*_impl` function.
///
/// Velocity fields are `None` unless the caller supplied a `record_modes` list
/// that includes at least one velocity component (e.g. `"ux"`, `"ux_max"`).
/// The FDTD path never populates velocity fields; the PSTD path does.
/// Extract full-grid `(p_max, p_min, p_rms, p_final)` from a recorder if
/// any pressure-statistics mode was requested. Returns `None` otherwise.
pub(crate) fn extract_full_grid_stats(
    recorder: &SensorRecorder,
) -> Option<(Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>)> {
    let stats = recorder.full_pressure_statistics()?;
    Some((
        stats.get_p_max().clone(),
        stats.get_p_min().clone(),
        stats.p_rms(),
        stats.get_p_final().clone(),
    ))
}

pub(crate) struct SimulationRunResult {
    /// Pressure time series at sensor positions: `(n_sensors, time_steps)`.
    pub sensor_data: ndarray::Array2<f64>,
    /// Pressure spatial statistics (p_max/min/rms/final sampled at sensors).
    pub stats: Option<SampledStatistics>,
    /// Staggered ux time series at sensor positions: `(n_sensors, time_steps)`.
    pub ux_data: Option<ndarray::Array2<f64>>,
    /// Staggered uy time series at sensor positions.
    pub uy_data: Option<ndarray::Array2<f64>>,
    /// Staggered uz time series at sensor positions.
    pub uz_data: Option<ndarray::Array2<f64>>,
    /// Acoustic x-intensity time series at sensor positions.
    pub ix_data: Option<ndarray::Array2<f64>>,
    /// Acoustic y-intensity time series at sensor positions.
    pub iy_data: Option<ndarray::Array2<f64>>,
    /// Acoustic z-intensity time series at sensor positions.
    pub iz_data: Option<ndarray::Array2<f64>>,
    /// Time-averaged x-intensity at sensor positions.
    pub i_avg_x: Option<ndarray::Array1<f64>>,
    /// Time-averaged y-intensity at sensor positions.
    pub i_avg_y: Option<ndarray::Array1<f64>>,
    /// Time-averaged z-intensity at sensor positions.
    pub i_avg_z: Option<ndarray::Array1<f64>>,
    /// Per-component velocity statistics sampled at sensor positions.
    pub velocity_stats: Option<SampledVelocityStats>,
    /// Full-grid pressure-statistics field (kernel-generation use).
    /// `(p_max, p_min, p_rms, p_final)` — each `Array3<f64>` shape
    /// `(nx, ny, nz)`. `None` when no `p_*` mode was requested.
    pub full_grid_stats: Option<(Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>)>,
}

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

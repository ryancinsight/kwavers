//! `Sensor` pyclass — acoustic field recorder configuration.

use leto::Array3;
use numpy::PyReadonlyArray3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Sensor for recording acoustic fields.
///
/// Mathematical Specification:
/// - Point sensor: p(t) at fixed location (x₀, y₀, z₀)
/// - Mask sensor: p(t) at multiple positions defined by binary mask
/// - Grid sensor: p(x, y, z, t) on entire grid
/// - Interpolation: trilinear for arbitrary positions
///
/// Equivalent to k-Wave sensor struct.
///
/// # `record_start_index` convention
///
/// Mirrors k-Wave's `sensor.record_start_index` (1-based).  The default `1`
/// records from the first time step.  Setting it to `N` (1 ≤ N ≤ Nt) starts
/// recording at step N and the output has `Nt - N + 1` columns.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Sensor {
    /// Sensor type
    pub(crate) sensor_type: String,
    /// Position for point sensor
    pub(crate) position: Option<[f64; 3]>,
    /// Binary mask for mask-based sensors
    pub(crate) mask: Option<Array3<bool>>,
    /// k-Wave-style recording mode strings (e.g. ["p", "p_max", "p_rms"])
    pub(crate) record_modes: Vec<String>,
    /// k-Wave 1-based start step for recording (default 1 = all steps).
    pub(crate) record_start_index: usize,
}

#[pymethods]
impl Sensor {
    /// Create a point sensor at specified position.
    ///
    /// Parameters
    /// ----------
    /// position : tuple
    ///     Sensor position [x, y, z] in meters
    ///
    /// Returns
    /// -------
    /// Sensor
    ///     Point sensor
    ///
    /// Examples
    /// --------
    /// >>> sensor = Sensor.point([0.02, 0.02, 0.02])
    #[staticmethod]
    fn point(position: (f64, f64, f64)) -> Self {
        Sensor {
            sensor_type: "point".to_string(),
            position: Some([position.0, position.1, position.2]),
            mask: None,
            record_modes: Vec::new(),
            record_start_index: 1,
        }
    }

    /// Create a mask-based sensor from a binary 3D mask.
    ///
    /// Records pressure at all True positions in the mask.
    /// Equivalent to k-Wave's kSensor(mask).
    ///
    /// Parameters
    /// ----------
    /// mask : ndarray
    ///     3D boolean mask (same shape as grid)
    ///
    /// Returns
    /// -------
    /// Sensor
    ///     Mask-based sensor recording at multiple points
    ///
    /// Examples
    /// --------
    /// >>> mask = np.zeros((32, 32, 32), dtype=bool)
    /// >>> mask[16, 16, 16] = True
    /// >>> sensor = Sensor.from_mask(mask)
    #[staticmethod]
    fn from_mask(mask: PyReadonlyArray3<bool>) -> PyResult<Self> {
        let mask_arr = mask.as_array().to_owned();
        if mask_arr.ndim() != 3 {
            return Err(PyValueError::new_err("Mask must be a 3D array"));
        }
        let num_sensors = mask_arr.iter().filter(|&&v| v).count();
        if num_sensors == 0 {
            return Err(PyValueError::new_err(
                "Sensor mask must have at least one active sensor",
            ));
        }
        Ok(Sensor {
            sensor_type: "mask".to_string(),
            position: None,
            mask: Some(crate::breast_fwi_bindings::complex_compat::nd_to_leto3(
                mask_arr,
            )),
            record_modes: Vec::new(),
            record_start_index: 1,
        })
    }

    /// Create a grid sensor recording entire field.
    ///
    /// Returns
    /// -------
    /// Sensor
    ///     Grid sensor
    ///
    /// Examples
    /// --------
    /// >>> sensor = Sensor.grid()
    #[staticmethod]
    fn grid() -> Self {
        Sensor {
            sensor_type: "grid".to_string(),
            position: None,
            mask: None,
            record_modes: Vec::new(),
            record_start_index: 1,
        }
    }

    /// Set k-Wave-style recording modes.
    ///
    /// Parameters
    /// ----------
    /// modes : list[str]
    ///     Recording mode strings. Supported: "p", "p_max", "p_min", "p_rms", "p_final", "all"
    ///
    /// Examples
    /// --------
    /// >>> sensor = Sensor.from_mask(mask)
    /// >>> sensor.set_record(["p", "p_max", "p_rms"])
    fn set_record(&mut self, modes: Vec<String>) {
        self.record_modes = modes;
    }

    /// Get current recording modes.
    #[getter]
    fn record(&self) -> Vec<String> {
        self.record_modes.clone()
    }

    /// Set the first time step to record (k-Wave 1-based convention).
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     First time step at which to start recording (≥ 1).
    ///     Equivalent to k-Wave `sensor.record_start_index`.
    ///     Setting this to `N` yields output with `Nt - N + 1` time samples.
    ///
    /// Examples
    /// --------
    /// >>> # Record only the last 300 time steps of a 1000-step simulation
    /// >>> sensor.set_record_start_index(701)
    fn set_record_start_index(&mut self, index: usize) -> PyResult<()> {
        if index < 1 {
            return Err(PyValueError::new_err(
                "record_start_index must be ≥ 1 (k-Wave 1-based convention)",
            ));
        }
        self.record_start_index = index;
        Ok(())
    }

    /// First time step to record (k-Wave 1-based, default 1).
    #[getter]
    fn record_start_index(&self) -> usize {
        self.record_start_index
    }

    /// Number of active sensor points.
    #[getter]
    fn num_sensors(&self) -> usize {
        match &self.mask {
            Some(m) => m.iter().filter(|&&v| v).count(),
            None if self.sensor_type == "point" => 1,
            _ => 0,
        }
    }

    /// Sensor type.
    #[getter]
    fn sensor_type(&self) -> &str {
        &self.sensor_type
    }

    /// String representation.
    fn __repr__(&self) -> String {
        match &self.sensor_type {
            t if t == "point" => {
                let pos = self.position.unwrap_or([0.0, 0.0, 0.0]);
                format!(
                    "Sensor.point(position=[{:.3e}, {:.3e}, {:.3e}])",
                    pos[0], pos[1], pos[2]
                )
            }
            t if t == "mask" => {
                let n = self
                    .mask
                    .as_ref()
                    .map_or(0, |m| m.iter().filter(|&&v| v).count());
                format!("Sensor.from_mask(num_sensors={})", n)
            }
            _ => "Sensor.grid()".to_string(),
        }
    }
}

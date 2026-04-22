use ndarray::Array3;
use numpy::PyReadonlyArray3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Sensor {
    pub(crate) sensor_type: String,
    pub(crate) position: Option<[f64; 3]>,
    pub(crate) mask: Option<Array3<bool>>,
    pub(crate) record_modes: Vec<String>,
}

#[pymethods]
impl Sensor {
    #[staticmethod]
    fn point(position: (f64, f64, f64)) -> Self {
        Self {
            sensor_type: "point".to_string(),
            position: Some([position.0, position.1, position.2]),
            mask: None,
            record_modes: Vec::new(),
        }
    }

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
        Ok(Self {
            sensor_type: "mask".to_string(),
            position: None,
            mask: Some(mask_arr),
            record_modes: Vec::new(),
        })
    }

    #[staticmethod]
    fn grid() -> Self {
        Self {
            sensor_type: "grid".to_string(),
            position: None,
            mask: None,
            record_modes: Vec::new(),
        }
    }

    fn set_record(&mut self, modes: Vec<String>) {
        self.record_modes = modes;
    }

    #[getter]
    fn record(&self) -> Vec<String> {
        self.record_modes.clone()
    }

    #[getter]
    fn num_sensors(&self) -> usize {
        match &self.mask {
            Some(m) => m.iter().filter(|&&v| v).count(),
            None if self.sensor_type == "point" => 1,
            _ => 0,
        }
    }

    #[getter]
    fn sensor_type(&self) -> &str {
        &self.sensor_type
    }

    pub(crate) fn __repr__(&self) -> String {
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

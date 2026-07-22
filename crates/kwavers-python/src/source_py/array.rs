use numpy::ndarray::Axis;
use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::Source;
use crate::breast_fwi_bindings::complex_compat::{leto3_to_nd3, nd_to_leto2};
use crate::kwave_array_py::KWaveArray;

#[pymethods]
impl Source {
    /// Source frequency `Hz`.
    #[getter]
    fn frequency(&self) -> f64 {
        self.frequency
    }

    /// Source amplitude `Pa`.
    #[getter]
    fn amplitude(&self) -> f64 {
        self.amplitude
    }

    /// Source type.
    #[getter]
    fn source_type(&self) -> &str {
        &self.source_type
    }

    /// Source injection mode ("additive", "additive_no_correction", or "dirichlet").
    #[getter]
    fn source_mode(&self) -> &str {
        &self.source_mode
    }

    /// Initial pressure field, if this source was constructed from `p0`.
    #[getter]
    fn initial_pressure<'py>(&self, py: Python<'py>) -> Option<Py<PyArray3<f64>>> {
        self.initial_pressure
            .as_ref()
            .map(|arr| PyArray3::from_owned_array(py, leto3_to_nd3(arr.clone())).into())
    }

    /// Create a source from a KWaveArray with a driving signal.
    #[staticmethod]
    #[pyo3(signature = (array, signal, frequency, mode=None))]
    fn from_kwave_array(
        array: &KWaveArray,
        signal: PyReadonlyArray1<f64>,
        frequency: f64,
        mode: Option<&str>,
    ) -> PyResult<Self> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Frequency must be positive"));
        }
        let signal_arr = signal.as_array().to_owned();
        if signal_arr.is_empty() {
            return Err(PyValueError::new_err("Signal must not be empty"));
        }
        let signal_matrix = nd_to_leto2(signal_arr.clone().insert_axis(Axis(0)).to_owned());
        let source_mode = match mode {
            Some("additive_no_correction") => "additive_no_correction".to_string(),
            Some("dirichlet") => "dirichlet".to_string(),
            _ => "additive".to_string(),
        };
        let amplitude = signal_arr.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        Ok(Source {
            source_type: "kwave_array".to_string(),
            frequency,
            amplitude,
            position: None,
            mask: None,
            signal: Some(signal_matrix),
            source_mode,
            initial_pressure: None,
            velocity_signal: None,
            direction: None,
            kwave_array: Some(array.inner.clone()),
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
        })
    }

    /// Create a source from a KWaveArray with per-element driving signals.
    #[staticmethod]
    #[pyo3(signature = (array, signals, frequency, mode=None))]
    fn from_kwave_array_per_element(
        array: &KWaveArray,
        signals: PyReadonlyArray2<f64>,
        frequency: f64,
        mode: Option<&str>,
    ) -> PyResult<Self> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Frequency must be positive"));
        }
        let signal_matrix = signals.as_array().to_owned();
        if signal_matrix.is_empty() {
            return Err(PyValueError::new_err("Signals must not be empty"));
        }
        let n_elements = array.inner.num_elements();
        if signal_matrix.shape()[0] != n_elements {
            return Err(PyValueError::new_err(format!(
                "signals has {} rows but array has {} elements",
                signal_matrix.shape()[0],
                n_elements
            )));
        }
        let source_mode = match mode {
            Some("additive_no_correction") => "additive_no_correction".to_string(),
            Some("dirichlet") => "dirichlet".to_string(),
            _ => "additive".to_string(),
        };
        let amplitude = signal_matrix
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        Ok(Source {
            source_type: "kwave_array_per_element".to_string(),
            frequency,
            amplitude,
            position: None,
            mask: None,
            signal: Some(nd_to_leto2(signal_matrix)),
            source_mode,
            initial_pressure: None,
            velocity_signal: None,
            direction: None,
            kwave_array: Some(array.inner.clone()),
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
        })
    }

    fn __repr__(&self) -> String {
        match &self.position {
            Some(pos) => format!(
                "Source.point(position=[{:.3e}, {:.3e}, {:.3e}], frequency={:.2e}, amplitude={:.2e})",
                pos[0], pos[1], pos[2], self.frequency, self.amplitude
            ),
            None if self.source_type == "mask" => format!(
                "Source.from_mask(frequency={:.2e}, amplitude={:.2e})",
                self.frequency, self.amplitude
            ),
            None if self.source_type == "kwave_array" => format!(
                "Source.from_kwave_array(frequency={:.2e}, amplitude={:.2e})",
                self.frequency, self.amplitude
            ),
            None => format!(
                "Source.plane_wave(frequency={:.2e}, amplitude={:.2e})",
                self.frequency, self.amplitude
            ),
        }
    }
}

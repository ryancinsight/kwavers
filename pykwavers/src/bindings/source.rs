use crate::{Grid, KWaveArray};
use ndarray::{Array1, Array3};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Source {
    pub(crate) source_type: String,
    pub(crate) frequency: f64,
    pub(crate) amplitude: f64,
    pub(crate) position: Option<[f64; 3]>,
    pub(crate) mask: Option<Array3<f64>>,
    pub(crate) signal: Option<Array1<f64>>,
    pub(crate) source_mode: String,
    pub(crate) initial_pressure: Option<Array3<f64>>,
    pub(crate) velocity_signal: Option<ndarray::Array3<f64>>,
    pub(crate) direction: Option<(f64, f64, f64)>,
    pub(crate) kwave_array: Option<kwavers::domain::source::kwave_array::KWaveArray>,
}

#[pymethods]
impl Source {
    #[staticmethod]
    #[pyo3(signature = (grid, frequency, amplitude, direction=None))]
    fn plane_wave(
        grid: &Grid,
        frequency: f64,
        amplitude: f64,
        direction: Option<(f64, f64, f64)>,
    ) -> PyResult<Self> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Frequency must be positive"));
        }
        if amplitude < 0.0 {
            return Err(PyValueError::new_err("Amplitude must be non-negative"));
        }

        let dir = direction.unwrap_or((0.0, 0.0, 1.0));
        let mag = (dir.0 * dir.0 + dir.1 * dir.1 + dir.2 * dir.2).sqrt();
        if mag < 1e-12 {
            return Err(PyValueError::new_err("Direction vector must be non-zero"));
        }

        let _ = &grid.inner;
        Ok(Self {
            source_type: "plane_wave".to_string(),
            frequency,
            amplitude,
            position: None,
            mask: None,
            signal: None,
            source_mode: "additive".to_string(),
            initial_pressure: None,
            velocity_signal: None,
            direction: Some((dir.0 / mag, dir.1 / mag, dir.2 / mag)),
            kwave_array: None,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (position, frequency, amplitude))]
    fn point(position: (f64, f64, f64), frequency: f64, amplitude: f64) -> PyResult<Self> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Frequency must be positive"));
        }
        if amplitude < 0.0 {
            return Err(PyValueError::new_err("Amplitude must be non-negative"));
        }

        Ok(Self {
            source_type: "point".to_string(),
            frequency,
            amplitude,
            position: Some([position.0, position.1, position.2]),
            mask: None,
            signal: None,
            source_mode: "additive".to_string(),
            initial_pressure: None,
            velocity_signal: None,
            direction: None,
            kwave_array: None,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (mask, signal, frequency, mode=None))]
    fn from_mask(
        mask: PyReadonlyArray3<f64>,
        signal: PyReadonlyArray1<f64>,
        frequency: f64,
        mode: Option<&str>,
    ) -> PyResult<Self> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Frequency must be positive"));
        }

        let mask_arr = mask.as_array().to_owned();
        if mask_arr.ndim() != 3 {
            return Err(PyValueError::new_err("Mask must be a 3D array"));
        }

        let signal_arr = signal.as_array().to_owned();
        if signal_arr.ndim() != 1 {
            return Err(PyValueError::new_err("Signal must be a 1D array"));
        }
        if signal_arr.is_empty() {
            return Err(PyValueError::new_err("Signal must not be empty"));
        }

        let source_mode = match mode {
            Some("additive_no_correction") => "additive_no_correction".to_string(),
            Some("dirichlet") => "dirichlet".to_string(),
            Some("additive") | None => "additive".to_string(),
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "Invalid source mode '{}'. Use 'additive', 'additive_no_correction', or 'dirichlet'",
                    other
                )))
            }
        };

        let amplitude = signal_arr.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        Ok(Self {
            source_type: "mask".to_string(),
            frequency,
            amplitude,
            position: None,
            mask: Some(mask_arr),
            signal: Some(signal_arr),
            source_mode,
            initial_pressure: None,
            velocity_signal: None,
            direction: None,
            kwave_array: None,
        })
    }

    #[staticmethod]
    fn from_initial_pressure(p0: PyReadonlyArray3<f64>) -> PyResult<Self> {
        let p0_arr = p0.as_array().to_owned();
        if p0_arr.iter().all(|&v| v == 0.0) {
            return Err(PyValueError::new_err("Initial pressure is all zeros"));
        }
        let amplitude = p0_arr.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        Ok(Self {
            source_type: "p0".to_string(),
            frequency: 0.0,
            amplitude,
            position: None,
            mask: None,
            signal: None,
            source_mode: "additive".to_string(),
            initial_pressure: Some(p0_arr),
            velocity_signal: None,
            direction: None,
            kwave_array: None,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (mask, ux=None, uy=None, uz=None, mode=None))]
    fn from_velocity_mask(
        mask: PyReadonlyArray3<f64>,
        ux: Option<PyReadonlyArray1<f64>>,
        uy: Option<PyReadonlyArray1<f64>>,
        uz: Option<PyReadonlyArray1<f64>>,
        mode: Option<&str>,
    ) -> PyResult<Self> {
        let mask_arr = mask.as_array().to_owned();
        let num_sources = mask_arr.iter().filter(|v| **v != 0.0).count();
        if num_sources == 0 {
            return Err(PyValueError::new_err(
                "Velocity mask contains no active points",
            ));
        }
        if ux.is_none() && uy.is_none() && uz.is_none() {
            return Err(PyValueError::new_err(
                "At least one velocity component (ux, uy, uz) must be provided",
            ));
        }

        let nt = if let Some(ref s) = ux {
            s.as_array().len()
        } else if let Some(ref s) = uy {
            s.as_array().len()
        } else if let Some(ref s) = uz {
            s.as_array().len()
        } else {
            unreachable!()
        };

        let mut u_signal = ndarray::Array3::<f64>::zeros((3, 1, nt));
        if let Some(ref sx) = ux {
            let arr = sx.as_array();
            for t in 0..nt {
                u_signal[[0, 0, t]] = arr[t];
            }
        }
        if let Some(ref sy) = uy {
            let arr = sy.as_array();
            if arr.len() != nt {
                return Err(PyValueError::new_err(format!(
                    "uy length {} differs from first signal length {}",
                    arr.len(),
                    nt
                )));
            }
            for t in 0..nt {
                u_signal[[1, 0, t]] = arr[t];
            }
        }
        if let Some(ref sz) = uz {
            let arr = sz.as_array();
            if arr.len() != nt {
                return Err(PyValueError::new_err(format!(
                    "uz length {} differs from first signal length {}",
                    arr.len(),
                    nt
                )));
            }
            for t in 0..nt {
                u_signal[[2, 0, t]] = arr[t];
            }
        }

        let source_mode = match mode {
            Some("dirichlet") => "dirichlet".to_string(),
            Some("additive_no_correction") => "additive_no_correction".to_string(),
            _ => "additive".to_string(),
        };

        let max_amp = u_signal.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        Ok(Self {
            source_type: "velocity".to_string(),
            frequency: 0.0,
            amplitude: max_amp,
            position: None,
            mask: Some(mask_arr),
            signal: None,
            source_mode,
            initial_pressure: None,
            velocity_signal: Some(u_signal),
            direction: None,
            kwave_array: None,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (mask, ux=None, uy=None, uz=None, mode=None))]
    fn from_velocity_mask_2d(
        mask: PyReadonlyArray3<f64>,
        ux: Option<PyReadonlyArray2<f64>>,
        uy: Option<PyReadonlyArray2<f64>>,
        uz: Option<PyReadonlyArray2<f64>>,
        mode: Option<&str>,
    ) -> PyResult<Self> {
        let mask_arr = mask.as_array().to_owned();
        let num_sources = mask_arr.iter().filter(|v| **v != 0.0).count();
        if num_sources == 0 {
            return Err(PyValueError::new_err(
                "Velocity mask contains no active points",
            ));
        }
        if ux.is_none() && uy.is_none() && uz.is_none() {
            return Err(PyValueError::new_err(
                "At least one velocity component (ux, uy, uz) must be provided",
            ));
        }

        let (n_sig, nt) = if let Some(ref s) = ux {
            let shape = s.as_array().shape().to_vec();
            (shape[0], shape[1])
        } else if let Some(ref s) = uy {
            let shape = s.as_array().shape().to_vec();
            (shape[0], shape[1])
        } else if let Some(ref s) = uz {
            let shape = s.as_array().shape().to_vec();
            (shape[0], shape[1])
        } else {
            unreachable!()
        };

        if n_sig != num_sources {
            return Err(PyValueError::new_err(format!(
                "Signal rows ({}) must match number of active mask points ({})",
                n_sig, num_sources
            )));
        }

        let mut u_signal = ndarray::Array3::<f64>::zeros((3, num_sources, nt));
        if let Some(ref sx) = ux {
            let arr = sx.as_array();
            for s in 0..num_sources {
                for t in 0..nt {
                    u_signal[[0, s, t]] = arr[[s, t]];
                }
            }
        }
        if let Some(ref sy) = uy {
            let arr = sy.as_array();
            if arr.shape()[0] != num_sources || arr.shape()[1] != nt {
                return Err(PyValueError::new_err(format!(
                    "uy shape {:?} must be ({}, {})",
                    arr.shape(),
                    num_sources,
                    nt
                )));
            }
            for s in 0..num_sources {
                for t in 0..nt {
                    u_signal[[1, s, t]] = arr[[s, t]];
                }
            }
        }
        if let Some(ref sz) = uz {
            let arr = sz.as_array();
            if arr.shape()[0] != num_sources || arr.shape()[1] != nt {
                return Err(PyValueError::new_err(format!(
                    "uz shape {:?} must be ({}, {})",
                    arr.shape(),
                    num_sources,
                    nt
                )));
            }
            for s in 0..num_sources {
                for t in 0..nt {
                    u_signal[[2, s, t]] = arr[[s, t]];
                }
            }
        }

        let source_mode = match mode {
            Some("dirichlet") => "dirichlet".to_string(),
            Some("additive_no_correction") => "additive_no_correction".to_string(),
            _ => "additive".to_string(),
        };

        let max_amp = u_signal.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        Ok(Self {
            source_type: "velocity".to_string(),
            frequency: 0.0,
            amplitude: max_amp,
            position: None,
            mask: Some(mask_arr),
            signal: None,
            source_mode,
            initial_pressure: None,
            velocity_signal: Some(u_signal),
            direction: None,
            kwave_array: None,
        })
    }

    #[getter]
    fn frequency(&self) -> f64 {
        self.frequency
    }

    #[getter]
    fn amplitude(&self) -> f64 {
        self.amplitude
    }

    #[getter]
    fn source_type(&self) -> &str {
        &self.source_type
    }

    #[getter]
    fn source_mode(&self) -> &str {
        &self.source_mode
    }

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
        let source_mode = match mode {
            Some("additive_no_correction") => "additive_no_correction".to_string(),
            Some("dirichlet") => "dirichlet".to_string(),
            _ => "additive".to_string(),
        };
        let amplitude = signal_arr.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        Ok(Self {
            source_type: "kwave_array".to_string(),
            frequency,
            amplitude,
            position: None,
            mask: None,
            signal: Some(signal_arr),
            source_mode,
            initial_pressure: None,
            velocity_signal: None,
            direction: None,
            kwave_array: Some(array.inner.clone()),
        })
    }

    pub(crate) fn __repr__(&self) -> String {
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

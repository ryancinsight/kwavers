use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::Source;

#[pymethods]
impl Source {
    /// Create a velocity source from a spatial mask and directional signals.
    ///
    /// Equivalent to k-Wave's `source.u_mask` / `source.ux` / `source.uy` / `source.uz`.
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

        Ok(Source {
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
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
        })
    }

    /// Create a velocity source from a 3D mask with per-source-point 2D signal arrays.
    ///
    /// Each source point gets a uniquely time-shifted velocity signal, matching
    /// k-wave's NotATransducer focused transducer behavior.
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

        Ok(Source {
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
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
        })
    }
}

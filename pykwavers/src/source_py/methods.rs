use ndarray::Axis;
use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::Source;
use crate::kwave_array_py::KWaveArray;

#[pymethods]
impl Source {
    /// Create a velocity source from a spatial mask and directional signals.
    ///
    /// Equivalent to k-Wave's `source.u_mask` / `source.ux` / `source.uy` / `source.uz`.
    ///
    /// Parameters
    /// ----------
    /// mask : ndarray (3D float64)
    ///     Spatial mask marking velocity source locations (nonzero = active)
    /// ux : ndarray (1D or 2D), optional
    ///     Velocity signal in x-direction [m/s]
    /// uy : ndarray (1D or 2D), optional
    ///     Velocity signal in y-direction [m/s]
    /// uz : ndarray (1D or 2D), optional
    ///     Velocity signal in z-direction [m/s]
    /// mode : str, optional
    ///     Source injection mode: "additive" (default), "additive_no_correction",
    ///     or "dirichlet"
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

        // At least one velocity component must be provided
        if ux.is_none() && uy.is_none() && uz.is_none() {
            return Err(PyValueError::new_err(
                "At least one velocity component (ux, uy, uz) must be provided",
            ));
        }

        // Determine time steps from first available signal
        let nt = if let Some(ref s) = ux {
            s.as_array().len()
        } else if let Some(ref s) = uy {
            s.as_array().len()
        } else if let Some(ref s) = uz {
            s.as_array().len()
        } else {
            unreachable!()
        };

        // Build [3, 1, nt] velocity signal array (broadcast to all sources)
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
    /// This method supports beamforming delays where each source point gets a
    /// uniquely time-shifted velocity signal, matching k-wave's NotATransducer
    /// focused transducer behavior.
    ///
    /// Parameters
    /// ----------
    /// mask : ndarray (3D)
    ///     3D binary mask of source locations (non-zero = source point).
    ///     Source points are iterated in row-major (C-order): x varies slowest.
    /// ux : ndarray (2D), optional
    ///     Velocity signal in x-direction, shape (n_sources, n_timesteps) [m/s]
    /// uy : ndarray (2D), optional
    ///     Velocity signal in y-direction, shape (n_sources, n_timesteps) [m/s]
    /// uz : ndarray (2D), optional
    ///     Velocity signal in z-direction, shape (n_sources, n_timesteps) [m/s]
    /// mode : str, optional
    ///     Source injection mode: "additive" (default), "additive_no_correction",
    ///     or "dirichlet"
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

        // Determine (n_sources, nt) from first available 2D signal
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

        // Build [3, n_sources, nt] velocity signal array
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

    /// Source frequency [Hz].
    #[getter]
    fn frequency(&self) -> f64 {
        self.frequency
    }

    /// Source amplitude [Pa].
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
            .map(|arr| PyArray3::from_owned_array(py, arr.clone()).into())
    }

    /// Create a source from a KWaveArray with a driving signal.
    ///
    /// The array geometry is rasterized onto the simulation grid at run time.
    ///
    /// Parameters
    /// ----------
    /// array : KWaveArray
    ///     Custom transducer array
    /// signal : ndarray
    ///     1D driving signal [Pa]
    /// frequency : float
    ///     Source frequency [Hz]
    /// mode : str, optional
    ///     Source injection mode: "additive" (default), "additive_no_correction", "dirichlet"
    ///
    /// Examples
    /// --------
    /// >>> arr = KWaveArray()
    /// >>> arr.add_disc_element((0.015, 0.015, 0.0), 0.01)
    /// >>> source = Source.from_kwave_array(arr, signal, frequency=1e6)
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
        let signal_matrix = signal_arr.clone().insert_axis(Axis(0)).to_owned();
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
    ///
    /// Parameters
    /// ----------
    /// array : KWaveArray
    /// signals : ndarray, shape (n_elements, n_time)
    ///     Per-element driving waveforms. Matches the output shape of
    ///     `kwave.utils.signals.create_cw_signals(t, f, amps, phases)`.
    /// frequency : float
    /// mode : str, optional
    ///
    /// Notes
    /// -----
    /// At run time, pykwavers pre-expands these into a per-active-cell signal
    /// matrix `s_cell[c, t] = Σ_i W_i[c] · s_i[t]` using each element's BLI
    /// weighted mask in MATLAB / Fortran-order active-cell enumeration,
    /// matching k-wave-python's `get_distributed_source_signal`.
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

    /// String representation.
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

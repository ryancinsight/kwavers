mod array;
mod elastic;
pub(crate) mod helpers;
mod velocity;

pub(crate) use helpers::pressure_signal_to_matrix;

use ndarray::{Array2, Array3, Axis};
use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Acoustic source for wave excitation.
///
/// Mathematical Specification:
/// - Pressure source: p(x, t) = A·sin(2πft + φ)
/// - Velocity source: v(x, t) = A·sin(2πft + φ)
/// - Initial pressure: p(x, 0) = p₀(x)
///
/// Equivalent to k-Wave source struct.
#[pyclass]
#[derive(Clone)]
pub struct Source {
    /// Source type identifier
    pub(crate) source_type: String,
    /// Frequency [Hz]
    pub(crate) frequency: f64,
    /// Amplitude [Pa] or [m/s]
    pub(crate) amplitude: f64,
    /// Position for point source
    pub(crate) position: Option<[f64; 3]>,
    /// Spatial mask for grid sources
    pub(crate) mask: Option<Array3<f64>>,
    /// Time signal matrix for grid sources (pressure), shape `[num_sources, time_steps]`
    pub(crate) signal: Option<Array2<f64>>,
    /// Source injection mode ("additive", "additive_no_correction", or "dirichlet")
    pub(crate) source_mode: String,
    /// Initial pressure distribution (for p0 / IVP sources)
    pub(crate) initial_pressure: Option<Array3<f64>>,
    /// Velocity signal [3, num_sources, time_steps] for velocity sources
    pub(crate) velocity_signal: Option<ndarray::Array3<f64>>,
    /// Propagation direction for plane wave sources
    pub(crate) direction: Option<(f64, f64, f64)>,
    /// KWaveArray for custom transducer geometry sources
    pub(crate) kwave_array: Option<kwavers_domain::source::kwave_array::KWaveArray>,
    /// Per-axis 1-D velocity-signal time series for the elastic
    /// velocity-source path (Phase A.3 of ADR 007). Each entry is `Some`
    /// when the corresponding component is to be driven; `None` otherwise.
    /// The `mask` field above carries the `u_mask` for this source path.
    pub(crate) elastic_ux_signal_1d: Option<ndarray::Array1<f64>>,
    pub(crate) elastic_uy_signal_1d: Option<ndarray::Array1<f64>>,
    pub(crate) elastic_uz_signal_1d: Option<ndarray::Array1<f64>>,
}

#[pymethods]
impl Source {
    /// Create a plane wave source.
    ///
    /// Parameters
    /// ----------
    /// grid : Grid
    ///     Computational grid
    /// frequency : float
    ///     Source frequency [Hz]
    /// amplitude : float
    ///     Pressure amplitude [Pa]
    /// direction : tuple, optional
    ///     Propagation direction (default: [0, 0, 1] = +z)
    ///
    /// Returns
    /// -------
    /// Source
    ///     Plane wave source
    ///
    /// Examples
    /// --------
    /// >>> source = Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
    #[staticmethod]
    #[pyo3(signature = (grid, frequency, amplitude, direction=None))]
    fn plane_wave(
        grid: &crate::grid_py::Grid,
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

        // Validate and normalize direction
        let dir = direction.unwrap_or((0.0, 0.0, 1.0));
        let mag = (dir.0 * dir.0 + dir.1 * dir.1 + dir.2 * dir.2).sqrt();
        if mag < 1e-12 {
            return Err(PyValueError::new_err("Direction vector must be non-zero"));
        }
        let norm_dir = (dir.0 / mag, dir.1 / mag, dir.2 / mag);
        let _ = &grid.inner; // retained for wavelength computation in future

        Ok(Source {
            source_type: "plane_wave".to_string(),
            frequency,
            amplitude,
            position: None,
            mask: None,
            signal: None,
            source_mode: "additive".to_string(),
            initial_pressure: None,
            velocity_signal: None,
            direction: Some(norm_dir),
            kwave_array: None,
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
        })
    }

    /// Create a point source.
    ///
    /// Parameters
    /// ----------
    /// position : tuple
    ///     Source position [x, y, z] in meters
    /// frequency : float
    ///     Source frequency [Hz]
    /// amplitude : float
    ///     Pressure amplitude [Pa]
    ///
    /// Returns
    /// -------
    /// Source
    ///     Point source
    ///
    /// Examples
    /// --------
    /// >>> source = Source.point([0.01, 0.01, 0.01], frequency=1e6, amplitude=1e5)
    #[staticmethod]
    #[pyo3(signature = (position, frequency, amplitude))]
    fn point(position: (f64, f64, f64), frequency: f64, amplitude: f64) -> PyResult<Self> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Frequency must be positive"));
        }
        if amplitude < 0.0 {
            return Err(PyValueError::new_err("Amplitude must be non-negative"));
        }

        Ok(Source {
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
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
        })
    }

    /// Create a grid source from a spatial mask and time signal.
    ///
    /// Parameters
    /// ----------
    /// mask : ndarray
    ///     3D spatial mask (same shape as grid)
    /// signal : ndarray
    ///     1D time signal [Pa] or 2D matrix `[num_sources, time_steps]`
    ///     For multi-row pressure sources, rows must follow MATLAB / Fortran-
    ///     order active-point enumeration to match k-wave-python.
    /// frequency : float
    ///     Source frequency [Hz]
    /// mode : str, optional
    ///     Source injection mode: "additive" (default), "additive_no_correction", or "dirichlet"
    #[staticmethod]
    #[pyo3(signature = (mask, signal, frequency, mode=None))]
    fn from_mask(
        mask: PyReadonlyArray3<f64>,
        signal: &Bound<'_, PyAny>,
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

        let signal_arr = pressure_signal_to_matrix(signal)?;

        let source_mode = match mode {
            Some("additive_no_correction") => "additive_no_correction".to_string(),
            Some("dirichlet") => "dirichlet".to_string(),
            Some("additive") | None => "additive".to_string(),
            Some(other) => return Err(PyValueError::new_err(format!(
                "Invalid source mode '{}'. Use 'additive', 'additive_no_correction', or 'dirichlet'",
                other
            ))),
        };

        let num_sources = mask_arr.iter().filter(|&&v| v != 0.0).count();
        if num_sources == 0 {
            return Err(PyValueError::new_err(
                "Source mask contains no active points",
            ));
        }

        let n_signal_rows = signal_arr.shape()[0];
        if n_signal_rows != 1 && n_signal_rows != num_sources {
            return Err(PyValueError::new_err(format!(
                "Signal rows must be 1 or match active source points: got {}, expected 1 or {}",
                n_signal_rows, num_sources
            )));
        }

        let amplitude = signal_arr.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));

        Ok(Source {
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
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
        })
    }

    /// Create an initial pressure (initial value problem) source.
    ///
    /// Equivalent to k-Wave's `source.p0`.
    ///
    /// Parameters
    /// ----------
    /// p0 : ndarray
    ///     2D or 3D initial pressure distribution [Pa]. A 2D field is lifted
    ///     to a single-slice 3D volume with `nz=1` to match the solver layout.
    #[staticmethod]
    fn from_initial_pressure(p0: &Bound<'_, PyAny>) -> PyResult<Self> {
        let p0_arr: Array3<f64> = if let Ok(p0_3d) = p0.extract::<PyReadonlyArray3<f64>>() {
            p0_3d.as_array().to_owned()
        } else if let Ok(p0_2d) = p0.extract::<PyReadonlyArray2<f64>>() {
            p0_2d.as_array().insert_axis(Axis(2)).to_owned()
        } else {
            return Err(PyValueError::new_err(
                "Initial pressure must be a 2D or 3D ndarray of float64 values",
            ));
        };
        if p0_arr.iter().all(|&v| v == 0.0) {
            return Err(PyValueError::new_err("Initial pressure is all zeros"));
        }
        let amplitude = p0_arr.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        Ok(Source {
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
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
        })
    }
}

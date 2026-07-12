use leto::Array3;
use numpy::ndarray::Axis;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::Source;
use crate::breast_fwi_bindings::complex_compat::nd_to_leto3;

#[pymethods]
impl Source {
    /// Create a particle-velocity source mask for the **elastic** solver.
    ///
    /// Equivalent to k-Wave's ``source.u_mask`` / ``source.ux`` /
    /// ``source.uy`` / ``source.uz`` inputs to ``pstdElastic2D`` /
    /// ``pstdElastic3D``. At each time step, the integrator's post-step
    /// velocity field is **assigned** at every grid point inside ``mask``
    /// with the supplied component signal sample for that step (Dirichlet
    /// override semantics — matches k-Wave's default for velocity sources
    /// in pstdElastic).
    ///
    /// Phase A.3 of ADR 007. Signals are 1-D ndarrays (broadcast across
    /// all mask points); per-point signal matrices ship in Phase A.4.
    #[staticmethod]
    #[pyo3(signature = (mask, ux=None, uy=None, uz=None, mode=None))]
    fn from_elastic_velocity_source(
        mask: PyReadonlyArray3<bool>,
        ux: Option<PyReadonlyArray1<f64>>,
        uy: Option<PyReadonlyArray1<f64>>,
        uz: Option<PyReadonlyArray1<f64>>,
        mode: Option<&str>,
    ) -> PyResult<Self> {
        let normalised_mode = match mode.unwrap_or("additive").to_ascii_lowercase().as_str() {
            "additive" => "additive",
            "dirichlet" => "dirichlet",
            other => {
                return Err(PyValueError::new_err(format!(
                    "mode must be 'additive' or 'dirichlet'; got '{}'",
                    other
                )));
            }
        };
        let mask_arr = mask.as_array();
        if mask_arr.ndim() != 3 {
            return Err(PyValueError::new_err("mask must be a 3D bool ndarray"));
        }
        let n_active = mask_arr.iter().filter(|&&v| v).count();
        if n_active == 0 {
            return Err(PyValueError::new_err(
                "mask must have at least one active point",
            ));
        }
        if ux.is_none() && uy.is_none() && uz.is_none() {
            return Err(PyValueError::new_err(
                "At least one of ux, uy, uz must be provided",
            ));
        }
        let convert = |opt: Option<PyReadonlyArray1<f64>>| -> Option<leto::Array1<f64>> {
            opt.map(|sig| sig.as_array().to_owned().into())
        };
        let mask_f64 = nd_to_leto3(mask_arr.mapv(|b| if b { 1.0 } else { 0.0 }));
        let amplitude = [&ux, &uy, &uz]
            .iter()
            .filter_map(|sig| {
                sig.as_ref()
                    .map(|s| s.as_array().iter().fold(0.0_f64, |a, &v| a.max(v.abs())))
            })
            .fold(0.0_f64, f64::max);
        Ok(Source {
            source_type: "elastic_velocity_source".to_string(),
            frequency: 0.0,
            amplitude,
            position: None,
            mask: Some(mask_f64),
            signal: None,
            source_mode: normalised_mode.to_string(),
            initial_pressure: None,
            velocity_signal: None,
            direction: None,
            kwave_array: None,
            elastic_ux_signal_1d: convert(ux),
            elastic_uy_signal_1d: convert(uy),
            elastic_uz_signal_1d: convert(uz),
        })
    }

    /// Create an initial-displacement source for the **elastic** solver.
    ///
    /// Sets the initial value of one displacement component (`ux`, `uy`, or
    /// `uz`) on the elastic wavefield while the other two components and
    /// all three velocity components are initialised to zero.
    #[staticmethod]
    #[pyo3(signature = (field, axis="z"))]
    fn from_initial_displacement(field: &Bound<'_, PyAny>, axis: &str) -> PyResult<Self> {
        let field_arr: Array3<f64> = if let Ok(f3) = field.extract::<PyReadonlyArray3<f64>>() {
            nd_to_leto3(f3.as_array().to_owned())
        } else if let Ok(f2) = field.extract::<PyReadonlyArray2<f64>>() {
            nd_to_leto3(f2.as_array().insert_axis(Axis(2)).to_owned())
        } else {
            return Err(PyValueError::new_err(
                "Initial displacement must be a 2D or 3D ndarray of float64 values",
            ));
        };
        if field_arr.iter().all(|&v| v == 0.0) {
            return Err(PyValueError::new_err("Initial displacement is all zeros"));
        }
        let axis_norm = match axis {
            "x" | "X" => "x",
            "y" | "Y" => "y",
            "z" | "Z" => "z",
            other => {
                return Err(PyValueError::new_err(format!(
                    "axis must be 'x', 'y', or 'z'; got '{}'",
                    other
                )))
            }
        };
        let amplitude = field_arr.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let source_type = format!("elastic_u0_{}", axis_norm);
        Ok(Source {
            source_type,
            frequency: 0.0,
            amplitude,
            position: None,
            mask: None,
            signal: None,
            source_mode: "additive".to_string(),
            initial_pressure: Some(field_arr),
            velocity_signal: None,
            direction: None,
            kwave_array: None,
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
        })
    }
}

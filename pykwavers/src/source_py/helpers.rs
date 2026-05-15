use ndarray::{Array2, Axis};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use kwavers::domain::source::array_2d::ApodizationType as KwaversApodizationType;

/// Convert a Python 1D or 2D signal array to a 2D signal matrix (rows = sources).
pub(crate) fn pressure_signal_to_matrix(signal: &Bound<'_, PyAny>) -> PyResult<Array2<f64>> {
    if let Ok(signal_1d) = signal.extract::<PyReadonlyArray1<f64>>() {
        let signal_arr = signal_1d.as_array();
        if signal_arr.is_empty() {
            return Err(PyValueError::new_err("Signal must not be empty"));
        }
        return Ok(signal_arr.insert_axis(Axis(0)).to_owned());
    }

    if let Ok(signal_2d) = signal.extract::<PyReadonlyArray2<f64>>() {
        let signal_arr = signal_2d.as_array().to_owned();
        if signal_arr.is_empty() {
            return Err(PyValueError::new_err("Signal must not be empty"));
        }
        return Ok(signal_arr);
    }

    Err(PyValueError::new_err(
        "Signal must be a 1D or 2D ndarray of float64 values",
    ))
}

/// Convert Python apodization string to kwavers type
pub(crate) fn parse_apodization_type(apodization: &str) -> PyResult<KwaversApodizationType> {
    match apodization {
        "Rectangular" => Ok(KwaversApodizationType::Rectangular),
        "Hanning" => Ok(KwaversApodizationType::Hanning),
        "Hamming" => Ok(KwaversApodizationType::Hamming),
        "Blackman" => Ok(KwaversApodizationType::Blackman),
        _ => Err(PyValueError::new_err(
            "Apodization must be one of: Rectangular, Hanning, Hamming, Blackman",
        )),
    }
}

/// Convert kwavers apodization type to Python string
pub(crate) fn apodization_to_string(apodization: &KwaversApodizationType) -> String {
    match apodization {
        KwaversApodizationType::Rectangular => "Rectangular".to_string(),
        KwaversApodizationType::Hanning => "Hanning".to_string(),
        KwaversApodizationType::Hamming => "Hamming".to_string(),
        KwaversApodizationType::Blackman => "Blackman".to_string(),
        KwaversApodizationType::Gaussian { sigma } => format!("Gaussian(sigma={})", sigma),
    }
}

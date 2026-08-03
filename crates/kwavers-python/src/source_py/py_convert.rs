use leto::Array2;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};

use crate::array_utils::{pyarray1_to_leto1, pyarray2_to_leto2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use kwavers_transducer::array_2d::ApodizationType as KwaversApodizationType;

/// Convert a Python 1D or 2D signal array to a 2D signal matrix (rows = sources).
pub(crate) fn pressure_signal_to_matrix(signal: &Bound<'_, PyAny>) -> PyResult<Array2<f64>> {
    if let Ok(signal_1d) = signal.extract::<PyReadonlyArray1<f64>>() {
        let signal_arr = pyarray1_to_leto1(&signal_1d)?;
        if signal_arr.is_empty() {
            return Err(PyValueError::new_err("Signal must not be empty"));
        }
        let length = signal_arr.shape()[0];
        return Ok(Array2::from_shape_fn([1, length], |[_, index]| {
            signal_arr[index]
        }));
    }

    if let Ok(signal_2d) = signal.extract::<PyReadonlyArray2<f64>>() {
        let signal_arr = pyarray2_to_leto2(&signal_2d)?;
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
        "Uniform" | "Rectangular" => Ok(KwaversApodizationType::Uniform),
        "Hanning" => Ok(KwaversApodizationType::Hanning),
        "Hamming" => Ok(KwaversApodizationType::Hamming),
        "Blackman" => Ok(KwaversApodizationType::Blackman),
        _ => Err(PyValueError::new_err(
            "Apodization must be one of: Uniform, Hanning, Hamming, Blackman",
        )),
    }
}

/// Convert kwavers apodization type to Python string
pub(crate) fn apodization_to_string(apodization: &KwaversApodizationType) -> String {
    match apodization {
        KwaversApodizationType::Uniform => "Uniform".to_string(),
        KwaversApodizationType::Hanning => "Hanning".to_string(),
        KwaversApodizationType::Hamming => "Hamming".to_string(),
        KwaversApodizationType::Blackman => "Blackman".to_string(),
        KwaversApodizationType::Tukey { r } => format!("Tukey(r={})", r),
        KwaversApodizationType::Gaussian { sigma } => format!("Gaussian(sigma={})", sigma),
        KwaversApodizationType::Kaiser { beta } => format!("Kaiser(beta={})", beta),
    }
}

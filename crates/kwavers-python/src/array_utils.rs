//! PyArray-only helpers for kwavers-python.
//!
//! This module centralises the small set of conversions needed between Python
//! NumPy arrays and the internal Leto / Rust `Vec` representation without
//! exposing the backing array-provider type.

use numpy::{
    Element, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyRuntimeError, Py, PyResult, Python};

fn shape_to_array<const N: usize>(shape: &[usize]) -> [usize; N] {
    shape
        .try_into()
        .expect("shape length matches dimensionality")
}

/// Copy a 1-D readonly NumPy array into a Rust `Vec`.
///
/// Contiguous inputs use a zero-copy view; non-contiguous inputs are copied to a
/// temporary contiguous buffer first.
pub fn copy_pyarray1_to_vec<'py, T>(array: &PyReadonlyArray1<'py, T>) -> PyResult<Vec<T>>
where
    T: Element + Copy,
{
    if let Ok(slice) = array.as_slice() {
        return Ok(slice.to_vec());
    }
    let copy = array.cast_array::<T>(false).map_err(|e| {
        PyRuntimeError::new_err(format!("failed to make 1-D array contiguous: {e}"))
    })?;
    copy.to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to read 1-D contiguous copy: {e}")))
}

/// Copy a 2-D readonly NumPy array into a flat `Vec` and return its shape.
pub fn copy_pyarray2_to_vec<'py, T>(
    array: &PyReadonlyArray2<'py, T>,
) -> PyResult<(Vec<T>, [usize; 2])>
where
    T: Element + Copy,
{
    let shape = shape_to_array(array.shape());
    if let Ok(slice) = array.as_slice() {
        return Ok((slice.to_vec(), shape));
    }
    let copy = array.cast_array::<T>(false).map_err(|e| {
        PyRuntimeError::new_err(format!("failed to make 2-D array contiguous: {e}"))
    })?;
    let data = copy
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to read 2-D contiguous copy: {e}")))?;
    Ok((data, shape))
}

/// Copy a 3-D readonly NumPy array into a flat `Vec` and return its shape.
pub fn copy_pyarray3_to_vec<'py, T>(
    array: &PyReadonlyArray3<'py, T>,
) -> PyResult<(Vec<T>, [usize; 3])>
where
    T: Element + Copy,
{
    let shape = shape_to_array(array.shape());
    if let Ok(slice) = array.as_slice() {
        return Ok((slice.to_vec(), shape));
    }
    let copy = array.cast_array::<T>(false).map_err(|e| {
        PyRuntimeError::new_err(format!("failed to make 3-D array contiguous: {e}"))
    })?;
    let data = copy
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to read 3-D contiguous copy: {e}")))?;
    Ok((data, shape))
}

/// Convert a 1-D readonly NumPy array into a leto 1-D array.
pub fn pyarray1_to_leto1<'py, T>(array: &PyReadonlyArray1<'py, T>) -> PyResult<leto::Array1<T>>
where
    T: Element + Copy + Clone,
{
    let data = copy_pyarray1_to_vec(array)?;
    let shape = array.shape();
    Ok(leto::Array1::from_shape_vec(shape[0], data).expect("data length matches 1-D shape"))
}

/// Convert a 2-D readonly NumPy array into a leto 2-D array.
#[allow(dead_code)]
pub fn pyarray2_to_leto2<'py, T>(array: &PyReadonlyArray2<'py, T>) -> PyResult<leto::Array2<T>>
where
    T: Element + Copy + Clone,
{
    let (data, shape) = copy_pyarray2_to_vec(array)?;
    Ok(leto::Array2::from_shape_vec(shape, data).expect("data length matches 2-D shape"))
}

/// Convert a 3-D readonly NumPy array into a leto 3-D array.
pub fn pyarray3_to_leto3<'py, T>(array: &PyReadonlyArray3<'py, T>) -> PyResult<leto::Array3<T>>
where
    T: Element + Copy + Clone,
{
    let (data, shape) = copy_pyarray3_to_vec(array)?;
    Ok(leto::Array3::from_shape_vec(shape, data).expect("data length matches 3-D shape"))
}

/// Convert a leto 1-D array into a Python 1-D NumPy array.
pub fn leto1_to_pyarray1<'py, T>(py: Python<'py>, arr: leto::Array1<T>) -> PyResult<Py<PyArray1<T>>>
where
    T: Element + Copy,
{
    let data = arr.into_vec();
    Ok(PyArray1::from_vec(py, data).unbind())
}

/// Convert a leto 2-D array into a Python 2-D NumPy array.
pub fn leto2_to_pyarray2<'py, T>(py: Python<'py>, arr: leto::Array2<T>) -> PyResult<Py<PyArray2<T>>>
where
    T: Element + Copy,
{
    let shape = arr.shape();
    let data = arr.into_vec();
    let arr1 = PyArray1::<T>::from_vec(py, data);
    Ok(arr1.reshape(shape)?.unbind())
}

/// Convert a leto 3-D array into a Python 3-D NumPy array.
pub fn leto3_to_pyarray3<'py, T>(py: Python<'py>, arr: leto::Array3<T>) -> PyResult<Py<PyArray3<T>>>
where
    T: Element + Copy,
{
    let shape = arr.shape();
    let data = arr.into_vec();
    let arr1 = PyArray1::<T>::from_vec(py, data);
    Ok(arr1.reshape(shape)?.unbind())
}

/// Create a 1-D NumPy array from a `Vec`.
pub fn vec_to_pyarray1<'py, T>(py: Python<'py>, data: Vec<T>) -> Py<PyArray1<T>>
where
    T: Element + Copy,
{
    PyArray1::from_vec(py, data).unbind()
}

/// Create a 2-D NumPy array from a flat `Vec` and shape.
pub fn vec_to_pyarray2<'py, T>(
    py: Python<'py>,
    shape: [usize; 2],
    data: Vec<T>,
) -> PyResult<Py<PyArray2<T>>>
where
    T: Element + Copy,
{
    let arr1 = PyArray1::<T>::from_vec(py, data);
    Ok(arr1.reshape(shape)?.unbind())
}

/// Create a 3-D NumPy array from a flat `Vec` and shape.
#[allow(dead_code)]
pub fn vec_to_pyarray3<'py, T>(
    py: Python<'py>,
    shape: [usize; 3],
    data: Vec<T>,
) -> PyResult<Py<PyArray3<T>>>
where
    T: Element + Copy,
{
    let arr1 = PyArray1::<T>::from_vec(py, data);
    Ok(arr1.reshape(shape)?.unbind())
}

/// Build a linearly spaced 1-D `Vec`.
pub fn linspace_vec(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + i as f64 * step).collect()
}

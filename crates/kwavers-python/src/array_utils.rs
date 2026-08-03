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

fn shape_to_array<const N: usize>(shape: &[usize]) -> PyResult<[usize; N]> {
    shape.try_into().map_err(|_| {
        PyRuntimeError::new_err(format!(
            "expected a {N}-D NumPy shape, received {} dimensions",
            shape.len()
        ))
    })
}

/// Copy a 1-D readonly NumPy array into a Rust `Vec`.
///
/// Contiguous inputs use the fast contiguous read path; non-contiguous inputs
/// are copied to a temporary contiguous buffer first. Both paths then copy into
/// Leto-owned storage.
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
    let shape = shape_to_array(array.shape())?;
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
    let shape = shape_to_array(array.shape())?;
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
    leto::Array1::from_shape_vec(shape[0], data).map_err(|error| {
        PyRuntimeError::new_err(format!("failed to construct Leto 1-D array: {error}"))
    })
}

/// Convert a 2-D readonly NumPy array into a leto 2-D array.
pub fn pyarray2_to_leto2<'py, T>(array: &PyReadonlyArray2<'py, T>) -> PyResult<leto::Array2<T>>
where
    T: Element + Copy + Clone,
{
    let (data, shape) = copy_pyarray2_to_vec(array)?;
    leto::Array2::from_shape_vec(shape, data).map_err(|error| {
        PyRuntimeError::new_err(format!("failed to construct Leto 2-D array: {error}"))
    })
}

/// Convert a 3-D readonly NumPy array into a leto 3-D array.
pub fn pyarray3_to_leto3<'py, T>(array: &PyReadonlyArray3<'py, T>) -> PyResult<leto::Array3<T>>
where
    T: Element + Copy + Clone,
{
    let (data, shape) = copy_pyarray3_to_vec(array)?;
    leto::Array3::from_shape_vec(shape, data).map_err(|error| {
        PyRuntimeError::new_err(format!("failed to construct Leto 3-D array: {error}"))
    })
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

#[cfg(test)]
mod tests {
    use super::{
        leto1_to_pyarray1, leto2_to_pyarray2, leto3_to_pyarray3, linspace_vec, pyarray1_to_leto1,
        pyarray2_to_leto2, pyarray3_to_leto3,
    };
    use numpy::{
        PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
        PyReadonlyArray3, PyUntypedArrayMethods,
    };
    use pyo3::types::PyAnyMethods;
    use pyo3::Python;
    use std::ffi::CString;

    #[test]
    fn pyarray_round_trip_preserves_rank_shape_and_values() {
        Python::initialize();
        Python::attach(|py| {
            let one = leto::Array1::from_shape_vec(3, vec![1.0_f64, 2.0, 3.0]).unwrap();
            let one_out = leto1_to_pyarray1(py, one).unwrap();
            assert_eq!(one_out.bind(py).shape(), [3]);
            assert_eq!(
                one_out.bind(py).readonly().as_slice().unwrap(),
                [1.0_f64, 2.0, 3.0]
            );

            let two = leto::Array2::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
            let two_out = leto2_to_pyarray2(py, two).unwrap();
            assert_eq!(two_out.bind(py).shape(), [2, 2]);
            assert_eq!(
                two_out.bind(py).readonly().as_slice().unwrap(),
                [1.0_f64, 2.0, 3.0, 4.0]
            );

            let three =
                leto::Array3::from_shape_vec([1, 2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
            let three_out = leto3_to_pyarray3(py, three).unwrap();
            assert_eq!(three_out.bind(py).shape(), [1, 2, 2]);
            assert_eq!(
                three_out.bind(py).readonly().as_slice().unwrap(),
                [1.0_f64, 2.0, 3.0, 4.0]
            );
        });
    }

    #[test]
    fn numpy_inputs_convert_to_leto_without_shape_changes() {
        Python::initialize();
        Python::attach(|py| {
            let one_input = PyArray1::from_vec(py, vec![1.0_f64, 2.0, 3.0]);
            let one = pyarray1_to_leto1(&one_input.readonly()).unwrap();
            assert_eq!(one.shape(), [3]);
            assert_eq!(one.into_vec(), [1.0_f64, 2.0, 3.0]);

            let two_input = PyArray2::from_vec2(py, &[vec![1.0_f64, 2.0], vec![3.0, 4.0]]).unwrap();
            let two = pyarray2_to_leto2(&two_input.readonly()).unwrap();
            assert_eq!(two.shape(), [2, 2]);
            assert_eq!(two.into_vec(), [1.0_f64, 2.0, 3.0, 4.0]);

            let three_input = PyArray3::from_vec3(
                py,
                &[
                    vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]],
                    vec![vec![5.0_f64, 6.0], vec![7.0, 8.0]],
                ],
            )
            .unwrap();
            let three = pyarray3_to_leto3(&three_input.readonly()).unwrap();
            assert_eq!(three.shape(), [2, 2, 2]);
            assert_eq!(
                three.into_vec(),
                [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            );
        });
    }

    #[test]
    fn non_contiguous_numpy_inputs_are_copied_in_c_order() {
        Python::initialize();
        Python::attach(|py| {
            let one_code = CString::new("__import__('numpy').arange(8.0)[::2]").unwrap();
            let one_input = py
                .eval(one_code.as_c_str(), None, None)
                .unwrap()
                .extract::<PyReadonlyArray1<f64>>()
                .unwrap();
            let one = pyarray1_to_leto1(&one_input).unwrap();
            assert_eq!(one.shape(), [4]);
            assert_eq!(one.into_vec(), [0.0, 2.0, 4.0, 6.0]);

            let two_code =
                CString::new("__import__('numpy').arange(12.0).reshape((3, 4))[:, ::2]").unwrap();
            let two_input = py
                .eval(two_code.as_c_str(), None, None)
                .unwrap()
                .extract::<PyReadonlyArray2<f64>>()
                .unwrap();
            let two = pyarray2_to_leto2(&two_input).unwrap();
            assert_eq!(two.shape(), [3, 2]);
            assert_eq!(two.into_vec(), [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);

            let three_code =
                CString::new("__import__('numpy').arange(24.0).reshape((2, 3, 4))[:, :, ::2]")
                    .unwrap();
            let three_input = py
                .eval(three_code.as_c_str(), None, None)
                .unwrap()
                .extract::<PyReadonlyArray3<f64>>()
                .unwrap();
            let three = pyarray3_to_leto3(&three_input).unwrap();
            assert_eq!(three.shape(), [2, 3, 2]);
            assert_eq!(
                three.into_vec(),
                [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0,]
            );
        });
    }

    #[test]
    fn linspace_handles_empty_singleton_and_endpoints() {
        assert!(linspace_vec(0.0_f64, 1.0, 0).is_empty());
        assert_eq!(linspace_vec(2.0_f64, 5.0, 1), [2.0_f64]);
        assert_eq!(linspace_vec(0.0_f64, 1.0, 3), [0.0_f64, 0.5, 1.0]);
    }
}

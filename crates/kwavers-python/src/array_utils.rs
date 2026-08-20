//! PyArray-only helpers for kwavers-python.
//!
//! This module centralises the small set of conversions needed between Python
//! NumPy arrays and the internal Leto / Rust `Vec` representation without
//! exposing the backing array-provider type.

use numpy::ndarray::{ArrayView, Dimension};
use numpy::{
    Element, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::{Py, PyResult, Python};

fn copy_array_view_to_vec<T, D>(array: ArrayView<'_, T, D>) -> Vec<T>
where
    T: Copy,
    D: Dimension,
{
    array
        .as_slice()
        .map_or_else(|| array.iter().copied().collect(), <[T]>::to_vec)
}

fn shape_to_array<const N: usize>(shape: &[usize]) -> [usize; N] {
    shape
        .try_into()
        .expect("shape length matches dimensionality")
}

/// Copy a 1-D readonly NumPy array into a Rust `Vec`.
///
/// Contiguous inputs copy once from their borrowed slice. Strided inputs copy
/// once in logical index order without allocating a temporary NumPy array.
///
/// # Migration note
///
/// Previously kwavers-python exposed the legacy 1-D array boundary at this
/// seam. The backing store is now `leto::Array1<T>`; this helper bridges the
/// Python/NumPy FFI surface to the Atlas host-array layer.
#[doc(alias = "ndarray")]
#[doc(alias = "Array1")]
pub fn copy_pyarray1_to_vec<'py, T>(array: &PyReadonlyArray1<'py, T>) -> PyResult<Vec<T>>
where
    T: Element + Copy,
{
    Ok(copy_array_view_to_vec(array.as_array()))
}

/// Copy a 2-D readonly NumPy array into a flat `Vec` and return its shape.
///
/// # Migration note
///
/// Replaces the legacy 2-D array type at this seam; the Atlas consumer-side
/// representation is `leto::Array2<T>` (row-major `VecStorage`).
#[doc(alias = "ndarray")]
#[doc(alias = "Array2")]
pub fn copy_pyarray2_to_vec<'py, T>(
    array: &PyReadonlyArray2<'py, T>,
) -> PyResult<(Vec<T>, [usize; 2])>
where
    T: Element + Copy,
{
    let shape = shape_to_array(array.shape());
    Ok((copy_array_view_to_vec(array.as_array()), shape))
}

/// Copy a 3-D readonly NumPy array into a flat `Vec` and return its shape.
///
/// # Migration note
///
/// Replaces the legacy 3-D array type at this seam; the Atlas consumer-side
/// representation is `leto::Array3<T>` (row-major `VecStorage`).
#[doc(alias = "ndarray")]
#[doc(alias = "Array3")]
pub fn copy_pyarray3_to_vec<'py, T>(
    array: &PyReadonlyArray3<'py, T>,
) -> PyResult<(Vec<T>, [usize; 3])>
where
    T: Element + Copy,
{
    let shape = shape_to_array(array.shape());
    Ok((copy_array_view_to_vec(array.as_array()), shape))
}

/// Convert a 1-D readonly NumPy array into a leto 1-D array.
///
/// This is the primary ndarray-replacement boundary: callers that previously
/// accepted the legacy 1-D array type now accept `PyReadonlyArray1<'py, T>`
/// at the FFI boundary and convert to `leto::Array1<T>` for interior use.
#[doc(alias = "ndarray")]
#[doc(alias = "Array1")]
#[doc(alias = "from_numpy")]
pub fn pyarray1_to_leto1<'py, T>(array: &PyReadonlyArray1<'py, T>) -> PyResult<leto::Array1<T>>
where
    T: Element + Copy + Clone,
{
    let data = copy_pyarray1_to_vec(array)?;
    let shape = array.shape();
    Ok(leto::Array1::from_shape_vec(shape[0], data).expect("data length matches 1-D shape"))
}

/// Convert a 2-D readonly NumPy array into a leto 2-D array.
///
/// Replaces the legacy 2-D array type at the FFI boundary.
#[doc(alias = "ndarray")]
#[doc(alias = "Array2")]
#[doc(alias = "from_numpy")]
pub fn pyarray2_to_leto2<'py, T>(array: &PyReadonlyArray2<'py, T>) -> PyResult<leto::Array2<T>>
where
    T: Element + Copy + Clone,
{
    let (data, shape) = copy_pyarray2_to_vec(array)?;
    Ok(leto::Array2::from_shape_vec(shape, data).expect("data length matches 2-D shape"))
}

/// Convert a 3-D readonly NumPy array into a leto 3-D array.
///
/// Replaces the legacy 3-D array type at the FFI boundary.
#[doc(alias = "ndarray")]
#[doc(alias = "Array3")]
#[doc(alias = "from_numpy")]
pub fn pyarray3_to_leto3<'py, T>(array: &PyReadonlyArray3<'py, T>) -> PyResult<leto::Array3<T>>
where
    T: Element + Copy + Clone,
{
    let (data, shape) = copy_pyarray3_to_vec(array)?;
    Ok(leto::Array3::from_shape_vec(shape, data).expect("data length matches 3-D shape"))
}

/// Convert a leto 1-D array into a Python 1-D NumPy array.
///
/// Replaces the legacy 1-D array → Python conversion at the FFI boundary.
/// The Atlas host-array type is `leto::Array1<T>`.
#[doc(alias = "ndarray")]
#[doc(alias = "to_numpy")]
#[doc(alias = "Array1")]
pub fn leto1_to_pyarray1<'py, T>(py: Python<'py>, arr: leto::Array1<T>) -> PyResult<Py<PyArray1<T>>>
where
    T: Element + Copy,
{
    let data = arr.into_vec();
    Ok(PyArray1::from_vec(py, data).unbind())
}

/// Convert a leto 2-D array into a Python 2-D NumPy array.
///
/// Replaces the legacy 2-D array → Python conversion.
#[doc(alias = "ndarray")]
#[doc(alias = "to_numpy")]
#[doc(alias = "Array2")]
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
///
/// Replaces the legacy 3-D array → Python conversion.
#[doc(alias = "ndarray")]
#[doc(alias = "to_numpy")]
#[doc(alias = "Array3")]
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
    use numpy::{
        get_array_module, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
        PyReadonlyArray3,
    };
    use pyo3::{
        ffi::c_str,
        types::{IntoPyDict, PyAnyMethods},
    };

    use super::*;

    #[test]
    fn strided_numpy_inputs_copy_once_in_logical_order() -> PyResult<()> {
        Python::attach(|py| {
            let locals = [("np", get_array_module(py)?)].into_py_dict(py)?;

            let one = py
                .eval(
                    c_str!("np.arange(8.0, dtype='float64')[::2]"),
                    None,
                    Some(&locals),
                )?
                .extract::<PyReadonlyArray1<'_, f64>>()?;
            assert_eq!(copy_pyarray1_to_vec(&one)?, [0.0, 2.0, 4.0, 6.0]);

            let two = py
                .eval(
                    c_str!("np.arange(12.0, dtype='float64').reshape(3, 4)[:, ::2]"),
                    None,
                    Some(&locals),
                )?
                .extract::<PyReadonlyArray2<'_, f64>>()?;
            assert_eq!(
                copy_pyarray2_to_vec(&two)?,
                (vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0], [3, 2])
            );

            let three = py
                .eval(
                    c_str!("np.arange(24.0, dtype='float64').reshape(2, 3, 4)[:, ::2, ::-1]"),
                    None,
                    Some(&locals),
                )?
                .extract::<PyReadonlyArray3<'_, f64>>()?;
            assert_eq!(
                copy_pyarray3_to_vec(&three)?,
                (
                    vec![
                        3.0, 2.0, 1.0, 0.0, 11.0, 10.0, 9.0, 8.0, 15.0, 14.0, 13.0, 12.0, 23.0,
                        22.0, 21.0, 20.0,
                    ],
                    [2, 2, 4],
                )
            );

            Ok(())
        })
    }

    #[test]
    fn leto_round_trips_preserve_values_and_shapes() -> PyResult<()> {
        Python::attach(|py| {
            let one = PyArray1::from_vec(py, vec![1_i32, -2, 3]);
            let one_leto = pyarray1_to_leto1(&one.readonly())?;
            let one_result = leto1_to_pyarray1(py, one_leto)?;
            assert_eq!(one_result.bind(py).shape(), [3]);
            assert_eq!(one_result.bind(py).readonly().as_slice()?, [1, -2, 3]);

            let two = PyArray1::from_vec(py, vec![0.5, 1.5, 2.5, 3.5]).reshape([2, 2])?;
            let two_leto = pyarray2_to_leto2(&two.readonly())?;
            let two_result = leto2_to_pyarray2(py, two_leto)?;
            assert_eq!(two_result.bind(py).shape(), [2, 2]);
            assert_eq!(
                two_result.bind(py).readonly().as_slice()?,
                [0.5, 1.5, 2.5, 3.5]
            );

            let three = PyArray1::from_vec(py, (0_i16..8).collect()).reshape([2, 2, 2])?;
            let three_leto = pyarray3_to_leto3(&three.readonly())?;
            let three_result = leto3_to_pyarray3(py, three_leto)?;
            assert_eq!(three_result.bind(py).shape(), [2, 2, 2]);
            assert_eq!(
                three_result.bind(py).readonly().as_slice()?,
                [0, 1, 2, 3, 4, 5, 6, 7]
            );

            Ok(())
        })
    }

    #[test]
    fn vector_outputs_preserve_values_and_shapes() -> PyResult<()> {
        Python::attach(|py| {
            let one = vec_to_pyarray1(py, vec![true, false, true]);
            assert_eq!(one.bind(py).shape(), [3]);
            assert_eq!(one.bind(py).readonly().as_slice()?, [true, false, true]);

            let two = vec_to_pyarray2(py, [2, 3], (0_u32..6).collect())?;
            assert_eq!(two.bind(py).shape(), [2, 3]);
            assert_eq!(two.bind(py).readonly().as_slice()?, [0, 1, 2, 3, 4, 5]);

            let three = vec_to_pyarray3(py, [1, 2, 2], vec![-1_i64, 2, -3, 4])?;
            assert_eq!(three.bind(py).shape(), [1, 2, 2]);
            assert_eq!(three.bind(py).readonly().as_slice()?, [-1, 2, -3, 4]);

            Ok(())
        })
    }

    #[test]
    fn linspace_handles_empty_singleton_and_regular_ranges() {
        assert!(linspace_vec(2.0, 5.0, 0).is_empty());
        assert_eq!(linspace_vec(2.0, 5.0, 1), [2.0]);
        assert_eq!(linspace_vec(-1.0, 1.0, 5), [-1.0, -0.5, 0.0, 0.5, 1.0]);
    }
}

//! Conversion utilities between tensor types.

use ndarray::ArrayD;

use super::ndarray_tensor::NdArrayTensor;
use super::traits::TensorView;

/// Convert ndarray to tensor.
#[must_use]
pub fn from_ndarray(arr: ArrayD<f64>) -> NdArrayTensor {
    NdArrayTensor::from_array(arr)
}

/// Convert tensor to ndarray.
pub fn to_ndarray(tensor: &dyn TensorView) -> ArrayD<f64> {
    tensor.to_ndarray()
}

/// Clone tensor as ndarray-backed tensor.
pub fn to_ndarray_tensor(tensor: &dyn TensorView) -> NdArrayTensor {
    NdArrayTensor::from_array(tensor.to_ndarray())
}

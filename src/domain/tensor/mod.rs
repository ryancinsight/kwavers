//! Unified Tensor Abstraction Layer
//!
//! This module provides a unified tensor interface that abstracts over different
//! backend implementations (ndarray for CPU, Burn for GPU/autodiff).
//!
//! # Design Rationale
//!
//! The kwavers library needs to support two distinct computational patterns:
//!
//! 1. **Forward solvers** (numerical integration): Use ndarray for efficient CPU
//!    computation without autodiff overhead. These solvers discretize PDEs using
//!    finite differences, finite elements, or spectral methods.
//!
//! 2. **Inverse solvers** (PINNs, optimization): Use Burn for automatic differentiation
//!    and GPU acceleration. These solvers train neural networks to approximate PDE
//!    solutions or perform gradient-based optimization.
//!
//! # Zero-Copy Interoperability
//!
//! When both ndarray and Burn are enabled, tensors can be converted with minimal
//! overhead using Burn's NdArray backend:
//!
//! ```text
//! ndarray::ArrayD ←→ Burn::Tensor (NdArray backend) [zero-copy when possible]
//! ```
//!
//! # Feature Gates
//!
//! - Default: ndarray backend only (minimal dependencies)
//! - `burn-ndarray`: Enable Burn with NdArray backend (autodiff, CPU)
//! - `burn-wgpu`: Enable Burn with WGPU backend (autodiff, GPU)
//! - `burn-cuda`: Enable Burn with CUDA backend (autodiff, GPU)
//!
//! # Architecture
//!
//! ```text
//! Domain Layer (this module)
//!     ↓
//! Solver Layer
//!     ├─ Forward Solvers  → use TensorView (read-only ndarray)
//!     └─ Inverse Solvers  → use DifferentiableTensor (Burn)
//! ```

use ndarray::{Array, ArrayD, Dimension, IxDyn};
use std::fmt;

/// Tensor shape specification
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create shape from dimensions
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Get size of specific dimension
    pub fn dim(&self, axis: usize) -> usize {
        self.dims[axis]
    }

    /// Get total number of elements
    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    /// Get dimensions as slice
    pub fn as_slice(&self) -> &[usize] {
        &self.dims
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }
}

/// Data type specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
}

/// Tensor backend specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CPU-only ndarray backend (no autodiff)
    NdArray,
    /// Burn with NdArray backend (autodiff on CPU)
    BurnNdArray,
    /// Burn with WGPU backend (autodiff on GPU)
    #[cfg(feature = "burn-wgpu")]
    BurnWgpu,
    /// Burn with CUDA backend (autodiff on GPU)
    #[cfg(feature = "burn-cuda")]
    BurnCuda,
}

/// Read-only tensor view abstraction
///
/// Provides a unified interface for accessing tensor data without
/// requiring autodiff or GPU capabilities. Suitable for forward solvers.
pub trait TensorView: Send + Sync {
    /// Get tensor shape
    fn shape(&self) -> Shape;

    /// Get data type
    fn dtype(&self) -> DType;

    /// Get backend
    fn backend(&self) -> Backend;

    /// Convert to ndarray (may involve copy if on GPU)
    fn to_ndarray_f64(&self) -> ArrayD<f64>;

    /// Get scalar value (for 0-dimensional tensors)
    fn to_scalar_f64(&self) -> Result<f64, String> {
        if self.shape().ndim() != 0 {
            return Err(format!(
                "Cannot convert tensor of shape {:?} to scalar",
                self.shape()
            ));
        }
        let arr = self.to_ndarray_f64();
        Ok(arr[[]])
    }

    /// Clone tensor data
    fn clone_data(&self) -> Box<dyn TensorView>;
}

/// Mutable tensor abstraction for forward solvers
///
/// Extends TensorView with mutation capabilities while maintaining
/// the constraint that autodiff is not required.
pub trait TensorMut: TensorView {
    /// Update tensor data from ndarray
    fn update_from_ndarray(&mut self, data: &ArrayD<f64>);

    /// Apply element-wise function
    fn map_inplace(&mut self, f: impl Fn(f64) -> f64);

    /// Fill with constant value
    fn fill(&mut self, value: f64);

    /// Scale by constant
    fn scale(&mut self, factor: f64) {
        self.map_inplace(|x| x * factor);
    }

    /// Add constant
    fn add_constant(&mut self, value: f64) {
        self.map_inplace(|x| x + value);
    }
}

/// Simple ndarray-backed tensor (default, no autodiff)
#[derive(Clone)]
pub struct NdArrayTensor {
    data: ArrayD<f64>,
}

impl NdArrayTensor {
    /// Create from ndarray
    pub fn from_array(data: ArrayD<f64>) -> Self {
        Self { data }
    }

    /// Create with shape filled with zeros
    pub fn zeros(shape: Shape) -> Self {
        let data = ArrayD::zeros(IxDyn(shape.as_slice()));
        Self { data }
    }

    /// Create with shape filled with ones
    pub fn ones(shape: Shape) -> Self {
        let data = ArrayD::ones(IxDyn(shape.as_slice()));
        Self { data }
    }

    /// Create from scalar
    pub fn scalar(value: f64) -> Self {
        let data = ArrayD::from_elem(IxDyn(&[]), value);
        Self { data }
    }

    /// Get mutable reference to underlying array
    pub fn as_array_mut(&mut self) -> &mut ArrayD<f64> {
        &mut self.data
    }

    /// Get reference to underlying array
    pub fn as_array(&self) -> &ArrayD<f64> {
        &self.data
    }

    /// Convert to owned array
    pub fn into_array(self) -> ArrayD<f64> {
        self.data
    }
}

impl TensorView for NdArrayTensor {
    fn shape(&self) -> Shape {
        Shape::new(self.data.shape().to_vec())
    }

    fn dtype(&self) -> DType {
        DType::F64
    }

    fn backend(&self) -> Backend {
        Backend::NdArray
    }

    fn to_ndarray_f64(&self) -> ArrayD<f64> {
        self.data.clone()
    }

    fn clone_data(&self) -> Box<dyn TensorView> {
        Box::new(self.clone())
    }
}

impl TensorMut for NdArrayTensor {
    fn update_from_ndarray(&mut self, data: &ArrayD<f64>) {
        assert_eq!(
            self.data.shape(),
            data.shape(),
            "Shape mismatch in update_from_ndarray"
        );
        self.data.assign(data);
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
        self.data.mapv_inplace(f);
    }

    fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }
}

impl fmt::Debug for NdArrayTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NdArrayTensor")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .finish()
    }
}

/// Conversion utilities between tensor types
pub mod convert {
    use super::*;

    /// Convert ndarray to tensor
    pub fn from_ndarray(arr: ArrayD<f64>) -> NdArrayTensor {
        NdArrayTensor::from_array(arr)
    }

    /// Convert tensor to ndarray
    pub fn to_ndarray(tensor: &dyn TensorView) -> ArrayD<f64> {
        tensor.to_ndarray_f64()
    }

    /// Clone tensor as ndarray-backed tensor
    pub fn to_ndarray_tensor(tensor: &dyn TensorView) -> NdArrayTensor {
        NdArrayTensor::from_array(tensor.to_ndarray_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_shape() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.dim(0), 2);
        assert_eq!(shape.dim(1), 3);
        assert_eq!(shape.dim(2), 4);
        assert_eq!(shape.size(), 24);
    }

    #[test]
    fn test_ndarray_tensor_creation() {
        let tensor = NdArrayTensor::zeros(Shape::new(vec![3, 4]));
        assert_eq!(tensor.shape().as_slice(), &[3, 4]);
        assert_eq!(tensor.dtype(), DType::F64);
        assert_eq!(tensor.backend(), Backend::NdArray);
    }

    #[test]
    fn test_ndarray_tensor_mutation() {
        let mut tensor = NdArrayTensor::zeros(Shape::new(vec![2, 2]));
        tensor.fill(5.0);

        let arr = tensor.to_ndarray_f64();
        assert!((arr[[0, 0]] - 5.0).abs() < 1e-10);
        assert!((arr[[1, 1]] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndarray_tensor_map() {
        let mut tensor = NdArrayTensor::ones(Shape::new(vec![3]));
        tensor.map_inplace(|x| x * 2.0);

        let arr = tensor.to_ndarray_f64();
        for val in arr.iter() {
            assert!((val - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_scalar_tensor() {
        let tensor = NdArrayTensor::scalar(42.0);
        assert_eq!(tensor.shape().ndim(), 0);
        assert_eq!(tensor.to_scalar_f64().unwrap(), 42.0);
    }

    #[test]
    fn test_conversion() {
        let arr = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let arr_dyn = arr.into_dyn();
        let tensor = convert::from_ndarray(arr_dyn.clone());

        let recovered = convert::to_ndarray(&tensor);
        assert_eq!(arr_dyn.shape(), recovered.shape());
        assert_eq!(arr_dyn[[0, 0]], recovered[[0, 0]]);
    }

    #[test]
    fn test_update_from_ndarray() {
        let mut tensor = NdArrayTensor::zeros(Shape::new(vec![2, 2]));
        let new_data = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();

        tensor.update_from_ndarray(&new_data);

        let arr = tensor.to_ndarray_f64();
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[0, 1]], 2.0);
        assert_eq!(arr[[1, 0]], 3.0);
        assert_eq!(arr[[1, 1]], 4.0);
    }

    #[test]
    #[should_panic(expected = "Cannot convert tensor of shape")]
    fn test_non_scalar_to_scalar_fails() {
        let tensor = NdArrayTensor::zeros(Shape::new(vec![2, 2]));
        let _ = tensor.to_scalar_f64().unwrap();
    }
}

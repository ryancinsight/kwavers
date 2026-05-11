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

mod ndarray_tensor;
mod traits;
mod types;
pub mod convert;

pub use ndarray_tensor::NdArrayTensor;
pub use traits::{TensorMut, TensorView};
pub use types::{Backend, DType, Shape};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

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

        let arr = tensor.to_ndarray();
        assert!((arr[[0, 0]] - 5.0).abs() < 1e-10);
        assert!((arr[[1, 1]] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndarray_tensor_map() {
        let mut tensor = NdArrayTensor::ones(Shape::new(vec![3]));
        tensor.map_inplace(|x| x * 2.0);

        let arr = tensor.to_ndarray();
        for val in arr.iter() {
            assert!((val - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_scalar_tensor() {
        let tensor = NdArrayTensor::scalar(42.0);
        assert_eq!(tensor.shape().ndim(), 0);
        assert_eq!(tensor.to_scalar().unwrap(), 42.0);
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

        let arr = tensor.to_ndarray();
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[0, 1]], 2.0);
        assert_eq!(arr[[1, 0]], 3.0);
        assert_eq!(arr[[1, 1]], 4.0);
    }

    #[test]
    #[should_panic(expected = "Cannot convert tensor of shape")]
    fn test_non_scalar_to_scalar_fails() {
        let tensor = NdArrayTensor::zeros(Shape::new(vec![2, 2]));
        let _ = tensor.to_scalar().unwrap();
    }
}

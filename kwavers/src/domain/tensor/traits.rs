//! Tensor trait abstractions: `TensorView`, `TensorMut`.

use ndarray::ArrayD;

use super::types::{TensorBackend, DType, Shape};

/// Read-only tensor view abstraction.
///
/// Provides a unified interface for accessing tensor data without
/// requiring autodiff or GPU capabilities. Suitable for forward solvers.
pub trait TensorView: Send + Sync {
    /// Get tensor shape.
    fn shape(&self) -> Shape;

    /// Get data type.
    fn dtype(&self) -> DType;

    /// Get backend.
    fn backend(&self) -> TensorBackend;

    /// Convert to ndarray (may involve copy if on GPU).
    fn to_ndarray(&self) -> ArrayD<f64>;

    /// Get scalar value (for 0-dimensional tensors).
    ///
    /// # Errors
    ///
    /// Returns an error string when the tensor is not 0-dimensional.
    fn to_scalar(&self) -> Result<f64, String> {
        if self.shape().ndim() != 0 {
            return Err(format!(
                "Cannot convert tensor of shape {:?} to scalar",
                self.shape()
            ));
        }
        let arr = self.to_ndarray();
        Ok(arr[[]])
    }

    /// Clone tensor data.
    fn clone_data(&self) -> Box<dyn TensorView>;
}

/// Mutable tensor abstraction for forward solvers.
///
/// Extends `TensorView` with mutation capabilities while maintaining
/// the constraint that autodiff is not required.
pub trait TensorMut: TensorView {
    /// Update tensor data from ndarray.
    fn update_from_ndarray(&mut self, data: &ArrayD<f64>);

    /// Apply element-wise function.
    fn map_inplace(&mut self, f: impl Fn(f64) -> f64 + Send + Sync);

    /// Fill with constant value.
    fn fill(&mut self, value: f64);

    /// Scale by constant.
    fn scale(&mut self, factor: f64) {
        self.map_inplace(|x| x * factor);
    }

    /// Add constant.
    fn add_constant(&mut self, value: f64) {
        self.map_inplace(|x| x + value);
    }
}

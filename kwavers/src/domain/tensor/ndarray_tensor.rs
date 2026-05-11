//! `NdArrayTensor`: ndarray-backed concrete tensor implementation.

use ndarray::{ArrayD, IxDyn};
use std::fmt;

use super::traits::{TensorMut, TensorView};
use super::types::{Backend, DType, Shape};

/// Simple ndarray-backed tensor (default, no autodiff).
#[derive(Clone)]
pub struct NdArrayTensor {
    data: ArrayD<f64>,
}

impl NdArrayTensor {
    /// Create from ndarray.
    #[must_use]
    pub fn from_array(data: ArrayD<f64>) -> Self {
        Self { data }
    }

    /// Create with shape filled with zeros.
    #[must_use]
    pub fn zeros(shape: Shape) -> Self {
        let data = ArrayD::zeros(IxDyn(shape.as_slice()));
        Self { data }
    }

    /// Create with shape filled with ones.
    #[must_use]
    pub fn ones(shape: Shape) -> Self {
        let data = ArrayD::ones(IxDyn(shape.as_slice()));
        Self { data }
    }

    /// Create from scalar.
    #[must_use]
    pub fn scalar(value: f64) -> Self {
        let data = ArrayD::from_elem(IxDyn(&[]), value);
        Self { data }
    }

    /// Get mutable reference to underlying array.
    pub fn as_array_mut(&mut self) -> &mut ArrayD<f64> {
        &mut self.data
    }

    /// Get reference to underlying array.
    #[must_use]
    pub fn as_array(&self) -> &ArrayD<f64> {
        &self.data
    }

    /// Convert to owned array.
    #[must_use]
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

    fn to_ndarray(&self) -> ArrayD<f64> {
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

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64 + Send + Sync) {
        self.data.par_mapv_inplace(f);
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

//! Primitive tensor types: `Shape`, `DType`, `TensorBackend`.

/// Tensor shape specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create shape from dimensions.
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    /// Get number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Get size of specific dimension.
    #[must_use]
    pub fn dim(&self, axis: usize) -> usize {
        self.dims[axis]
    }

    /// Get total number of elements.
    #[must_use]
    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    /// Get dimensions as slice.
    #[must_use]
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

/// Data type specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
}

/// Tensor backend specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorBackend {
    /// CPU-only ndarray backend (no autodiff).
    NdArray,
    /// Burn with NdArray backend (autodiff on CPU).
    BurnNdArray,
    /// Burn with WGPU backend (autodiff on GPU).
    #[cfg(feature = "burn-wgpu")]
    BurnWgpu,
    /// Burn with CUDA backend (autodiff on GPU).
    #[cfg(feature = "burn-cuda")]
    BurnCuda,
}

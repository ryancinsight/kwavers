// src/utils/mod.rs - Clean module exports without RefCell antipatterns
pub mod array_utils;
pub mod format;
pub mod iterators;
pub mod numerical;
pub mod sparse_matrix;
pub mod stencil;

// Re-export commonly used utilities
pub use self::numerical::NumericalUtils;
pub use self::sparse_matrix::CompressedSparseRowMatrix;
pub use self::stencil::{Stencil, StencilValue};
pub use crate::fft::{fft_3d_array, get_fft_for_grid, ifft_3d_array, FFT_CACHE};

#[cfg(test)]
pub mod test_helpers;

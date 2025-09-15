// src/utils/mod.rs - Clean module exports without RefCell antipatterns
pub mod array_utils;
pub mod differential_operators;
pub mod fft_cache;
pub mod fft_operations;
pub mod field_analysis;
pub mod format;
pub mod iterators;
pub mod kwave; // Modular k-Wave utilities
pub mod laplacian; // Unified Laplacian operator
pub mod linear_algebra;
pub mod sparse_matrix;
pub mod spectral;
pub mod stencil;

// Re-export commonly used utilities
pub use self::fft_operations::{fft_3d_array, ifft_3d_array};
pub use self::field_analysis::{
    calculate_beam_pattern, calculate_beam_width, calculate_directivity, calculate_field_metrics,
    calculate_intensity, calculate_mechanical_index, calculate_thermal_index, find_focal_plane,
    find_focus, find_peak_pressure, BeamPatternConfig, FarFieldMethod, FieldMetrics,
};
pub use self::sparse_matrix::CompressedSparseRowMatrix;
pub use self::stencil::{Stencil, StencilValue};

// Export differential operators with unique names to avoid conflicts
pub use self::differential_operators::{
    curl as curl_op, divergence as divergence_op, gradient as gradient_op,
    laplacian as laplacian_op, spectral_laplacian, transverse_laplacian, FDCoefficients,
    SpatialOrder,
};

// Use modern FFT cache instead of RefCell antipattern
pub use self::fft_cache::{get_fft_for_grid, FFT_CACHE};

#[cfg(test)]
pub mod test_helpers;

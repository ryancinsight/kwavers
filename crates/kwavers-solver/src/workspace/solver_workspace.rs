//! SolverWorkspace: pre-allocated FFT and real-valued scratch buffers.

use super::ScratchArena;
use kwavers_domain::grid::Grid;
use ndarray::Array3;
use num_complex::Complex;

/// Pre-allocated workspace for solver operations.
#[derive(Debug)]
pub struct SolverWorkspace {
    /// FFT workspace for complex operations
    pub fft_buffer: Array3<Complex<f64>>,
    /// Real-valued workspace for intermediate calculations
    pub real_buffer: Array3<f64>,
    /// K-space workspace for spectral operations
    pub k_space_buffer: Array3<f64>,
    /// Additional real workspace for in-place operations
    pub temp_buffer: Array3<f64>,
    /// Grid dimensions for validation
    pub(super) grid_shape: (usize, usize, usize),
}

impl SolverWorkspace {
    /// Create a new workspace for the given grid.
    pub fn new(grid: &Grid) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        Self {
            fft_buffer: Array3::zeros(shape),
            real_buffer: Array3::zeros(shape),
            k_space_buffer: Array3::zeros(shape),
            temp_buffer: Array3::zeros(shape),
            grid_shape: shape,
        }
    }

    /// Validate that the workspace matches the expected grid dimensions.
    pub fn validate_shape(&self, grid: &Grid) -> bool {
        self.grid_shape == (grid.nx, grid.ny, grid.nz)
    }

    /// Zero all workspace arrays (useful for debugging).
    pub fn clear(&mut self) {
        self.fft_buffer.fill(Complex::new(0.0, 0.0));
        self.real_buffer.fill(0.0);
        self.k_space_buffer.fill(0.0);
        self.temp_buffer.fill(0.0);
    }

    /// Get the statically pre-allocated memory in bytes.
    ///
    /// 1 × `fft_buffer` (`Complex<f64>`, 16 B/elem) + 3 × real buffers (8 B/elem).
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let complex_size = std::mem::size_of::<Complex<f64>>();
        let real_size = std::mem::size_of::<f64>();
        let n = self.grid_shape.0 * self.grid_shape.1 * self.grid_shape.2;
        complex_size * n + 3 * real_size * n
    }
}

impl ScratchArena for SolverWorkspace {
    #[inline]
    fn memory_bytes(&self) -> usize {
        self.memory_usage()
    }

    #[inline]
    fn clear(&mut self) {
        // Delegate to the inherent method — single source of truth.
        SolverWorkspace::clear(self);
    }
}

//! Workspace Management for Born Series Solvers
//!
//! This module provides memory-efficient workspace management for Born series
//! computations, enabling reuse of intermediate arrays and reducing memory
//! allocation overhead for iterative methods.

use ndarray::Array3;
use num_complex::Complex64;

/// Workspace for Born series computations
#[derive(Debug)]
pub struct BornWorkspace {
    /// Primary workspace array for field computations
    pub field_workspace: Array3<Complex64>,
    /// Secondary workspace for Green's function operations
    pub green_workspace: Array3<Complex64>,
    /// Workspace for heterogeneity potential computation
    pub heterogeneity_workspace: Array3<Complex64>,
    /// Temporary arrays for FFT operations (if needed)
    pub fft_temp: Vec<Array3<Complex64>>,
    /// Residual computation workspace
    pub residual_workspace: Array3<f64>,
}

impl BornWorkspace {
    /// Create a new workspace with specified dimensions
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let shape = (nx, ny, nz);
        Self {
            field_workspace: Array3::zeros(shape),
            green_workspace: Array3::zeros(shape),
            heterogeneity_workspace: Array3::zeros(shape),
            fft_temp: Vec::new(),
            residual_workspace: Array3::zeros(shape),
        }
    }

    /// Resize workspace to new dimensions
    pub fn resize(&mut self, nx: usize, ny: usize, nz: usize) {
        let shape = (nx, ny, nz);
        self.field_workspace = Array3::zeros(shape);
        self.green_workspace = Array3::zeros(shape);
        self.heterogeneity_workspace = Array3::zeros(shape);
        self.residual_workspace = Array3::zeros(shape);
        self.fft_temp.clear();
    }

    /// Clear all workspace arrays (set to zero)
    pub fn clear(&mut self) {
        self.field_workspace.fill(Complex64::new(0.0, 0.0));
        self.green_workspace.fill(Complex64::new(0.0, 0.0));
        self.heterogeneity_workspace.fill(Complex64::new(0.0, 0.0));
        self.residual_workspace.fill(0.0);
    }

    /// Get memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let complex_size = std::mem::size_of::<Complex64>();
        let real_size = std::mem::size_of::<f64>();

        self.field_workspace.len() * complex_size
            + self.green_workspace.len() * complex_size
            + self.heterogeneity_workspace.len() * complex_size
            + self.residual_workspace.len() * real_size
            + self
                .fft_temp
                .iter()
                .map(|arr| arr.len() * complex_size)
                .sum::<usize>()
    }
}

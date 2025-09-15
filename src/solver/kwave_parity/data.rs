//! k-Wave Data Structures
//!
//! Internal data structures for k-Wave solver implementation,
//! following GRASP principle of cohesive, focused modules.

use ndarray::Array3;
use rustfft::num_complex::Complex64;

/// Helper struct for k-space initialization data
pub(super) struct KSpaceData {
    pub kappa: Array3<f64>,
    pub k_vec: (Array3<f64>, Array3<f64>, Array3<f64>),
}

/// Helper struct for field array initialization
pub(super) struct FieldArrays {
    pub p: Array3<f64>,
    pub p_k: Array3<Complex64>,
    pub ux: Array3<f64>,
    pub uy: Array3<f64>,
    pub uz: Array3<f64>,
}

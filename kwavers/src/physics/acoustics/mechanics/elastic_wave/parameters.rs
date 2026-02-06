//! Parameter structures for elastic wave updates
//!
//! This module defines parameter containers that reduce coupling between
//! components following SOLID principles.

use ndarray::{Array3, ArrayView3};
use num_complex::Complex;

/// Type alias for complex 3D arrays used in spectral methods
pub type Complex3D = Array3<Complex<f64>>;

/// Parameters for stress update operations
/// Follows SOLID principles by reducing parameter coupling
#[derive(Debug)]
pub struct StressUpdateParams<'a> {
    pub vx_fft: &'a Complex3D,
    pub vy_fft: &'a Complex3D,
    pub vz_fft: &'a Complex3D,
    pub sxx_fft: &'a Complex3D,
    pub syy_fft: &'a Complex3D,
    pub szz_fft: &'a Complex3D,
    pub sxy_fft: &'a Complex3D,
    pub sxz_fft: &'a Complex3D,
    pub syz_fft: &'a Complex3D,
    pub kx: &'a Array3<f64>,
    pub ky: &'a Array3<f64>,
    pub kz: &'a Array3<f64>,
    pub lame_lambda: &'a Array3<f64>,
    pub lame_mu: &'a Array3<f64>,
    pub density: ArrayView3<'a, f64>,
    pub dt: f64,
}

/// Parameters for velocity update operations
/// Follows SOLID principles by reducing parameter coupling
#[derive(Debug)]
pub struct VelocityUpdateParams<'a> {
    pub vx_fft: &'a Complex3D,
    pub vy_fft: &'a Complex3D,
    pub vz_fft: &'a Complex3D,
    pub txx_fft: &'a Complex3D,
    pub tyy_fft: &'a Complex3D,
    pub tzz_fft: &'a Complex3D,
    pub txy_fft: &'a Complex3D,
    pub txz_fft: &'a Complex3D,
    pub tyz_fft: &'a Complex3D,
    pub kx: &'a Array3<f64>,
    pub ky: &'a Array3<f64>,
    pub kz: &'a Array3<f64>,
    pub density: ArrayView3<'a, f64>,
    pub dt: f64,
}

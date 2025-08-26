//! Field structures for elastic wave simulation
//!
//! This module defines the field containers used in elastic wave propagation,
//! following SOLID principles with clear separation of concerns.

use ndarray::Array3;
use num_complex::Complex;

/// Type alias for complex 3D arrays to reduce type complexity
pub type Complex3D = Array3<Complex<f64>>;

/// Stress field components in 3D elastic media
/// Follows SOLID principles with single responsibility
#[derive(Debug)]
pub struct StressFields {
    /// Normal stress components
    pub txx: Complex3D,
    pub tyy: Complex3D,
    pub tzz: Complex3D,
    /// Shear stress components (symmetric tensor)
    pub txy: Complex3D,
    pub txz: Complex3D,
    pub tyz: Complex3D,
}

/// Velocity field components in 3D elastic media
#[derive(Debug)]
pub struct VelocityFields {
    pub vx: Complex3D,
    pub vy: Complex3D,
    pub vz: Complex3D,
}

impl StressFields {
    /// Create new stress fields with given dimensions
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            txx: Array3::zeros((nx, ny, nz)),
            tyy: Array3::zeros((nx, ny, nz)),
            tzz: Array3::zeros((nx, ny, nz)),
            txy: Array3::zeros((nx, ny, nz)),
            txz: Array3::zeros((nx, ny, nz)),
            tyz: Array3::zeros((nx, ny, nz)),
        }
    }

    /// Reset all fields to zero
    pub fn reset(&mut self) {
        self.txx.fill(Complex::new(0.0, 0.0));
        self.tyy.fill(Complex::new(0.0, 0.0));
        self.tzz.fill(Complex::new(0.0, 0.0));
        self.txy.fill(Complex::new(0.0, 0.0));
        self.txz.fill(Complex::new(0.0, 0.0));
        self.tyz.fill(Complex::new(0.0, 0.0));
    }
}

impl VelocityFields {
    /// Create new velocity fields with given dimensions
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            vx: Array3::zeros((nx, ny, nz)),
            vy: Array3::zeros((nx, ny, nz)),
            vz: Array3::zeros((nx, ny, nz)),
        }
    }

    /// Reset all fields to zero
    pub fn reset(&mut self) {
        self.vx.fill(Complex::new(0.0, 0.0));
        self.vy.fill(Complex::new(0.0, 0.0));
        self.vz.fill(Complex::new(0.0, 0.0));
    }
}

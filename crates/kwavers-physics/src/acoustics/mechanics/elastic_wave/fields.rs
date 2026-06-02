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
    /// Normal stress components (real values for time-domain)
    pub txx: Array3<f64>,
    pub tyy: Array3<f64>,
    pub tzz: Array3<f64>,
    /// Shear stress components (symmetric tensor)
    pub txy: Array3<f64>,
    pub txz: Array3<f64>,
    pub tyz: Array3<f64>,
}

/// Velocity field components in 3D elastic media
#[derive(Debug)]
pub struct VelocityFields {
    pub vx: Array3<f64>,
    pub vy: Array3<f64>,
    pub vz: Array3<f64>,
}

impl StressFields {
    /// Create new stress fields with given dimensions
    #[must_use]
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
        self.txx.fill(0.0);
        self.tyy.fill(0.0);
        self.tzz.fill(0.0);
        self.txy.fill(0.0);
        self.txz.fill(0.0);
        self.tyz.fill(0.0);
    }
}

impl VelocityFields {
    /// Create new velocity fields with given dimensions
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            vx: Array3::zeros((nx, ny, nz)),
            vy: Array3::zeros((nx, ny, nz)),
            vz: Array3::zeros((nx, ny, nz)),
        }
    }

    /// Reset all fields to zero
    pub fn reset(&mut self) {
        self.vx.fill(0.0);
        self.vy.fill(0.0);
        self.vz.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// After filling all stress components with non-zero values, `reset` zeroes
    /// every element.
    #[test]
    fn stress_fields_reset_zeroes_all_components() {
        let mut f = StressFields::new(4, 4, 4);
        f.txx.fill(1.0);
        f.tyy.fill(2.0);
        f.tzz.fill(3.0);
        f.txy.fill(4.0);
        f.txz.fill(5.0);
        f.tyz.fill(6.0);
        f.reset();
        for v in f
            .txx
            .iter()
            .chain(f.tyy.iter())
            .chain(f.tzz.iter())
            .chain(f.txy.iter())
            .chain(f.txz.iter())
            .chain(f.tyz.iter())
        {
            assert_eq!(*v, 0.0);
        }
    }

    /// After filling all velocity components with non-zero values, `reset`
    /// zeroes every element.
    #[test]
    fn velocity_fields_reset_zeroes_all_components() {
        let mut f = VelocityFields::new(4, 4, 4);
        f.vx.fill(1.5);
        f.vy.fill(2.5);
        f.vz.fill(3.5);
        f.reset();
        for v in f.vx.iter().chain(f.vy.iter()).chain(f.vz.iter()) {
            assert_eq!(*v, 0.0);
        }
    }
}

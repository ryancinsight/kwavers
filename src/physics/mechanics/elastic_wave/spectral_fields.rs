//! Spectral field representations for elastic wave propagation
//!
//! This module provides complex-valued field structures required for
//! spectral methods using FFT-based derivatives.

use ndarray::Array3;
use num_complex::Complex;

/// Complex-valued stress field components for spectral methods
#[derive(Debug, Clone)]
pub struct SpectralStressFields {
    /// Normal stress components in frequency domain
    pub txx: Array3<Complex<f64>>,
    pub tyy: Array3<Complex<f64>>,
    pub tzz: Array3<Complex<f64>>,
    /// Shear stress components in frequency domain
    pub txy: Array3<Complex<f64>>,
    pub txz: Array3<Complex<f64>>,
    pub tyz: Array3<Complex<f64>>,
}

impl SpectralStressFields {
    /// Create new spectral stress fields
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

    /// Initialize from real-valued stress fields via FFT
    #[must_use]
    pub fn from_real(real_fields: &super::fields::StressFields) -> Self {
        use crate::utils::fft_3d_array;

        let (nx, ny, nz) = real_fields.txx.dim();
        let mut spectral = Self::new(nx, ny, nz);

        // Convert real fields to complex for FFT
        spectral.txx = fft_3d_array(&real_fields.txx);
        spectral.tyy = fft_3d_array(&real_fields.tyy);
        spectral.tzz = fft_3d_array(&real_fields.tzz);
        spectral.txy = fft_3d_array(&real_fields.txy);
        spectral.txz = fft_3d_array(&real_fields.txz);
        spectral.tyz = fft_3d_array(&real_fields.tyz);

        spectral
    }

    /// Convert back to real-valued fields via inverse FFT
    #[must_use]
    pub fn to_real(&self) -> super::fields::StressFields {
        use crate::utils::ifft_3d_array;

        let (nx, ny, nz) = self.txx.dim();
        let mut real_fields = super::fields::StressFields::new(nx, ny, nz);

        // Take real part after inverse FFT (ifft_3d_array returns real Array3)
        real_fields.txx = ifft_3d_array(&self.txx);
        real_fields.tyy = ifft_3d_array(&self.tyy);
        real_fields.tzz = ifft_3d_array(&self.tzz);
        real_fields.txy = ifft_3d_array(&self.txy);
        real_fields.txz = ifft_3d_array(&self.txz);
        real_fields.tyz = ifft_3d_array(&self.tyz);

        real_fields
    }
}

/// Complex-valued velocity field components for spectral methods
#[derive(Debug, Clone)]
pub struct SpectralVelocityFields {
    /// Velocity components in frequency domain
    pub vx: Array3<Complex<f64>>,
    pub vy: Array3<Complex<f64>>,
    pub vz: Array3<Complex<f64>>,
}

impl SpectralVelocityFields {
    /// Create new spectral velocity fields
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            vx: Array3::zeros((nx, ny, nz)),
            vy: Array3::zeros((nx, ny, nz)),
            vz: Array3::zeros((nx, ny, nz)),
        }
    }

    /// Initialize from real-valued velocity fields via FFT
    #[must_use]
    pub fn from_real(real_fields: &super::fields::VelocityFields) -> Self {
        use crate::utils::fft_3d_array;

        let (nx, ny, nz) = real_fields.vx.dim();
        let mut spectral = Self::new(nx, ny, nz);

        spectral.vx = fft_3d_array(&real_fields.vx);
        spectral.vy = fft_3d_array(&real_fields.vy);
        spectral.vz = fft_3d_array(&real_fields.vz);

        spectral
    }

    /// Convert back to real-valued fields via inverse FFT
    #[must_use]
    pub fn to_real(&self) -> super::fields::VelocityFields {
        use crate::utils::ifft_3d_array;

        let (nx, ny, nz) = self.vx.dim();
        let mut real_fields = super::fields::VelocityFields::new(nx, ny, nz);

        // ifft_3d_array returns real Array3 directly
        real_fields.vx = ifft_3d_array(&self.vx);
        real_fields.vy = ifft_3d_array(&self.vy);
        real_fields.vz = ifft_3d_array(&self.vz);

        real_fields
    }
}

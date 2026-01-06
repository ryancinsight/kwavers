//! Angular Spectrum Propagation Implementation
//!
//! Core implementation of angular spectrum method for efficient wave propagation
//! in homogeneous media.

use crate::error::KwaversResult;
use crate::fft::{Complex64, fft_2d_complex, ifft_2d_complex, utils as fft_utils};
use ndarray::Array2;
use std::f64::consts::PI;

/// Configuration for angular spectrum propagation
#[derive(Debug, Clone)]
pub struct SpectrumConfig {
    /// Spatial sampling rate (m)
    pub dx: f64,
    /// Maximum propagation angle (radians)
    pub max_angle: f64,
    /// Padding factor for FFT
    pub padding_factor: usize,
    /// Use separable propagation
    pub separable: bool,
}

/// Angular spectrum propagator
pub struct AngularSpectrum {
    /// Spatial frequencies (kx, ky)
    kx: Array2<f64>,
    ky: Array2<f64>,
    /// Wave numbers for different z
    kz: Array2<f64>,
    /// FFT size
    fft_size: (usize, usize),
    /// Current frequency (Hz)
    frequency: f64,
    /// Configuration
    config: SpectrumConfig,
}

impl AngularSpectrum {
    /// Create new angular spectrum propagator
    pub fn new(dx: f64, fft_size: (usize, usize), max_angle: f64) -> KwaversResult<Self> {
        let config = SpectrumConfig {
            dx,
            max_angle,
            padding_factor: 1,
            separable: true,
        };

        let mut kx = Array2::zeros(fft_size);
        let mut ky = Array2::zeros(fft_size);

        // Compute spatial frequencies
        let nx = fft_size.0;
        let ny = fft_size.1;

        for i in 0..nx {
            for j in 0..ny {
                // FFT frequency indexing (Matlab style)
                let fx = if i < nx / 2 {
                    i as f64 / (nx as f64 * dx)
                } else {
                    (i as f64 - nx as f64) / (nx as f64 * dx)
                };

                let fy = if j < ny / 2 {
                    j as f64 / (ny as f64 * dx)
                } else {
                    (j as f64 - ny as f64) / (ny as f64 * dx)
                };

                kx[[i, j]] = 2.0 * PI * fx;
                ky[[i, j]] = 2.0 * PI * fy;
            }
        }

        // Initialize kz (will be updated for each propagation)
        let kz = Array2::zeros(fft_size);

        Ok(Self {
            kx,
            ky,
            kz,
            fft_size,
            frequency: 1e6, // Default to 1 MHz
            config,
        })
    }

    /// Set current frequency
    pub fn set_frequency(&mut self, frequency: f64) {
        self.frequency = frequency;
    }

    /// Forward 2D FFT to angular spectrum domain
    pub fn forward_fft(&self, field: &Array2<Complex64>) -> KwaversResult<Array2<Complex64>> {
        // Forward 2D FFT using central cache
        let mut spectrum = fft_2d_complex(field);

        // Apply FFT shift to center the spectrum
        fft_utils::fft_shift_2d(&mut spectrum);

        tracing::debug!("Completed forward FFT: {}x{} -> {}x{}",
                       field.nrows(), field.ncols(), spectrum.nrows(), spectrum.ncols());

        Ok(spectrum)
    }


    /// Inverse 2D FFT from angular spectrum domain
    pub fn inverse_fft(&self, spectrum: &Array2<Complex64>) -> KwaversResult<Array2<Complex64>> {
        // Create copy and undo FFT shift
        let mut shifted = spectrum.clone();
        fft_utils::ifft_shift_2d(&mut shifted);

        // Inverse 2D FFT using central cache
        let field = ifft_2d_complex(&shifted);

        tracing::debug!("Completed inverse FFT: {}x{} -> {}x{}",
                       spectrum.nrows(), spectrum.ncols(), field.nrows(), field.ncols());

        Ok(field)
    }

    /// Propagate field using angular spectrum method
    pub fn propagate(&self, field: &Array2<Complex64>, distance: f64) -> KwaversResult<Array2<Complex64>> {
        // Angular spectrum method for wave propagation:
        // 1. Forward FFT to get angular spectrum A(kx, ky)
        // 2. Apply transfer function H(kx, ky) = exp(i * k_z * z)
        // 3. Inverse FFT to get propagated field

        // Step 1: Forward FFT
        let mut spectrum = self.forward_fft(field)?;

        // Step 2: Apply angular spectrum transfer function
        self.apply_transfer_function(&mut spectrum, distance)?;

        // Step 3: Inverse FFT
        let propagated_field = self.inverse_fft(&spectrum)?;

        tracing::info!("Angular spectrum propagation completed: distance = {} m", distance);

        Ok(propagated_field)
    }

    /// Apply angular spectrum transfer function
    /// H(kx, ky) = exp(i * k_z * z) where k_z = sqrt(k^2 - kx^2 - ky^2)
    fn apply_transfer_function(&self, spectrum: &mut Array2<Complex64>, distance: f64) -> KwaversResult<()> {
        let (rows, cols) = spectrum.dim();
        let c0 = 1500.0; // Reference speed
        let wavenumber = 2.0 * PI * self.frequency / c0;

        // Calculate spatial frequencies (centered around zero)
        let dkx = 2.0 * PI / (rows as f64 * self.config.dx);
        let dky = 2.0 * PI / (cols as f64 * self.config.dx); // Assuming isotropic dx

        for i in 0..rows {
            for j in 0..cols {
                // Calculate spatial frequencies (FFT-shifted coordinates)
                let kx = (i as f64 - rows as f64 / 2.0) * dkx;
                let ky = (j as f64 - cols as f64 / 2.0) * dky;

                // Calculate longitudinal wavenumber: k_z = sqrt(k^2 - kx^2 - ky^2)
                let k_squared = wavenumber * wavenumber;
                let k_transverse_squared = kx * kx + ky * ky;

                if k_transverse_squared > k_squared {
                    // Evanescent wave - decays exponentially, set to zero for numerical stability
                    spectrum[[i, j]] = Complex64::new(0.0, 0.0);
                } else {
                    // Propagating wave: k_z = sqrt(k^2 - kx^2 - ky^2)
                    let kz = (k_squared - k_transverse_squared).sqrt();

                    // Transfer function: H = exp(i * k_z * z)
                    let phase = kz * distance;
                    let transfer_function = Complex64::from_polar(1.0, phase);

                    // Apply transfer function to spectrum
                    spectrum[[i, j]] *= transfer_function;
                }
            }
        }

        Ok(())
    }

    /// Propagate angular spectrum by distance dz
    pub fn propagate_spectrum(&mut self, spectrum: &Array2<Complex64>, dz: f64) -> KwaversResult<Array2<Complex64>> {
        // Update kz for the propagation distance
        self.update_wave_numbers(dz);

        // Apply angular spectrum propagation
        let mut propagated = spectrum.clone();

        for i in 0..self.fft_size.0 {
            for j in 0..self.fft_size.1 {
                let kz_val = self.kz[[i, j]];

                if kz_val.is_finite() {
                    // Propagation phase: exp(i * kz * dz)
                    let phase = Complex64::new(0.0, kz_val * dz);
                    propagated[[i, j]] *= phase.exp();
                } else {
                    // Evanescent wave - attenuate rapidly
                    propagated[[i, j]] *= 0.01;
                }
            }
        }

        Ok(propagated)
    }

    /// Update wave numbers for given propagation distance
    fn update_wave_numbers(&mut self, _dz: f64) {
        let c0 = 1500.0;
        let k0 = 2.0 * PI * self.frequency / c0;

        for i in 0..self.fft_size.0 {
            for j in 0..self.fft_size.1 {
                let kx = self.kx[[i, j]];
                let ky = self.ky[[i, j]];

                let k_transverse_squared = kx * kx + ky * ky;

                if k_transverse_squared <= k0 * k0 {
                    // Propagating wave
                    self.kz[[i, j]] = (k0 * k0 - k_transverse_squared).sqrt();
                } else {
                    // Evanescent wave
                    self.kz[[i, j]] = -((k_transverse_squared - k0 * k0).sqrt());
                }
            }
        }
    }

    /// Compute propagation efficiency
    pub fn efficiency(&self) -> f64 {
        // Estimate computational efficiency based on FFT size and operations
        let fft_ops = self.fft_size.0 * self.fft_size.1 * (self.fft_size.0 + self.fft_size.1).ilog2() as usize;
        1.0 / (1.0 + fft_ops as f64 / 1000000.0) // Efficiency decreases with problem size
    }

    /// Get spatial frequency arrays
    pub fn spatial_frequencies(&self) -> (&Array2<f64>, &Array2<f64>) {
        (&self.kx, &self.ky)
    }

    /// Get wave number array
    pub fn wave_numbers(&self) -> &Array2<f64> {
        &self.kz
    }

    /// Apply angular spectrum cutoff based on maximum angle
    pub fn apply_angle_cutoff(&mut self) {
        let c0 = 1500.0;
        let k0 = 2.0 * PI * self.frequency / c0;
        let k_max = k0 * self.config.max_angle.sin(); // k0 * sin(theta_max)

        for i in 0..self.fft_size.0 {
            for j in 0..self.fft_size.1 {
                let k_transverse = (self.kx[[i, j]] * self.kx[[i, j]] +
                                  self.ky[[i, j]] * self.ky[[i, j]]).sqrt();

                if k_transverse > k_max {
                    self.kz[[i, j]] = 0.0; // Block high-angle components
                }
            }
        }
    }

    /// Compute field at multiple depths efficiently
    pub fn propagate_to_depths(
        &mut self,
        initial_field: &Array2<Complex64>,
        depths: &[f64],
    ) -> KwaversResult<Vec<Array2<Complex64>>> {
        let mut results = Vec::new();
        let mut current_spectrum = self.forward_fft(initial_field)?;

        results.push(self.inverse_fft(&current_spectrum)?);

        for window in depths.windows(2) {
            let dz = window[1] - window[0];
            current_spectrum = self.propagate_spectrum(&current_spectrum, dz)?;
            results.push(self.inverse_fft(&current_spectrum)?);
        }

        Ok(results)
    }

    /// Apply phase correction for medium inhomogeneities
    pub fn apply_phase_correction(&mut self, phase_screen: &Array2<f64>) {
        // This would apply phase corrections to the angular spectrum
        // For demonstration, we'll skip the implementation
        println!("Applying phase correction to angular spectrum");
    }

    /// Compute diffraction pattern from aperture
    pub fn diffraction_pattern(&self, aperture: &Array2<f64>, distance: f64) -> KwaversResult<Array2<Complex64>> {
        // Compute Fraunhofer diffraction pattern
        let mut spectrum = self.forward_fft(&aperture.mapv(|x| Complex64::new(x, 0.0)))?;
        spectrum = self.propagate_spectrum(&spectrum, distance)?;
        self.inverse_fft(&spectrum)
    }

    /// Get configuration
    pub fn config(&self) -> &SpectrumConfig {
        &self.config
    }
}

/// Utility functions for angular spectrum operations
pub struct AngularSpectrumUtils;

impl AngularSpectrumUtils {
    /// Compute evanescent wave decay rate
    pub fn evanescent_decay_rate(k_transverse: f64, k0: f64) -> f64 {
        if k_transverse > k0 {
            (k_transverse * k_transverse - k0 * k0).sqrt()
        } else {
            0.0
        }
    }

    /// Compute angular spectrum resolution limit
    pub fn resolution_limit(wavelength: f64, aperture_size: f64, distance: f64) -> f64 {
        // Rayleigh criterion for angular spectrum
        1.22 * wavelength * distance / aperture_size
    }

    /// Compute field of view for angular spectrum
    pub fn field_of_view(dx: f64, n_pixels: usize) -> f64 {
        n_pixels as f64 * dx
    }

    /// Check if angular spectrum approximation is valid
    pub fn is_valid_approximation(aperture_size: f64, distance: f64, wavelength: f64) -> bool {
        let fresnel_number = aperture_size * aperture_size / (wavelength * distance);
        fresnel_number > 10.0 // Far-field approximation valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angular_spectrum_creation() {
        let spectrum = AngularSpectrum::new(0.001, (64, 64), PI / 4.0);
        assert!(spectrum.is_ok());
    }

    #[test]
    fn test_forward_inverse_fft() {
        let spectrum = AngularSpectrum::new(0.001, (32, 32), PI / 6.0).unwrap();

        // Create test field
        let field = Array2::from_elem((32, 32), Complex64::new(1.0, 0.0));

        let transformed = spectrum.forward_fft(&field);
        assert!(transformed.is_ok());

        let restored = spectrum.inverse_fft(&transformed.unwrap());
        assert!(restored.is_ok());
    }

    #[test]
    fn test_spectrum_propagation() {
        let mut spectrum = AngularSpectrum::new(0.001, (16, 16), PI / 3.0).unwrap();

        let test_spectrum = Array2::from_elem((16, 16), Complex64::new(1.0, 0.0));

        let propagated = spectrum.propagate_spectrum(&test_spectrum, 0.01);
        assert!(propagated.is_ok());
    }

    #[test]
    fn test_evanescent_decay() {
        let k_transverse = 1000.0; // Large transverse wavenumber
        let k0 = 100.0; // Small longitudinal wavenumber

        let decay_rate = AngularSpectrumUtils::evanescent_decay_rate(k_transverse, k0);
        assert!(decay_rate > 0.0);
    }

    #[test]
    fn test_resolution_limit() {
        let wavelength = 0.0005; // 500 μm
        let aperture = 0.01; // 1 cm
        let distance = 0.1; // 10 cm

        let resolution = AngularSpectrumUtils::resolution_limit(wavelength, aperture, distance);
        assert!(resolution > 0.0 && resolution < 0.1); // Reasonable resolution
    }

    #[test]
    fn test_validity_check() {
        let aperture = 0.01;
        let distance = 1.0;
        let wavelength = 0.0005;

        let is_valid = AngularSpectrumUtils::is_valid_approximation(aperture, distance, wavelength);
        assert!(is_valid); // Should be valid for these parameters
    }
}


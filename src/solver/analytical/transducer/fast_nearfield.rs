//! Fast Nearfield Method (FNM) for Efficient Transducer Field Computation
//!
//! This module implements the Fast Nearfield Method for computing acoustic pressure
//! fields from transducer elements with O(n) complexity.

use crate::domain::source::transducers::rectangular::RectangularTransducer;
use crate::math::fft::{fft_2d_complex, ifft_2d_complex, Complex64};
use ndarray::{s, Array2, Array3, Axis};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Configuration for Fast Nearfield Method
#[derive(Debug, Clone)]
pub struct FNMConfig {
    /// Grid spacing in x-direction (m)
    pub dx: f64,
    /// Grid spacing in y-direction (m)
    pub dy: f64,
    /// Number of angular spectrum points (Nx, Ny)
    pub angular_spectrum_size: (usize, usize),
    /// Maximum k-space extent (as fraction of Nyquist)
    pub k_max_factor: f64,
    /// Use separable approximation for faster computation
    pub separable_approximation: bool,
}

impl Default for FNMConfig {
    fn default() -> Self {
        Self {
            dx: 0.1e-3, // 0.1 mm
            dy: 0.1e-3, // 0.1 mm
            angular_spectrum_size: (512, 512),
            k_max_factor: 2.0,
            separable_approximation: false,
        }
    }
}

/// Angular spectrum factors for a given z-plane
#[derive(Debug, Clone)]
pub struct AngularSpectrumFactors {
    /// Z distance (m)
    pub z: f64,
    /// Angular spectrum of Green's function (complex)
    pub green_spectrum: Array2<Complex64>,
    /// kx coordinates
    pub kx: Vec<f64>,
    /// ky coordinates
    pub ky: Vec<f64>,
}

/// Fast Nearfield Method solver
#[derive(Debug)]
pub struct FastNearfieldSolver {
    /// Configuration
    config: FNMConfig,
    /// Cached angular spectrum factors by z-distance
    cached_factors: HashMap<u64, AngularSpectrumFactors>,
    /// Current transducer geometry
    transducer: Option<RectangularTransducer>,
    /// Wave speed (m/s)
    c0: f64,
    /// Density (kg/m³)
    rho0: f64,
}

impl FastNearfieldSolver {
    /// Create new FNM solver
    pub fn new(config: FNMConfig) -> Result<Self, String> {
        Ok(Self {
            config,
            cached_factors: HashMap::new(),
            transducer: None,
            c0: 1500.0,   // Default water speed
            rho0: 1000.0, // Default water density
        })
    }

    /// Set transducer geometry
    pub fn set_transducer(&mut self, transducer: RectangularTransducer) {
        self.transducer = Some(transducer);
        self.cached_factors.clear(); // Clear cache when transducer changes
    }

    /// Set medium properties
    pub fn set_medium(&mut self, c0: f64, rho0: f64) {
        self.c0 = c0;
        self.rho0 = rho0;
        self.cached_factors.clear(); // Clear cache when medium changes
    }

    /// Precompute angular spectrum factors for a given z-distance
    pub fn precompute_factors(&mut self, z: f64) -> Result<(), String> {
        let transducer = self
            .transducer
            .as_ref()
            .ok_or("Transducer not set. Call set_transducer() first.")?;

        let z_key = (z * 1e9) as u64; // Use nanometer precision for key

        if self.cached_factors.contains_key(&z_key) {
            return Ok(()); // Already computed
        }

        let factors = self.compute_angular_spectrum_factors(transducer, z)?;
        self.cached_factors.insert(z_key, factors);

        Ok(())
    }

    /// Compute angular spectrum factors for Green's function
    fn compute_angular_spectrum_factors(
        &self,
        transducer: &RectangularTransducer,
        z: f64,
    ) -> Result<AngularSpectrumFactors, String> {
        let (n_kx, n_ky) = self.config.angular_spectrum_size;
        let k = transducer.wavenumber(self.c0);

        // Create k-space grid
        let dkx = 2.0 * PI / (self.config.dx * n_kx as f64);
        let dky = 2.0 * PI / (self.config.dx * n_ky as f64);

        let _kx_max = self.config.k_max_factor * PI / self.config.dx;
        let _ky_max = self.config.k_max_factor * PI / self.config.dy;

        let mut kx = Vec::with_capacity(n_kx);
        let mut ky = Vec::with_capacity(n_ky);

        // Create k-space coordinates (centered at zero)
        for i in 0..n_kx {
            let i_centered = if i < n_kx / 2 {
                i as f64
            } else {
                i as f64 - n_kx as f64
            };
            kx.push(i_centered * dkx);
        }

        for i in 0..n_ky {
            let i_centered = if i < n_ky / 2 {
                i as f64
            } else {
                i as f64 - n_ky as f64
            };
            ky.push(i_centered * dky);
        }

        // Compute angular spectrum of Green's function
        // Based on McGough (2004) and Kelly & McGough (2006)
        let mut green_spectrum = Array2::<Complex64>::zeros((n_kx, n_ky));

        for (i, &kx_val) in kx.iter().enumerate() {
            for (j, &ky_val) in ky.iter().enumerate() {
                let k_rho_squared = kx_val * kx_val + ky_val * ky_val;

                if k_rho_squared < k * k {
                    // Propagating wave
                    let kz = (k * k - k_rho_squared).sqrt();

                    if z > 0.0 {
                        // Forward propagation: angular spectrum factor for Rayleigh-Sommerfeld
                        // Ĝ(kx,ky,z) = (kz/k) * exp(j*kz*z) for the angular spectrum
                        let phase_factor = Complex64::new(0.0, kz * z).exp();
                        let amplitude_factor = kz / k;
                        green_spectrum[[i, j]] =
                            Complex64::new(0.0, 1.0) * amplitude_factor * phase_factor;
                    } else if z == 0.0 {
                        // On-axis case - direct field
                        green_spectrum[[i, j]] = Complex64::new(1.0, 0.0);
                    } else {
                        // Backward propagation (less common but mathematically valid)
                        let phase_factor = Complex64::new(0.0, kz * z).exp();
                        let amplitude_factor = kz / k;
                        green_spectrum[[i, j]] =
                            Complex64::new(0.0, -1.0) * amplitude_factor * phase_factor;
                    }
                } else {
                    // Evanescent wave - set to zero (no propagation)
                    green_spectrum[[i, j]] = Complex64::new(0.0, 0.0);
                }
            }
        }

        Ok(AngularSpectrumFactors {
            z,
            green_spectrum,
            kx,
            ky,
        })
    }

    /// Compute pressure field from transducer velocity distribution
    pub fn compute_field(
        &self,
        velocity: &Array2<Complex64>,
        z: f64,
    ) -> Result<Array2<Complex64>, String> {
        let transducer = self
            .transducer
            .as_ref()
            .ok_or("Transducer not set. Call set_transducer() first.")?;

        let z_key = (z * 1e9) as u64;
        let factors = self.cached_factors.get(&z_key).ok_or(format!(
            "Angular spectrum factors not computed for z = {} m. Call precompute_factors() first.",
            z
        ))?;

        // Check velocity array dimensions match transducer elements
        let (n_elem_x, n_elem_y) = transducer.elements;
        if velocity.nrows() != n_elem_x || velocity.ncols() != n_elem_y {
            return Err(format!(
                "Velocity array dimensions ({}, {}) don't match transducer elements ({}, {})",
                velocity.nrows(),
                velocity.ncols(),
                n_elem_x,
                n_elem_y
            ));
        }

        // Zero-pad velocity to angular spectrum size
        let padded_velocity = self.zero_pad_velocity(velocity);

        // Forward FFT (angular spectrum of velocity)
        let velocity_spectrum = fft_2d_complex(&padded_velocity);

        // Multiply by Green's function angular spectrum
        let mut pressure_spectrum = velocity_spectrum;
        for ((i, j), val) in pressure_spectrum.indexed_iter_mut() {
            *val *= factors.green_spectrum[[i, j]];
        }

        // Scaling factor from Rayleigh-Sommerfeld theory
        let k = transducer.wavenumber(self.c0);
        let scaling = Complex64::new(0.0, self.rho0 * self.c0 * k / (2.0 * PI));

        for val in pressure_spectrum.iter_mut() {
            *val *= scaling;
        }

        // Inverse FFT to get spatial pressure field
        // Note: ifft_2d_complex already normalizes by 1/N
        let pressure_field = ifft_2d_complex(&pressure_spectrum);

        // Extract the central region corresponding to the transducer aperture
        let (n_kx, n_ky) = self.config.angular_spectrum_size;
        let start_x = (n_kx - n_elem_x) / 2;
        let start_y = (n_ky - n_elem_y) / 2;

        let result = pressure_field
            .slice(s![start_x..start_x + n_elem_x, start_y..start_y + n_elem_y])
            .to_owned();

        Ok(result)
    }

    /// Compute field at multiple z-distances (efficient batch computation)
    pub fn compute_field_stack(
        &self,
        velocity: &Array2<Complex64>,
        z_values: &[f64],
    ) -> Result<Array3<Complex64>, String> {
        let mut results = Vec::new();

        for &z in z_values {
            let field = self.compute_field(velocity, z)?;
            results.push(field);
        }

        // Stack into 3D array (z, x, y)
        let shape = (z_values.len(), results[0].nrows(), results[0].ncols());
        let mut stacked = Array3::<Complex64>::zeros(shape);

        for (i, field) in results.into_iter().enumerate() {
            stacked.index_axis_mut(Axis(0), i).assign(&field);
        }

        Ok(stacked)
    }

    /// Zero-pad velocity distribution to angular spectrum size
    fn zero_pad_velocity(&self, velocity: &Array2<Complex64>) -> Array2<Complex64> {
        let (n_elem_x, n_elem_y) = velocity.dim();
        let (n_kx, n_ky) = self.config.angular_spectrum_size;

        let mut padded = Array2::<Complex64>::zeros((n_kx, n_ky));

        let start_x = (n_kx - n_elem_x) / 2;
        let start_y = (n_ky - n_elem_y) / 2;

        padded
            .slice_mut(s![start_x..start_x + n_elem_x, start_y..start_y + n_elem_y])
            .assign(velocity);

        padded
    }

    /// Get cached z-distances
    pub fn cached_z_distances(&self) -> Vec<f64> {
        self.cached_factors
            .keys()
            .map(|&key| key as f64 / 1e9)
            .collect()
    }

    /// Clear angular spectrum cache
    pub fn clear_cache(&mut self) {
        self.cached_factors.clear();
    }

    /// Get memory usage estimate (bytes)
    pub fn memory_usage(&self) -> usize {
        let mut total = 0;

        // Cache memory
        for factors in self.cached_factors.values() {
            total += factors.green_spectrum.len() * std::mem::size_of::<Complex64>();
            total += factors.kx.len() * std::mem::size_of::<f64>();
            total += factors.ky.len() * std::mem::size_of::<f64>();
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnm_solver_creation() {
        let config = FNMConfig::default();
        let solver = FastNearfieldSolver::new(config);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_transducer_setup() {
        let config = FNMConfig::default();
        let mut solver = FastNearfieldSolver::new(config).unwrap();

        let transducer = RectangularTransducer {
            width: 10e-3,
            height: 10e-3,
            frequency: 1e6,
            elements: (32, 32),
        };

        solver.set_transducer(transducer);

        let (elem_width, elem_height) = solver.transducer.as_ref().unwrap().element_size();
        assert!((elem_width - 10e-3 / 32.0).abs() < 1e-9);
        assert!((elem_height - 10e-3 / 32.0).abs() < 1e-9);
    }

    #[test]
    fn test_precompute_factors() {
        let config = FNMConfig::default();
        let mut solver = FastNearfieldSolver::new(config).unwrap();

        let transducer = RectangularTransducer {
            width: 5e-3,
            height: 5e-3,
            frequency: 2e6,
            elements: (16, 16),
        };

        solver.set_transducer(transducer);
        solver.set_medium(1500.0, 1000.0);

        let result = solver.precompute_factors(25e-3); // 25 mm
        assert!(result.is_ok());

        // Check that factors were cached
        assert_eq!(solver.cached_z_distances().len(), 1);
        assert!((solver.cached_z_distances()[0] - 25e-3).abs() < 1e-9);
    }

    #[test]
    fn test_field_computation() {
        let config = FNMConfig {
            angular_spectrum_size: (64, 64), // Smaller for testing
            ..Default::default()
        };
        let mut solver = FastNearfieldSolver::new(config).unwrap();

        let transducer = RectangularTransducer {
            width: 5e-3,
            height: 5e-3,
            frequency: 2e6,
            elements: (16, 16),
        };

        solver.set_transducer(transducer);
        solver.precompute_factors(25e-3).unwrap();

        // Uniform velocity distribution
        let velocity = Array2::<Complex64>::from_elem((16, 16), Complex64::new(1.0, 0.0));

        let pressure = solver.compute_field(&velocity, 25e-3);
        assert!(pressure.is_ok());

        let pressure_field = pressure.unwrap();
        assert_eq!(pressure_field.dim(), (16, 16));

        // Check that result is not zero (basic sanity check)
        let sum: Complex64 = pressure_field.iter().sum();
        assert!(sum.norm() > 0.0);
    }

    #[test]
    fn test_memory_usage() {
        let config = FNMConfig::default();
        let mut solver = FastNearfieldSolver::new(config).unwrap();

        let transducer = RectangularTransducer {
            width: 10e-3,
            height: 10e-3,
            frequency: 1e6,
            elements: (32, 32),
        };

        solver.set_transducer(transducer);
        solver.precompute_factors(50e-3).unwrap();

        let usage = solver.memory_usage();
        assert!(usage > 0);

        // Clear cache and check memory drops
        solver.clear_cache();
        let usage_after_clear = solver.memory_usage();
        assert_eq!(usage_after_clear, 0);
    }
}

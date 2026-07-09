//! FastNearfieldSolver implementation.

use apollo::{fft_2d_complex, ifft_2d_complex, Complex64 as ApolloComplex64};
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_transducer::transducers::rectangular::RectangularTransducer;
use leto::Array2 as LetoArray2;
use leto::{Array2, Array3};
use kwavers_math::fft::Complex64;
use std::collections::HashMap;

use super::types::{AngularSpectrumFactors, FNMConfig};
use kwavers_core::constants::numerical::TWO_PI;

#[derive(Debug)]
pub struct FastNearfieldSolver {
    /// Configuration
    pub(super) config: FNMConfig,
    /// Cached angular spectrum factors by z-distance
    pub(super) cached_factors: HashMap<u64, AngularSpectrumFactors>,
    /// Current transducer geometry
    pub(super) transducer: Option<RectangularTransducer>,
    /// Wave speed (m/s)
    pub(super) c0: f64,
    /// Density (kg/m³)
    pub(super) rho0: f64,
    /// Precomputed kx coordinates
    kx: Vec<f64>,
    /// Precomputed ky coordinates
    ky: Vec<f64>,
}

impl FastNearfieldSolver {
    /// Create new FNM solver
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: FNMConfig) -> Result<Self, String> {
        let (n_kx, n_ky) = config.angular_spectrum_size;
        let dkx = TWO_PI / (config.dx * n_kx as f64);
        let dky = TWO_PI / (config.dy * n_ky as f64);

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

        Ok(Self {
            config,
            cached_factors: HashMap::new(),
            transducer: None,
            c0: SOUND_SPEED_WATER_SIM,
            rho0: DENSITY_WATER_NOMINAL,
            kx,
            ky,
        })
    }

    /// Set transducer geometry
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn set_transducer(&mut self, transducer: RectangularTransducer) {
        self.transducer = Some(transducer);
        self.cached_factors.clear(); // Clear cache when transducer changes
    }

    /// Set medium properties
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn set_medium(&mut self, c0: f64, rho0: f64) {
        self.c0 = c0;
        self.rho0 = rho0;
        self.cached_factors.clear(); // Clear cache when medium changes
    }

    /// Precompute angular spectrum factors for a given z-distance
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_angular_spectrum_factors(
        &self,
        transducer: &RectangularTransducer,
        z: f64,
    ) -> Result<AngularSpectrumFactors, String> {
        let (n_kx, n_ky) = self.config.angular_spectrum_size;
        let k = transducer.wavenumber(self.c0);

        // Compute angular spectrum of Green's function
        // Based on McGough (2004) and Kelly & McGough (2006)
        let mut green_spectrum = Array2::<Complex64>::from_elem((n_kx, n_ky), Complex64::default());

        for (i, &kx_val) in self.kx.iter().enumerate() {
            for (j, &ky_val) in self.ky.iter().enumerate() {
                let k_rho_squared = kx_val.mul_add(kx_val, ky_val * ky_val);

                if k_rho_squared < k * k {
                    // Propagating wave
                    let kz = k.mul_add(k, -k_rho_squared).sqrt();

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
            kx: self.kx.clone(),
            ky: self.ky.clone(),
        })
    }

    /// Compute pressure field from transducer velocity distribution
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
        if velocity.shape()[0] != n_elem_x || velocity.shape()[1] != n_elem_y {
            return Err(format!(
                "Velocity array dimensions ({}, {}) don't match transducer elements ({}, {})",
                velocity.shape()[0],
                velocity.shape()[1],
                n_elem_x,
                n_elem_y
            ));
        }

        // Zero-pad velocity to angular spectrum size
        let padded_velocity = self.zero_pad_velocity(velocity);
        let (n_kx, n_ky) = self.config.angular_spectrum_size;
        let padded_velocity = LetoArray2::from_shape_vec(
            [n_kx, n_ky],
            padded_velocity
                .iter()
                .map(|value| ApolloComplex64::new(value.re, value.im))
                .collect(),
        )
        .expect("fast-nearfield padded velocity shape must match its Leto FFT shape");

        // Forward FFT (angular spectrum of velocity)
        let velocity_spectrum = fft_2d_complex(&padded_velocity);

        // Multiply by Green's function angular spectrum
        let mut pressure_spectrum = velocity_spectrum;
        for i in 0..n_kx {
            for j in 0..n_ky {
                let factor = factors.green_spectrum[[i, j]];
                pressure_spectrum[[i, j]] *= ApolloComplex64::new(factor.re, factor.im);
            }
        }

        // Scaling factor from Rayleigh-Sommerfeld theory
        let k = transducer.wavenumber(self.c0);
        let scaling = ApolloComplex64::new(0.0, self.rho0 * self.c0 * k / (TWO_PI));

        for i in 0..n_kx {
            for j in 0..n_ky {
                pressure_spectrum[[i, j]] *= scaling;
            }
        }

        // Inverse FFT to get spatial pressure field
        // Note: ifft_2d_complex already normalizes by 1/N
        let pressure_field = ifft_2d_complex(&pressure_spectrum);

        // Extract the central region corresponding to the transducer aperture
        let start_x = (n_kx - n_elem_x) / 2;
        let start_y = (n_ky - n_elem_y) / 2;

        let mut result = Array2::<Complex64>::from_elem((n_elem_x, n_elem_y), Complex64::default());
        for i in 0..n_elem_x {
            for j in 0..n_elem_y {
                let value = pressure_field[[start_x + i, start_y + j]];
                result[[i, j]] = Complex64::new(value.re, value.im);
            }
        }

        Ok(result)
    }

    /// Compute field at multiple z-distances (efficient batch computation)
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
        let shape = ((z_values.shape()[0] * z_values.shape()[1] * z_values.shape()[2]), results[0].nrows(), results[0].ncols());
        let mut stacked = Array3::<Complex64>::zeros(shape);

        for (i, field) in results.into_iter().enumerate() {
            stacked.index_axis_mut(0, i).unwrap().assign(&field);
        }

        Ok(stacked)
    }

    /// Zero-pad velocity distribution to angular spectrum size
    fn zero_pad_velocity(&self, velocity: &Array2<Complex64>) -> Array2<Complex64> {
        let [n_elem_x, n_elem_y] = velocity.shape();
        let (n_kx, n_ky) = self.config.angular_spectrum_size;

        let mut padded = Array2::<Complex64>::from_elem((n_kx, n_ky), Complex64::default());

        let start_x = (n_kx - n_elem_x) / 2;
        let start_y = (n_ky - n_elem_y) / 2;

        padded
            .slice_mut(s![start_x..start_x + n_elem_x, start_y..start_y + n_elem_y]).unwrap().unwrap().assign(velocity);

        padded
    }

    /// Get cached z-distances
    #[must_use]
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
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let mut total = 0;

        // Cache memory
        for factors in self.cached_factors.values() {
            total += (factors.green_spectrum.shape()[0] * factors.green_spectrum.shape()[1] * factors.green_spectrum.shape()[2]) * std::mem::size_of::<Complex64>();
            total += (factors.kx.shape()[0] * factors.kx.shape()[1] * factors.kx.shape()[2]) * std::mem::size_of::<f64>();
            total += (factors.ky.shape()[0] * factors.ky.shape()[1] * factors.ky.shape()[2]) * std::mem::size_of::<f64>();
        }

        // Base memory for precomputed vectors
        total += (self.kx.shape()[0] * self.kx.shape()[1] * self.kx.shape()[2]) * std::mem::size_of::<f64>();
        total += (self.ky.shape()[0] * self.ky.shape()[1] * self.ky.shape()[2]) * std::mem::size_of::<f64>();

        total
    }
}

//! Complex-valued parabolic diffraction operator for proper energy conservation.
//!
//! # Pre-allocated scratch buffer
//!
//! `scratch` (`Array2<Complex64>`, shape (nx, ny)) is allocated once in
//! `ParabolicDiffractionOperator::new` and reused on every `apply_complex` call.
//! The 2-D FFT plan is obtained directly through Apollo's cached Leto-native API.
//!
//! Per `apply_complex` call this eliminates:
//! - 1 × `field.to_owned()` clone (~262 KB for 128×128)
//! - 1 × `fft_2d_complex()` allocating forward transform
//! - 1 × `ifft_2d_complex()` allocating inverse transform
//!
//! When called for every retarded-time slice (nt = 1000), one diffraction
//! half-step saves 3000 allocations (~786 MB total clones + FFT temporaries).

use apollo::{fft_2d_complex_inplace, ifft_2d_complex_inplace, Complex64 as ApolloComplex64};
use kwavers_math::fft::Complex64;
use leto::Array2 as LetoArray2;
use leto::{Array2, ArrayViewMut2};
use moirai_parallel::{enumerate_mut_with, Adaptive};

use super::KZKConfig;
use kwavers_core::constants::numerical::TWO_PI;

/// Parabolic diffraction operator using complex-valued computations for energy preservation.
///
/// All hot-path operations are zero-allocation: the (nx, ny) complex scratch
/// buffer and the FFT plan are pre-allocated at construction.
pub struct ParabolicDiffractionOperator {
    config: KZKConfig,
    kx2: Array2<f64>,
    ky2: Array2<f64>,
    /// Pre-allocated complex scratch buffer (nx, ny).
    ///
    /// Reused on every `apply_complex` call to hold the complex field during
    /// in-place FFT → phase-multiply → in-place IFFT.  Eliminates one clone
    /// and two allocating FFT round-trips per time slice.
    scratch: LetoArray2<ApolloComplex64>,
}

impl std::fmt::Debug for ParabolicDiffractionOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParabolicDiffractionOperator")
            .field("config", &self.config)
            .field("kx2_shape", &self.kx2.shape())
            .field("ky2_shape", &self.ky2.shape())
            .finish()
    }
}

impl ParabolicDiffractionOperator {
    /// Create new complex parabolic diffraction operator.
    ///
    /// Pre-computes the transverse wavenumber grids `kx²` and `ky²`,
    /// allocates the complex scratch buffer. FFT plan caching is owned by
    /// Apollo.
    #[must_use]
    pub fn new(config: &KZKConfig) -> Self {
        let nx = config.nx;
        let ny = config.ny;

        // Create k-space grids for transverse wavenumbers
        let mut kx2 = Array2::zeros((nx, ny));
        let mut ky2 = Array2::zeros((nx, ny));

        let dkx = TWO_PI / (nx as f64 * config.dx);
        let dky = TWO_PI / (ny as f64 * config.dx);

        // Standard FFT k-space ordering
        for i in 0..nx {
            let kx = if i <= nx / 2 {
                i as f64 * dkx
            } else {
                (i as f64 - nx as f64) * dkx
            };

            for j in 0..ny {
                kx2[[i, j]] = kx * kx;
            }
        }

        for j in 0..ny {
            let ky = if j <= ny / 2 {
                j as f64 * dky
            } else {
                (j as f64 - ny as f64) * dky
            };

            for i in 0..nx {
                ky2[[i, j]] = ky * ky;
            }
        }

        let scratch = LetoArray2::<ApolloComplex64>::zeros([nx, ny]);

        Self {
            config: config.clone(),
            kx2,
            ky2,
            scratch,
        }
    }

    /// Apply diffraction step to complex field (zero-allocation hot path).
    ///
    /// ## Algorithm
    ///
    /// 1. Copy `field` into `self.scratch` (one assign, no heap allocation).
    /// 2. Forward complex-to-complex in-place FFT on `self.scratch`.
    /// 3. Multiply each k-space mode by the parabolic propagator:
    ///    ```text
    ///    H(k_T) = exp(−i k_T² Δz / (2k₀))
    ///    ```
    ///    where k_T² = kx² + ky² and k₀ = ω₀/c₀.
    /// 4. Inverse complex-to-complex in-place FFT (includes 1/N normalisation).
    /// 5. Assign `self.scratch` back to `field`.
    ///
    /// Steps 2 and 4 are performed in-place on the pre-allocated scratch buffer,
    /// eliminating all per-call heap allocation.
    ///
    /// ## Sign convention
    ///
    /// The phase factor is `exp(−i·phase)` with `phase = k_T² Δz/(2k₀) ≥ 0`.
    /// The negative sign is physically correct (convergent phase accumulation
    /// in the forward direction); the positive sign would apply the conjugate
    /// propagator and reverse phase curvature.
    ///
    /// References: Lee & Hamilton (1995) eq. (4); Aanonsen et al. (1984) §3.
    pub fn apply_complex(&mut self, field: &mut ArrayViewMut2<Complex64>, step_size: f64) {
        let k0 = TWO_PI * self.config.frequency / self.config.c0;

        // Step 1: copy field slice into scratch (no heap allocation).
        let scratch = self
            .scratch
            .as_slice_mut()
            .expect("invariant: complex KZK diffraction scratch is standard-layout");
        if let Some(field_values) = field.as_slice() {
            enumerate_mut_with::<Adaptive, _, _>(scratch, |idx, s| {
                let value = field_values[idx];
                *s = ApolloComplex64::new(value.re, value.im);
            });
        } else {
            for (s, &value) in scratch.iter_mut().zip(field.as_view().iter()) {
                *s = ApolloComplex64::new(value.re, value.im);
            }
        }

        // Step 2: forward complex-to-complex in-place FFT.
        fft_2d_complex_inplace(&mut self.scratch);

        // Step 3: apply parabolic diffraction propagator H(k_T) per mode.
        //
        // H(k_T) = exp(−i k_T² Δz / (2k₀))
        //
        // Derived from ∂P̂/∂z = −i k_T²/(2k₀) P̂ where k₀ = ω₀/c₀.
        // The sign is NEGATIVE; the previous (wrong) sign exp(+i·phase) applied
        // the conjugate propagator and reversed the phase curvature of all modes.
        //
        // Refs: Lee & Hamilton (1995) eq. (4); Aanonsen et al. (1984) §3.
        //
        // ## Theorem (race-freedom, Step 3)
        //
        // The update is a pointwise `scratch[i,j] *= H(kx2[i,j], ky2[i,j])`.
        // Each element depends only on the collocated kx2 and ky2 values
        // (immutable read) and overwrites its own scratch cell (mutable write).
        // No two Moirai tasks share memory because each scratch index is visited
        // exactly once.
        {
            let kx2 = self
                .kx2
                .as_slice()
                .expect("invariant: complex KZK diffraction kx2 is standard-layout");
            let ky2 = self
                .ky2
                .as_slice()
                .expect("invariant: complex KZK diffraction ky2 is standard-layout");
            let scratch = self
                .scratch
                .as_slice_mut()
                .expect("invariant: complex KZK diffraction scratch is standard-layout");
            enumerate_mut_with::<Adaptive, _, _>(scratch, |idx, s| {
                let kt2 = kx2[idx] + ky2[idx];
                let phase = kt2 * step_size / (2.0 * k0);
                // exp(−i·phase): negative sign is physically correct
                *s *= ApolloComplex64::from_polar(1.0, -phase);
            });
        }

        // Step 4: inverse complex-to-complex in-place FFT (includes 1/N norm).
        ifft_2d_complex_inplace(&mut self.scratch);

        // Step 5: copy scratch back to the field view.
        let scratch = self
            .scratch
            .as_slice()
            .expect("invariant: complex KZK diffraction scratch is standard-layout");
        if let Some(field_values) = field.as_mut_slice() {
            enumerate_mut_with::<Adaptive, _, _>(field_values, |idx, value| {
                let scratch_value = scratch[idx];
                *value = Complex64::new(scratch_value.re, scratch_value.im);
            });
        } else {
            for (([_, _], value), &s) in field
                .reborrow()
                .indexed_iter_mut()
                .expect("invariant: 2-D field view yields indexed iterator")
                .zip(scratch.iter())
            {
                *value = Complex64::new(s.re, s.im);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::nonlinear::kzk::constants::{
        DEFAULT_BEAM_WAIST, DEFAULT_FREQUENCY, DEFAULT_GRID_SIZE, DEFAULT_WAVELENGTH,
    };
    use crate::validation::measure_beam_radius;
    use eunomia::assert_relative_eq;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use std::f64::consts::PI;

    #[test]
    fn test_complex_energy_conservation() {
        let config = KZKConfig {
            nx: 64,
            ny: 64,
            dx: DEFAULT_WAVELENGTH / 8.0,
            frequency: DEFAULT_FREQUENCY,
            c0: SOUND_SPEED_WATER_SIM,
            ..Default::default()
        };

        let mut op = ParabolicDiffractionOperator::new(&config);

        // Create complex Gaussian beam
        let beam_waist = DEFAULT_BEAM_WAIST;
        let mut field =
            Array2::<Complex64>::from_elem([config.nx, config.ny], Complex64::default());

        for i in 0..config.nx {
            for j in 0..config.ny {
                let x = (i as f64 - config.nx as f64 / 2.0) * config.dx;
                let y = (j as f64 - config.ny as f64 / 2.0) * config.dx;
                let r2 = x * x + y * y;
                field[[i, j]] = Complex64::new((-r2 / (beam_waist * beam_waist)).exp(), 0.0);
            }
        }

        let initial_energy: f64 = field.iter().map(|c| c.norm_sqr()).sum();

        // Propagate
        let dz = DEFAULT_WAVELENGTH;
        let mut field_view = field.view_mut();
        op.apply_complex(&mut field_view, dz);

        let final_energy: f64 = field.iter().map(|c| c.norm_sqr()).sum();

        println!("Complex field energy conservation:");
        println!("Initial energy: {:.6}", initial_energy);
        println!("Final energy: {:.6}", final_energy);
        println!("Energy ratio: {:.6}", final_energy / initial_energy);

        // Energy should be conserved
        assert!((final_energy / initial_energy - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_complex_gaussian_beam_propagation() {
        let config = KZKConfig {
            nx: DEFAULT_GRID_SIZE,
            ny: DEFAULT_GRID_SIZE,
            dx: DEFAULT_WAVELENGTH / 6.0,
            frequency: DEFAULT_FREQUENCY,
            c0: SOUND_SPEED_WATER_SIM,
            ..Default::default()
        };

        let mut op = ParabolicDiffractionOperator::new(&config);

        // Create complex Gaussian beam
        let beam_waist = DEFAULT_BEAM_WAIST;
        let mut field =
            Array2::<Complex64>::from_elem([config.nx, config.ny], Complex64::default());

        for i in 0..config.nx {
            for j in 0..config.ny {
                let x = (i as f64 - config.nx as f64 / 2.0) * config.dx;
                let y = (j as f64 - config.ny as f64 / 2.0) * config.dx;
                let r2 = x * x + y * y;
                field[[i, j]] = Complex64::new((-r2 / (beam_waist * beam_waist)).exp(), 0.0);
            }
        }

        // Propagate to Rayleigh distance
        let wavelength = config.c0 / config.frequency;
        let z_r = PI * beam_waist * beam_waist / wavelength;
        let steps = 50;
        let dz = z_r / steps as f64;

        for _ in 0..steps {
            let mut field_view = field.view_mut();
            op.apply_complex(&mut field_view, dz);
        }

        // Measure beam radius from intensity
        let intensity = field.mapv(|c| c.norm_sqr());
        let measured = measure_beam_radius(&intensity, config.dx);
        let expected = beam_waist * 2.0_f64.sqrt();

        println!("Complex field Gaussian beam test:");
        println!("Measured radius: {:.3}mm", measured * 1000.0);
        println!("Expected radius: {:.3}mm", expected * 1000.0);
        println!(
            "Error: {:.1}%",
            (measured - expected).abs() / expected * 100.0
        );

        // Should match within 5% with proper complex field handling
        assert_relative_eq!(measured, expected, epsilon = 0.05 * expected);
    }
}

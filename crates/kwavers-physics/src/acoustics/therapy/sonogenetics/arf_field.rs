//! Volumetric Acoustic Radiation Force (ARF) Field
//!
//! Computes the per-voxel ARF body force density from a PSTD/FDTD simulation
//! by accumulating time-averaged intensity over acoustic cycles.
//!
//! # Physics
//!
//! Time-averaged acoustic intensity (linear acoustics Poynting vector magnitude;
//! Temkin 2001 §2.5):
//!
//!   I(x) = ⟨p²(x,t)⟩ / (ρ(x) · c(x))   [W/m²]
//!
//! Acoustic radiation force body force density (Nyborg 1965; Nightingale 2002):
//!
//!   F(x) = 2 · α(x) · I(x) / c(x)        [N/m³]
//!
//! The factor 2 applies to a progressive wave in an absorbing medium where all
//! absorbed momentum is transferred to the medium (Sarvazyan 2010 Eq. 4).
//!
//! # Usage
//!
//! ```rust,ignore
//! let mut arf = VolumetricArfField::new(nx, ny, nz);
//! // Inside PSTD time loop (every step or every N steps):
//! arf.accumulate(solver.pressure());
//! // After at least one complete acoustic cycle:
//! arf.finalize(&absorption, &sound_speed, &density)?;
//! let body_force = arf.arf_density(); // F(x) [N/m³]
//! let intensity  = arf.intensity();   // I(x) [W/m²]
//! ```
//!
//! # References
//!
//! - Nyborg, W.L. (1965). Acoustic streaming. *Physical Acoustics*, 2B, 265-331.
//! - Nightingale, K. et al. (2002). Acoustic radiation force impulse imaging.
//!   *Ultrasound Med. Biol.*, 28(2), 227-235.
//! - Sarvazyan, A.P. et al. (2010). Acoustic radiation force — a review.
//!   *Curr. Med. Imaging Rev.*, 6(1), 15-25.
//! - Temkin, S. (2001). *Elements of Acoustics*. Acoustical Society of America.

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use ndarray::{Array3, Zip};

/// Volumetric ARF field accumulator and body-force extractor.
///
/// Accumulates instantaneous pressure snapshots from the running PSTD/FDTD solver,
/// computes the time-averaged intensity, and derives the per-voxel ARF body force
/// density `F(x) = 2·α·I/c`.
///
/// # Invariants
///
/// - `n_samples > 0` must hold before calling `finalize`.
/// - `sound_speed` and `density` passed to `finalize` must be strictly positive
///   at every voxel; voxels violating this produce zero output.
/// - `intensity` and `arf_density` fields are only valid after `finalize` has been called.
#[derive(Debug)]
pub struct VolumetricArfField {
    /// Running sum of p² for time-averaging.
    p_sq_sum: Array3<f64>,
    /// Number of accumulated pressure samples.
    n_samples: usize,
    /// Time-averaged intensity I(x) = ⟨p²⟩/(ρ·c) [W/m²].
    intensity: Array3<f64>,
    /// ARF body force density F(x) = 2·α·I/c [N/m³].
    arf_density: Array3<f64>,
    /// Grid shape (nx, ny, nz).
    shape: (usize, usize, usize),
}

impl VolumetricArfField {
    /// Create a new accumulator for a grid of given dimensions.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let shape = (nx, ny, nz);
        Self {
            p_sq_sum: Array3::zeros(shape),
            n_samples: 0,
            intensity: Array3::zeros(shape),
            arf_density: Array3::zeros(shape),
            shape,
        }
    }

    /// Accumulate one pressure snapshot.
    ///
    /// Computes `p²` element-wise and adds to the running sum.
    /// Call once per simulation time step (or once per N steps for sub-cycle sampling).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `pressure` dimensions do not match the grid shape
    /// passed to [`new`][VolumetricArfField::new].
    pub fn accumulate(&mut self, pressure: &Array3<f64>) {
        debug_assert_eq!(
            pressure.dim(),
            self.shape,
            "pressure shape {:?} ≠ accumulator shape {:?}",
            pressure.dim(),
            self.shape
        );
        Zip::from(&mut self.p_sq_sum)
            .and(pressure)
            .par_for_each(|acc, &p| *acc += p * p);
        self.n_samples += 1;
    }

    /// Finalize intensity and ARF body-force fields from the accumulated p² sum.
    ///
    /// # Arguments
    ///
    /// - `absorption`   — per-voxel absorption coefficient α(x) [Np/m]
    /// - `sound_speed`  — per-voxel sound speed c(x) [m/s]; must be > 0
    /// - `density`      — per-voxel density ρ(x) [kg/m³]; must be > 0
    ///
    /// # Errors
    ///
    /// Returns `Err(KwaversError::Validation(...))` if no samples have been accumulated.
    ///
    /// # Formulas
    ///
    /// ```text
    /// I(x) = ⟨p²⟩ / (ρ·c)       [Poynting vector; Temkin 2001 §2.5]
    /// F(x) = 2·α·I / c           [Nyborg 1965; Sarvazyan 2010 Eq. 4]
    /// ```
    pub fn finalize(
        &mut self,
        absorption: &Array3<f64>,
        sound_speed: &Array3<f64>,
        density: &Array3<f64>,
    ) -> KwaversResult<()> {
        if self.n_samples == 0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "n_samples".to_owned(),
                value: 0.0,
                reason: "must accumulate at least one pressure snapshot before finalizing"
                    .to_owned(),
            }));
        }
        let scale = 1.0 / self.n_samples as f64;
        Zip::from(&mut self.intensity)
            .and(&mut self.arf_density)
            .and(&self.p_sq_sum)
            .and(absorption)
            .and(sound_speed)
            .and(density)
            .par_for_each(|intensity, arf, &p_sq, &alpha, &c, &rho| {
                let p_sq_mean = p_sq * scale;
                if c > 0.0 && rho > 0.0 {
                    // I(x) = ⟨p²⟩ / (ρ·c)  [W/m²]
                    *intensity = p_sq_mean / (rho * c);
                    // F(x) = 2·α·I / c  [N/m³]
                    *arf = 2.0 * alpha * *intensity / c;
                } else {
                    *intensity = 0.0;
                    *arf = 0.0;
                }
            });
        Ok(())
    }

    /// Time-averaged acoustic intensity I(x) [W/m²].
    ///
    /// Valid only after [`finalize`][VolumetricArfField::finalize] has been called.
    #[inline]
    #[must_use]
    pub fn intensity(&self) -> &Array3<f64> {
        &self.intensity
    }

    /// ARF body force density F(x) = 2·α·I/c [N/m³].
    ///
    /// Valid only after [`finalize`][VolumetricArfField::finalize] has been called.
    #[inline]
    #[must_use]
    pub fn arf_density(&self) -> &Array3<f64> {
        &self.arf_density
    }

    /// Number of accumulated pressure samples.
    #[inline]
    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Reset the p² accumulator and sample count.
    ///
    /// Intensity and ARF fields retain their last finalized values.
    pub fn reset_accumulator(&mut self) {
        self.p_sq_sum.fill(0.0);
        self.n_samples = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use ndarray::Array3;

    /// Uniform pressure field: I = p² / (ρ·c); F = 2·α·I / c.
    ///
    /// Analytical reference:
    ///   p = 1000 Pa, ρ = 1000 kg/m³, c = 1500 m/s, α = 5.0 Np/m
    ///   I = 1000² / (1000 · 1500) = 1e6/1.5e6 = 0.6667 W/m²
    ///   F = 2 · 5 · 0.6667 / 1500 = 4.444e-3 N/m³
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_uniform_field_analytical() {
        let (nx, ny, nz) = (4, 4, 4);
        let p_val = 1000.0_f64;
        let rho = DENSITY_WATER_NOMINAL;
        let c = SOUND_SPEED_WATER_SIM;
        let alpha = 5.0_f64;

        let mut arf = VolumetricArfField::new(nx, ny, nz);
        let pressure = Array3::from_elem((nx, ny, nz), p_val);

        // Accumulate same field 10 times (time average = p² invariant)
        for _ in 0..10 {
            arf.accumulate(&pressure);
        }

        let absorption = Array3::from_elem((nx, ny, nz), alpha);
        let sound_speed = Array3::from_elem((nx, ny, nz), c);
        let density = Array3::from_elem((nx, ny, nz), DENSITY_WATER_NOMINAL);

        arf.finalize(&absorption, &sound_speed, &density).unwrap();

        let expected_intensity = p_val * p_val / (rho * c);
        let expected_arf = 2.0 * alpha * expected_intensity / c;

        // Every voxel must match the analytical value
        for v in arf.intensity().iter() {
            assert_relative_eq!(*v, expected_intensity, max_relative = 1e-12);
        }
        for v in arf.arf_density().iter() {
            assert_relative_eq!(*v, expected_arf, max_relative = 1e-12);
        }
    }

    /// Zero absorption: F must be zero everywhere; I is non-zero.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_zero_absorption_yields_zero_arf() {
        let (nx, ny, nz) = (4, 4, 4);
        let mut arf = VolumetricArfField::new(nx, ny, nz);
        let pressure = Array3::from_elem((nx, ny, nz), 500.0_f64);
        arf.accumulate(&pressure);

        let absorption = Array3::zeros((nx, ny, nz));
        let sound_speed = Array3::from_elem((nx, ny, nz), SOUND_SPEED_WATER_SIM);
        let density = Array3::from_elem((nx, ny, nz), DENSITY_WATER_NOMINAL);

        arf.finalize(&absorption, &sound_speed, &density).unwrap();

        for &v in arf.arf_density().iter() {
            assert_eq!(v, 0.0, "ARF must be zero when absorption is zero");
        }
        // Intensity must still be non-zero
        let any_nonzero = arf.intensity().iter().any(|&v| v > 0.0);
        assert!(
            any_nonzero,
            "Intensity must be positive when pressure is nonzero"
        );
    }

    /// Calling finalize with n_samples = 0 must return an error.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_finalize_before_accumulate_is_error() {
        let mut arf = VolumetricArfField::new(2, 2, 2);
        let absorption = Array3::zeros((2, 2, 2));
        let sound_speed = Array3::from_elem((2, 2, 2), SOUND_SPEED_WATER_SIM);
        let density = Array3::from_elem((2, 2, 2), DENSITY_WATER_NOMINAL);
        let result = arf.finalize(&absorption, &sound_speed, &density);
        assert!(
            result.is_err(),
            "finalize before accumulate must return Err"
        );
    }

    /// Reset clears the accumulator but preserves the last finalized fields.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_reset_preserves_last_finalized() {
        let (nx, ny, nz) = (2, 2, 2);
        let mut arf = VolumetricArfField::new(nx, ny, nz);
        let pressure = Array3::from_elem((nx, ny, nz), 200.0_f64);
        arf.accumulate(&pressure);
        let absorption = Array3::from_elem((nx, ny, nz), 2.0);
        let sound_speed = Array3::from_elem((nx, ny, nz), SOUND_SPEED_WATER_SIM);
        let density = Array3::from_elem((nx, ny, nz), DENSITY_WATER_NOMINAL);
        arf.finalize(&absorption, &sound_speed, &density).unwrap();

        let intensity_before = arf.intensity()[[0, 0, 0]];
        arf.reset_accumulator();

        assert_eq!(arf.n_samples(), 0);
        // Intensity field must be unchanged after reset
        assert_relative_eq!(
            arf.intensity()[[0, 0, 0]],
            intensity_before,
            max_relative = 1e-15
        );
    }
}

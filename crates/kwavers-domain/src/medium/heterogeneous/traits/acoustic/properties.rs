//! Acoustic properties implementation for heterogeneous media
//!
//! **Single Responsibility**: Pure acoustic property access per SOLID principles
//! **Evidence-Based**: Following Hamilton & Blackstock (1998) acoustic theory

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use crate::medium::heterogeneous::{
    core::HeterogeneousMedium, interpolation::HetTrilinearInterpolator,
};
use crate::medium::{
    acoustic::AcousticProperties,
    core::{ArrayAccess, CoreMedium, MIN_PHYSICAL_DENSITY, MIN_PHYSICAL_SOUND_SPEED},
};

impl CoreMedium for HeterogeneousMedium {
    /// Get sound speed at grid point with physical limits
    #[inline]
    fn sound_speed(&self, i: usize, j: usize, k: usize) -> f64 {
        self.sound_speed[[i, j, k]].max(MIN_PHYSICAL_SOUND_SPEED)
    }

    /// Get density at grid point with physical limits
    #[inline]
    fn density(&self, i: usize, j: usize, k: usize) -> f64 {
        self.density[[i, j, k]].max(MIN_PHYSICAL_DENSITY)
    }

    /// Get nonlinearity parameter at grid point
    #[inline]
    fn nonlinearity(&self, i: usize, j: usize, k: usize) -> f64 {
        self.nonlinearity[[i, j, k]]
    }

    /// Get absorption coefficient at grid point
    #[inline]
    fn absorption(&self, i: usize, j: usize, k: usize) -> f64 {
        self.absorption[[i, j, k]]
    }

    /// Get maximum sound speed in the medium
    fn max_sound_speed(&self) -> f64 {
        crate::medium::max_sound_speed(&self.sound_speed)
    }

    /// Get reference frequency for absorption calculations
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn reference_frequency(&self) -> f64 {
        self.reference_frequency
    }

    /// Check if medium properties are spatially varying
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[inline]
    fn is_homogeneous(&self) -> bool {
        false
    }

    /// Validate medium properties against grid dimensions
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn validate(&self, grid: &Grid) -> KwaversResult<()> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let expected_shape = [nx, ny, nz];

        if self.density.shape() != expected_shape {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: format!("{expected_shape:?}"),
                    actual: format!("{:?}", self.density.shape()),
                },
            ));
        }

        Ok(())
    }
}

impl ArrayAccess for HeterogeneousMedium {
    /// Get sound speed array view (zero-copy)
    fn sound_speed_array(&self) -> ndarray::ArrayView3<'_, f64> {
        self.sound_speed.view()
    }

    /// Get density array view (zero-copy)  
    fn density_array(&self) -> ndarray::ArrayView3<'_, f64> {
        self.density.view()
    }

    /// Get nonlinearity array view (zero-copy)
    fn nonlinearity_array(&self) -> ndarray::ArrayView3<'_, f64> {
        self.nonlinearity.view()
    }

    /// Get absorption array view (zero-copy)
    fn absorption_array(&self) -> ndarray::ArrayView3<'_, f64> {
        self.absorption.view()
    }
}

impl AcousticProperties for HeterogeneousMedium {
    /// Absorption coefficient at continuous coordinates and frequency.
    ///
    /// **Theorem (Szabo power-law absorption, Szabo 1994 J.Acoust.Soc.Am.).**
    /// The frequency-dependent absorption coefficient α(f) [Np/m] satisfies
    ///
    /// ```text
    ///   α(f) = α₀_Np · (f / f_ref)^y
    /// ```
    ///
    /// where α₀_Np is the reference absorption in Np/m (converted from
    /// dB/(MHz^y cm) via `DB_TO_NP * 100 * f_ref_MHz^y`), and y is the
    /// per-voxel power-law exponent stored in `self.alpha_power`.  Both α₀
    /// and y are trilinearly interpolated when `use_trilinear_interpolation`
    /// is set.
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        let base_absorption = HetTrilinearInterpolator::get_field_value(
            &self.absorption,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        );
        let exponent = HetTrilinearInterpolator::get_field_value(
            &self.alpha_power,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        );

        let freq_ratio = frequency / self.reference_frequency;
        base_absorption * freq_ratio.powf(exponent)
    }

    /// Power-law prefactor α₀ [dB/(MHz^y·cm)] at the given continuous coordinates.
    ///
    /// Returns the spatially interpolated absorption prefactor from `self.absorption`.
    /// The default trait implementation returns 0.0 (lossless), so this override is
    /// required for `effective_alpha_db` in the solver to pick up the medium's field.
    fn alpha_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        HetTrilinearInterpolator::get_field_value(
            &self.absorption,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }

    /// Per-voxel power-law exponent y at the given continuous coordinates.
    ///
    /// Used by solvers that need the exponent separately from the absorption
    /// coefficient (e.g. fractional-Laplacian PSTD absorber).
    fn alpha_power(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        HetTrilinearInterpolator::get_field_value(
            &self.alpha_power,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }

    /// Get acoustic diffusivity at continuous coordinates
    fn acoustic_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let sound_speed = HetTrilinearInterpolator::get_field_value(
            &self.sound_speed,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        );
        let density = HetTrilinearInterpolator::get_field_value(
            &self.density,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        );

        // Acoustic diffusivity: D = c²/ρ (classical fluid mechanics formula)
        // Exact for homogeneous fluids, per Morse & Ingard (1968) "Theoretical Acoustics"
        sound_speed * sound_speed / density
    }
}

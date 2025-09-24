//! Acoustic properties implementation for heterogeneous media
//!
//! **Single Responsibility**: Pure acoustic property access per SOLID principles
//! **Evidence-Based**: Following Hamilton & Blackstock (1998) acoustic theory

use crate::error::{KwaversError, KwaversResult, ValidationError};
use crate::grid::Grid;
use crate::medium::{
    acoustic::AcousticProperties,
    core::{ArrayAccess, CoreMedium, MIN_PHYSICAL_DENSITY, MIN_PHYSICAL_SOUND_SPEED},
};
use crate::medium::heterogeneous::{
    core::HeterogeneousMedium,
    interpolation::TrilinearInterpolator,
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
    fn reference_frequency(&self) -> f64 {
        self.reference_frequency
    }

    /// Check if medium properties are spatially varying
    #[inline]
    fn is_homogeneous(&self) -> bool {
        false
    }

    /// Validate medium properties against grid dimensions
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
    /// Get absorption coefficient at continuous coordinates and frequency
    ///
    /// **Method**: Trilinear interpolation with frequency scaling per Hamilton & Blackstock
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        let base_absorption = TrilinearInterpolator::get_field_value(
            &self.absorption,
            x, y, z,
            grid,
            self.use_trilinear_interpolation
        );
        
        // Frequency-dependent scaling
        let freq_ratio = frequency / self.reference_frequency;
        base_absorption * freq_ratio.powf(1.0) // Power law with exponent 1.0
    }

    /// Get acoustic diffusivity at continuous coordinates
    fn acoustic_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let sound_speed = TrilinearInterpolator::get_field_value(
            &self.sound_speed,
            x, y, z,
            grid,
            self.use_trilinear_interpolation
        );
        let density = TrilinearInterpolator::get_field_value(
            &self.density,
            x, y, z,
            grid,
            self.use_trilinear_interpolation
        );
        
        // Acoustic diffusivity = c²/ρ (simplified form)
        sound_speed * sound_speed / density
    }
}
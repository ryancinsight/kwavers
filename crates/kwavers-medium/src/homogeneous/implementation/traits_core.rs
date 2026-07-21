use leto::{Array3, ArrayView3, ArrayViewMut3};

use crate::core::{ArrayAccess, CoreMedium};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;

use super::HomogeneousMedium;

impl CoreMedium for HomogeneousMedium {
    fn density(&self, _i: usize, _j: usize, _k: usize) -> f64 {
        self.density
    }

    fn sound_speed(&self, _i: usize, _j: usize, _k: usize) -> f64 {
        self.sound_speed
    }

    fn reference_frequency(&self) -> f64 {
        self.reference_frequency
    }

    fn absorption(&self, _i: usize, _j: usize, _k: usize) -> f64 {
        self.absorption_alpha
    }

    fn nonlinearity(&self, _i: usize, _j: usize, _k: usize) -> f64 {
        self.nonlinearity
    }

    fn max_sound_speed(&self) -> f64 {
        self.sound_speed
    }

    fn is_homogeneous(&self) -> bool {
        true
    }

    fn validate(&self, _grid: &Grid) -> KwaversResult<()> {
        if self.density <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "density".to_owned(),
                value: self.density,
                reason: "Density must be positive".to_owned(),
            }));
        }

        if self.sound_speed <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "sound_speed".to_owned(),
                value: self.sound_speed,
                reason: "Sound speed must be positive".to_owned(),
            }));
        }

        Ok(())
    }
}

impl ArrayAccess for HomogeneousMedium {
    fn density_array(&self) -> ArrayView3<'_, f64> {
        self.density_cache
            .get_or_init(self.grid_shape, self.density)
    }

    fn sound_speed_array(&self) -> ArrayView3<'_, f64> {
        self.sound_speed_cache
            .get_or_init(self.grid_shape, self.sound_speed)
    }

    fn density_array_mut(&mut self) -> Option<ArrayViewMut3<'_, f64>> {
        // A truly homogeneous medium has no per-voxel density to mutate.
        None
    }

    fn sound_speed_array_mut(&mut self) -> Option<ArrayViewMut3<'_, f64>> {
        // A truly homogeneous medium has no per-voxel sound speed to mutate.
        None
    }

    fn absorption_array(&self) -> ArrayView3<'_, f64> {
        self.absorption_cache.get_or_init_with(|| {
            let alpha_at_ref = self.absorption_alpha
                * (self.reference_frequency / kwavers_core::constants::MHZ_TO_HZ)
                    .powf(self.absorption_power);
            Array3::from_elem(self.grid_shape, alpha_at_ref)
        })
    }

    fn nonlinearity_array(&self) -> ArrayView3<'_, f64> {
        self.nonlinearity_cache
            .get_or_init(self.grid_shape, self.nonlinearity)
    }
}

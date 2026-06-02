//! Read-only field views, point-wise samplers, and dimension/spacing queries.

use ndarray::ArrayView2;

use super::CylindricalMediumProjection;
use crate::grid::{CylindricalTopology, Grid};
use crate::medium::Medium;

impl<'a, M: Medium> CylindricalMediumProjection<'a, M> {
    /// Get the projected sound speed field (nz × nr)
    #[inline]
    #[must_use]
    pub fn sound_speed_field(&self) -> ArrayView2<'_, f64> {
        self.sound_speed_2d.view()
    }

    /// Get the projected density field (nz × nr)
    #[inline]
    #[must_use]
    pub fn density_field(&self) -> ArrayView2<'_, f64> {
        self.density_2d.view()
    }

    /// Get the projected absorption field (nz × nr)
    #[inline]
    #[must_use]
    pub fn absorption_field(&self) -> ArrayView2<'_, f64> {
        self.absorption_2d.view()
    }

    /// Get the projected nonlinearity field (nz × nr)
    ///
    /// Returns `Some(view)` if the medium has nonlinearity, `None` otherwise.
    #[inline]
    #[must_use]
    pub fn nonlinearity_field(&self) -> Option<ArrayView2<'_, f64>> {
        self.nonlinearity_2d.as_ref().map(|arr| arr.view())
    }

    /// Get maximum sound speed in the projected medium (m/s)
    #[inline]
    #[must_use]
    pub fn max_sound_speed(&self) -> f64 {
        self.max_sound_speed
    }

    /// Get minimum sound speed in the projected medium (m/s)
    #[inline]
    #[must_use]
    pub fn min_sound_speed(&self) -> f64 {
        self.min_sound_speed
    }

    /// Check if the projected medium is homogeneous
    #[inline]
    #[must_use]
    pub fn is_homogeneous(&self) -> bool {
        self.is_homogeneous
    }

    /// Get the underlying 3D medium reference
    #[inline]
    #[must_use]
    pub fn medium(&self) -> &M {
        self.medium
    }

    /// Get the grid reference
    #[inline]
    #[must_use]
    pub fn grid(&self) -> &Grid {
        self.grid
    }

    /// Get the cylindrical topology reference
    #[inline]
    #[must_use]
    pub fn topology(&self) -> &CylindricalTopology {
        self.topology
    }

    /// Get sound speed at a specific (iz, ir) point
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds (debug builds only).
    #[inline]
    #[must_use]
    pub fn sound_speed_at(&self, iz: usize, ir: usize) -> f64 {
        self.sound_speed_2d[[iz, ir]]
    }

    /// Get density at a specific (iz, ir) point
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds (debug builds only).
    #[inline]
    #[must_use]
    pub fn density_at(&self, iz: usize, ir: usize) -> f64 {
        self.density_2d[[iz, ir]]
    }

    /// Get absorption at a specific (iz, ir) point
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds (debug builds only).
    #[inline]
    #[must_use]
    pub fn absorption_at(&self, iz: usize, ir: usize) -> f64 {
        self.absorption_2d[[iz, ir]]
    }

    /// Get nonlinearity at a specific (iz, ir) point
    ///
    /// Returns 0.0 if the medium has no nonlinearity.
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds (debug builds only).
    #[inline]
    #[must_use]
    pub fn nonlinearity_at(&self, iz: usize, ir: usize) -> f64 {
        self.nonlinearity_2d
            .as_ref()
            .map_or(0.0, |arr| arr[[iz, ir]])
    }

    /// Get grid dimensions (nz, nr)
    #[inline]
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.topology.nz, self.topology.nr)
    }

    /// Get grid spacing (dz, dr)
    #[inline]
    #[must_use]
    pub fn spacing(&self) -> (f64, f64) {
        (self.topology.dz, self.topology.dr)
    }
}

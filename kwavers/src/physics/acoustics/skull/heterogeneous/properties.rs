use super::constants::{HU_CORTICAL, HU_WATER};
use super::model::HeterogeneousSkull;
use super::types::SkullLayer;

impl HeterogeneousSkull {
    /// Compute bone volume fraction (BVF) from Hounsfield unit.
    ///
    /// ```text
    /// φ = clamp((HU − HU_water) / (HU_cortical − HU_water), 0, 1)
    /// ```
    #[inline]
    #[must_use] 
    pub fn bone_volume_fraction(hu: f64) -> f64 {
        ((hu - HU_WATER) / (HU_CORTICAL - HU_WATER)).clamp(0.0, 1.0)
    }

    /// Classify a voxel into a skull layer based on its BVF.
    #[inline]
    #[must_use] 
    pub fn classify_layer(hu: f64) -> SkullLayer {
        let phi = Self::bone_volume_fraction(hu);
        if phi < 0.15 {
            SkullLayer::SoftTissue
        } else if phi < 0.75 {
            SkullLayer::Diploe
        } else {
            SkullLayer::Cortical
        }
    }

    /// Get acoustic impedance at position [Pa·s/m = Rayl].
    #[must_use] 
    pub fn impedance_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.density[[i, j, k]] * self.sound_speed[[i, j, k]]
    }
}

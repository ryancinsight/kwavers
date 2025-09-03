//! Tissue property caching utilities

use crate::medium::absorption::{TissueType, TISSUE_PROPERTIES};
use ndarray::Array3;
use std::collections::HashMap;

/// Cache for tissue properties to avoid repeated lookups
pub struct TissuePropertyCache {
    density_cache: HashMap<TissueType, f64>,
    sound_speed_cache: HashMap<TissueType, f64>,
    absorption_cache: HashMap<TissueType, f64>,
    nonlinearity_cache: HashMap<TissueType, f64>,
}

impl Default for TissuePropertyCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TissuePropertyCache {
    /// Create a new property cache
    #[must_use]
    pub fn new() -> Self {
        Self {
            density_cache: HashMap::new(),
            sound_speed_cache: HashMap::new(),
            absorption_cache: HashMap::new(),
            nonlinearity_cache: HashMap::new(),
        }
    }

    /// Get cached density for tissue type
    pub fn get_density(&mut self, tissue_type: TissueType) -> f64 {
        *self.density_cache.entry(tissue_type).or_insert_with(|| {
            TISSUE_PROPERTIES
                .get(&tissue_type)
                .map_or(1000.0, |p| p.density)
        })
    }

    /// Get cached sound speed for tissue type
    pub fn get_sound_speed(&mut self, tissue_type: TissueType) -> f64 {
        *self
            .sound_speed_cache
            .entry(tissue_type)
            .or_insert_with(|| {
                TISSUE_PROPERTIES
                    .get(&tissue_type)
                    .map_or(1500.0, |p| p.sound_speed)
            })
    }

    /// Populate array with cached values
    pub fn populate_density_array(
        &mut self,
        tissue_map: &Array3<TissueType>,
        output: &mut Array3<f64>,
    ) {
        ndarray::Zip::from(tissue_map)
            .and(output)
            .for_each(|&tissue, out| {
                *out = self.get_density(tissue);
            });
    }

    /// Populate array with cached sound speed values
    pub fn populate_sound_speed_array(
        &mut self,
        tissue_map: &Array3<TissueType>,
        output: &mut Array3<f64>,
    ) {
        ndarray::Zip::from(tissue_map)
            .and(output)
            .for_each(|&tissue, out| {
                *out = self.get_sound_speed(tissue);
            });
    }
}

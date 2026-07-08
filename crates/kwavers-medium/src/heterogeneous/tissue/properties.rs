//! Tissue property caching utilities

use crate::absorption::{AbsorptionTissueType, TISSUE_PROPERTIES};
use crate::parallel::zip_mut_ref;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use leto::Array3;
use std::collections::HashMap;

/// Cache for tissue properties to avoid repeated lookups
#[derive(Debug)]
pub struct TissuePropertyCache {
    density_cache: HashMap<AbsorptionTissueType, f64>,
    sound_speed_cache: HashMap<AbsorptionTissueType, f64>,
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
        }
    }

    /// Get cached density for tissue type
    pub fn get_density(&mut self, tissue_type: AbsorptionTissueType) -> f64 {
        *self.density_cache.entry(tissue_type).or_insert_with(|| {
            TISSUE_PROPERTIES
                .get(&tissue_type)
                .map_or(DENSITY_WATER_NOMINAL, |p| p.density)
        })
    }

    /// Get cached sound speed for tissue type
    pub fn get_sound_speed(&mut self, tissue_type: AbsorptionTissueType) -> f64 {
        *self
            .sound_speed_cache
            .entry(tissue_type)
            .or_insert_with(|| {
                TISSUE_PROPERTIES
                    .get(&tissue_type)
                    .map_or(SOUND_SPEED_WATER_SIM, |p| p.sound_speed)
            })
    }

    /// Populate array with cached values
    pub fn populate_density_array(
        &mut self,
        tissue_map: &Array3<AbsorptionTissueType>,
        output: &mut Array3<f64>,
    ) {
        assert_eq!(
            output.shape(),
            tissue_map.shape(),
            "invariant: tissue density output shape must match tissue map"
        );

        for tissue in tissue_map.iter().copied() {
            self.get_density(tissue);
        }

        let density_cache = &self.density_cache;
        zip_mut_ref(output, tissue_map, |out, &tissue| {
            *out = density_cache
                .get(&tissue)
                .copied()
                .unwrap_or(DENSITY_WATER_NOMINAL);
        });
    }

    /// Populate array with cached sound speed values
    pub fn populate_sound_speed_array(
        &mut self,
        tissue_map: &Array3<AbsorptionTissueType>,
        output: &mut Array3<f64>,
    ) {
        assert_eq!(
            output.shape(),
            tissue_map.shape(),
            "invariant: tissue sound-speed output shape must match tissue map"
        );

        for tissue in tissue_map.iter().copied() {
            self.get_sound_speed(tissue);
        }

        let sound_speed_cache = &self.sound_speed_cache;
        zip_mut_ref(output, tissue_map, |out, &tissue| {
            *out = sound_speed_cache
                .get(&tissue)
                .copied()
                .unwrap_or(SOUND_SPEED_WATER_SIM);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn populate_density_array_matches_tissue_database() {
        let tissue_map = Array3::from_shape_vec(
            (2, 1, 2),
            vec![
                AbsorptionTissueType::Water,
                AbsorptionTissueType::Brain,
                AbsorptionTissueType::Fat,
                AbsorptionTissueType::SoftTissue,
            ],
        )
        .expect("invariant: tissue map fixture shape matches data length");
        let mut output = Array3::zeros([2, 1, 2]);
        let mut cache = TissuePropertyCache::new();

        cache.populate_density_array(&tissue_map, &mut output);

        for ((i, j, k), &tissue) in tissue_map.indexed_iter() {
            let expected = TISSUE_PROPERTIES
                .get(&tissue)
                .map_or(DENSITY_WATER_NOMINAL, |p| p.density);
            assert_eq!(output[[i, j, k]], expected);
        }
    }

    #[test]
    fn populate_sound_speed_array_matches_tissue_database() {
        let tissue_map = Array3::from_shape_vec(
            (2, 1, 2),
            vec![
                AbsorptionTissueType::Water,
                AbsorptionTissueType::Brain,
                AbsorptionTissueType::Fat,
                AbsorptionTissueType::SoftTissue,
            ],
        )
        .expect("invariant: tissue map fixture shape matches data length");
        let mut output = Array3::zeros([2, 1, 2]);
        let mut cache = TissuePropertyCache::new();

        cache.populate_sound_speed_array(&tissue_map, &mut output);

        for ((i, j, k), &tissue) in tissue_map.indexed_iter() {
            let expected = TISSUE_PROPERTIES
                .get(&tissue)
                .map_or(SOUND_SPEED_WATER_SIM, |p| p.sound_speed);
            assert_eq!(output[[i, j, k]], expected);
        }
    }

    #[test]
    #[should_panic(expected = "tissue density output shape")]
    fn populate_density_array_rejects_shape_mismatch() {
        let tissue_map = Array3::from_elem([2, 1, 1], AbsorptionTissueType::Water);
        let mut output = Array3::zeros([1, 1, 1]);
        let mut cache = TissuePropertyCache::new();

        cache.populate_density_array(&tissue_map, &mut output);
    }
}

//! `HeterogeneousTissueMedium` implementation

use super::{TissueMap, TissueRegion};
use crate::error::{KwaversResult, ValidationError};
use crate::grid::Grid;
use crate::medium::absorption::TissueType;
use crate::medium::{
    absorption::{tissue::TissueProperties, TISSUE_PROPERTIES},
    core::{CoreMedium, MIN_PHYSICAL_DENSITY, MIN_PHYSICAL_SOUND_SPEED},
};
use ndarray::Array3;

/// Heterogeneous tissue medium with spatial tissue type variations
#[derive(Debug, Clone)]
pub struct HeterogeneousTissueMedium {
    /// 3D map of tissue types
    tissue_map: TissueMap,
    /// Grid structure
    grid: Grid,
    /// Cache for density values
    cached_density: Option<Array3<f64>>,
    /// Cache for sound speed values
    cached_sound_speed: Option<Array3<f64>>,
}

impl HeterogeneousTissueMedium {
    /// Create a new heterogeneous tissue medium
    pub fn new(grid: Grid, default_tissue: TissueType) -> Self {
        let tissue_map = Array3::from_elem((grid.nx, grid.ny, grid.nz), default_tissue);

        Self {
            tissue_map,
            grid,
            cached_density: None,
            cached_sound_speed: None,
        }
    }

    /// Set tissue type for a region
    pub fn set_tissue_region(&mut self, region: &TissueRegion) -> KwaversResult<()> {
        region.validate()?;

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
                    if region.contains(x, y, z) {
                        self.tissue_map[[i, j, k]] = region.tissue_type;
                    }
                }
            }
        }

        // Clear caches when tissue map changes
        self.clear_caches();
        Ok(())
    }

    /// Clear cached values
    fn clear_caches(&mut self) {
        self.cached_density = None;
        self.cached_sound_speed = None;
    }

    /// Get tissue properties at an index
    fn get_tissue_properties(&self, i: usize, j: usize, k: usize) -> &'static TissueProperties {
        let tissue_type = &self.tissue_map[[i, j, k]];
        TISSUE_PROPERTIES
            .get(tissue_type)
            .expect("Tissue type should have properties")
    }
}

impl CoreMedium for HeterogeneousTissueMedium {
    fn density(&self, i: usize, j: usize, k: usize) -> f64 {
        self.get_tissue_properties(i, j, k)
            .density
            .max(MIN_PHYSICAL_DENSITY)
    }

    fn sound_speed(&self, i: usize, j: usize, k: usize) -> f64 {
        self.get_tissue_properties(i, j, k)
            .sound_speed
            .max(MIN_PHYSICAL_SOUND_SPEED)
    }

    fn absorption(&self, i: usize, j: usize, k: usize) -> f64 {
        self.get_tissue_properties(i, j, k).alpha_0
    }

    fn nonlinearity(&self, i: usize, j: usize, k: usize) -> f64 {
        self.get_tissue_properties(i, j, k).nonlinearity
    }

    fn max_sound_speed(&self) -> f64 {
        // Compute max across all tissue types present
        let mut max_speed: f64 = 0.0;
        for tissue_type in &self.tissue_map {
            if let Some(props) = TISSUE_PROPERTIES.get(tissue_type) {
                max_speed = max_speed.max(props.sound_speed);
            }
        }
        max_speed.max(MIN_PHYSICAL_SOUND_SPEED)
    }

    fn is_homogeneous(&self) -> bool {
        // Check if all elements are the same tissue type
        let first = &self.tissue_map[[0, 0, 0]];
        self.tissue_map.iter().all(|t| t == first)
    }

    fn validate(&self, grid: &Grid) -> KwaversResult<()> {
        if self.tissue_map.shape() != [grid.nx, grid.ny, grid.nz] {
            return Err(ValidationError::DimensionMismatch {
                expected: format!("[{}, {}, {}]", grid.nx, grid.ny, grid.nz),
                actual: format!("{:?}", self.tissue_map.shape()),
            }
            .into());
        }
        Ok(())
    }
}

use super::super::types::{LayerSpec, TissueType};
use super::super::utils::get_tissue_properties;
use crate::domain::grid::GridDimensions;
use crate::domain::medium::optical_map::{Layer, OpticalPropertyMap, OpticalPropertyMapBuilder};
use crate::domain::medium::properties::OpticalPropertyData;

/// Layered tissue phantom builder
///
/// Models stratified media with horizontal layers (e.g., skin/fat/muscle).
#[derive(Debug)]
pub struct LayeredTissuePhantomBuilder {
    pub(super) dimensions: Option<GridDimensions>,
    pub(super) layers: Vec<LayerSpec>,
    pub(super) wavelength_nm: f64,
}

impl LayeredTissuePhantomBuilder {
    /// Set grid dimensions
    #[must_use]
    pub fn dimensions(mut self, dims: GridDimensions) -> Self {
        self.dimensions = Some(dims);
        self
    }

    /// Set wavelength (nm)
    #[must_use]
    pub fn wavelength(mut self, wavelength_nm: f64) -> Self {
        self.wavelength_nm = wavelength_nm;
        self
    }

    /// Add skin epidermis layer
    #[must_use]
    pub fn add_skin_layer(mut self, z_min: f64, z_max: f64) -> Self {
        self.layers.push(LayerSpec {
            z_min,
            z_max,
            tissue_type: TissueType::SkinEpidermis,
        });
        self
    }

    /// Add dermis layer
    #[must_use]
    pub fn add_dermis_layer(mut self, z_min: f64, z_max: f64) -> Self {
        self.layers.push(LayerSpec {
            z_min,
            z_max,
            tissue_type: TissueType::SkinDermis,
        });
        self
    }

    /// Add fat layer
    #[must_use]
    pub fn add_fat_layer(mut self, z_min: f64, z_max: f64) -> Self {
        self.layers.push(LayerSpec {
            z_min,
            z_max,
            tissue_type: TissueType::Fat,
        });
        self
    }

    /// Add muscle layer
    #[must_use]
    pub fn add_muscle_layer(mut self, z_min: f64, z_max: f64) -> Self {
        self.layers.push(LayerSpec {
            z_min,
            z_max,
            tissue_type: TissueType::Muscle,
        });
        self
    }

    /// Add liver layer
    #[must_use]
    pub fn add_liver_layer(mut self, z_min: f64, z_max: f64) -> Self {
        self.layers.push(LayerSpec {
            z_min,
            z_max,
            tissue_type: TissueType::Liver,
        });
        self
    }

    /// Build phantom
    /// # Panics
    /// - Panics if `Dimensions must be set before building`.
    ///
    #[must_use]
    pub fn build(self) -> OpticalPropertyMap {
        let dims = self
            .dimensions
            .expect("Dimensions must be set before building");

        let mut builder = OpticalPropertyMapBuilder::new(dims);
        builder.set_background(OpticalPropertyData::soft_tissue());

        for layer_spec in &self.layers {
            let props = get_tissue_properties(layer_spec.tissue_type);
            builder.add_layer(Layer::new(layer_spec.z_min, layer_spec.z_max, props));
        }

        builder.build()
    }
}

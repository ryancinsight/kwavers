use super::super::types::TumorSpec;
use super::super::utils::compute_tumor_properties;
use kwavers_optics::chromophores::HemoglobinDatabase;
use kwavers_grid::GridDimensions;
use crate::medium::optical_map::{OpticalPropertyMap, OpticalPropertyMapBuilder, Region};
use crate::medium::properties::OpticalPropertyData;

/// Tumor detection phantom builder
#[derive(Debug)]
pub struct TumorDetectionPhantomBuilder {
    pub(super) dimensions: Option<GridDimensions>,
    pub(super) background: OpticalPropertyData,
    pub(super) tumors: Vec<TumorSpec>,
    pub(super) wavelength_nm: f64,
}

impl TumorDetectionPhantomBuilder {
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

    /// Set background tissue properties
    #[must_use]
    pub fn background(mut self, props: OpticalPropertyData) -> Self {
        self.background = props;
        self
    }

    /// Add tumor lesion
    #[must_use]
    pub fn add_tumor(mut self, center: [f64; 3], radius: f64, so2: f64) -> Self {
        self.tumors.push(TumorSpec {
            center,
            radius,
            so2,
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
        builder.set_background(self.background);

        let hb_db = HemoglobinDatabase::default();

        for tumor in &self.tumors {
            let props = compute_tumor_properties(&hb_db, self.wavelength_nm, tumor.so2);
            builder.add_region(Region::sphere(tumor.center, tumor.radius), props);
        }

        builder.build()
    }
}

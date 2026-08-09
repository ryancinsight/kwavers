use super::super::properties::compute_tumor_properties;
use super::super::types::TumorSpec;
use super::super::PhantomError;
use kwavers_grid::GridDimensions;
use kwavers_medium::optical_map::{OpticalPropertyMap, OpticalPropertyMapBuilder, Region};
use kwavers_medium::properties::OpticalPropertyData;

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

    /// Build phantom.
    ///
    /// # Errors
    ///
    /// Returns [`PhantomError::MissingDimensions`] when dimensions were not
    /// configured, or propagates optical-property validation errors from
    /// Hyperion and the medium contract.
    pub fn build(self) -> Result<OpticalPropertyMap, PhantomError> {
        let dims = self.dimensions.ok_or(PhantomError::MissingDimensions)?;

        let mut builder = OpticalPropertyMapBuilder::new(dims);
        builder.set_background(self.background);

        for tumor in &self.tumors {
            let props = compute_tumor_properties(self.wavelength_nm, tumor.so2)?;
            builder.add_region(Region::sphere(tumor.center, tumor.radius), props);
        }

        Ok(builder.build())
    }
}

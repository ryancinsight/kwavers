use super::super::types::VesselGeometry;
use super::super::utils::compute_blood_properties;
use kwavers_grid::GridDimensions;
use kwavers_medium::optical_map::{OpticalPropertyMap, OpticalPropertyMapBuilder, Region};
use kwavers_medium::properties::OpticalPropertyData;
use kwavers_optics::chromophores::HemoglobinDatabase;

/// Vascular phantom builder
#[derive(Debug)]
pub struct VascularPhantomBuilder {
    pub(super) dimensions: Option<GridDimensions>,
    pub(super) background: OpticalPropertyData,
    pub(super) vessels: Vec<VesselGeometry>,
    pub(super) wavelength_nm: f64,
}

impl VascularPhantomBuilder {
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

    /// Add vessel segment (cylinder)
    #[must_use]
    pub fn add_vessel(mut self, start: [f64; 3], end: [f64; 3], radius: f64, so2: f64) -> Self {
        self.vessels.push(VesselGeometry {
            start,
            end,
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

        for vessel in &self.vessels {
            let props = compute_blood_properties(&hb_db, self.wavelength_nm, vessel.so2);
            builder.add_region(
                Region::cylinder(vessel.start, vessel.end, vessel.radius),
                props,
            );
        }

        builder.build()
    }
}

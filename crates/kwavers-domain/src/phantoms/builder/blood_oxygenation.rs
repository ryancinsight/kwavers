use super::super::types::{PhantomVesselType, TumorSpec, VesselSpec};
use super::super::utils::{compute_blood_properties, compute_tumor_properties};
use kwavers_optics::chromophores::HemoglobinDatabase;
use kwavers_grid::GridDimensions;
use crate::medium::optical_map::{OpticalPropertyMap, OpticalPropertyMapBuilder, Region};
use crate::medium::properties::OpticalPropertyData;

/// Blood oxygenation phantom builder
///
/// Models arterial/venous vessels and tumors with specified oxygenation levels.
#[derive(Debug)]
pub struct BloodOxygenationPhantomBuilder {
    pub(super) dimensions: Option<GridDimensions>,
    pub(super) background: OpticalPropertyData,
    pub(super) vessels: Vec<VesselSpec>,
    pub(super) tumors: Vec<TumorSpec>,
    pub(super) wavelength_nm: f64,
}

impl BloodOxygenationPhantomBuilder {
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

    /// Add arterial vessel (high oxygenation: sO₂ ≈ 0.95-0.99)
    #[must_use]
    pub fn add_artery(mut self, center: [f64; 3], radius: f64, so2: f64) -> Self {
        self.vessels.push(VesselSpec {
            center,
            radius,
            so2,
            vessel_type: PhantomVesselType::Artery,
        });
        self
    }

    /// Add venous vessel (lower oxygenation: sO₂ ≈ 0.60-0.75)
    #[must_use]
    pub fn add_vein(mut self, center: [f64; 3], radius: f64, so2: f64) -> Self {
        self.vessels.push(VesselSpec {
            center,
            radius,
            so2,
            vessel_type: PhantomVesselType::Vein,
        });
        self
    }

    /// Add capillary (intermediate oxygenation: sO₂ ≈ 0.70-0.85)
    #[must_use]
    pub fn add_capillary(mut self, center: [f64; 3], radius: f64, so2: f64) -> Self {
        self.vessels.push(VesselSpec {
            center,
            radius,
            so2,
            vessel_type: PhantomVesselType::Capillary,
        });
        self
    }

    /// Add tumor region (typically hypoxic: sO₂ ≈ 0.50-0.70)
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

        for vessel in &self.vessels {
            let props = compute_blood_properties(&hb_db, self.wavelength_nm, vessel.so2);
            builder.add_region(Region::sphere(vessel.center, vessel.radius), props);
        }

        for tumor in &self.tumors {
            let props = compute_tumor_properties(&hb_db, self.wavelength_nm, tumor.so2);
            builder.add_region(Region::sphere(tumor.center, tumor.radius), props);
        }

        builder.build()
    }
}

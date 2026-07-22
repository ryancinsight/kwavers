use super::OpticalPropertyData;
use kwavers_core::constants::optical::{
    REFRACTIVE_INDEX_SOFT_TISSUE, REFRACTIVE_INDEX_SOFT_TISSUE_NIR, REFRACTIVE_INDEX_WATER,
};

impl OpticalPropertyData {
    /// Construct with validation of physical constraints
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(
        absorption_coefficient: f64,
        scattering_coefficient: f64,
        anisotropy: f64,
        refractive_index: f64,
    ) -> Result<Self, String> {
        Self::from_si(
            absorption_coefficient,
            scattering_coefficient,
            anisotropy,
            refractive_index,
        )
    }

    /// Vacuum optical properties.
    #[must_use]
    pub fn vacuum() -> Self {
        Self::from_si(0.0, 0.0, 0.0, 1.0)
            .expect("invariant: vacuum preset satisfies optical constraints")
    }

    /// Water optical properties (visible spectrum, ~550 nm)
    #[must_use]
    pub fn water() -> Self {
        Self::from_si(0.01, 0.001, 0.0, REFRACTIVE_INDEX_WATER)
            .expect("invariant: water preset satisfies optical constraints")
    }

    /// Soft tissue optical properties (generic, ~650 nm)
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self::from_si(0.5, 100.0, 0.9, REFRACTIVE_INDEX_SOFT_TISSUE)
            .expect("invariant: soft-tissue preset satisfies optical constraints")
    }

    /// Blood optical properties (oxygenated, ~650 nm)
    #[must_use]
    pub fn blood_oxygenated() -> Self {
        Self::from_si(50.0, 200.0, 0.95, REFRACTIVE_INDEX_SOFT_TISSUE)
            .expect("invariant: oxygenated-blood preset satisfies optical constraints")
    }

    /// Blood optical properties (deoxygenated, ~650 nm)
    #[must_use]
    pub fn blood_deoxygenated() -> Self {
        Self::from_si(80.0, 200.0, 0.95, REFRACTIVE_INDEX_SOFT_TISSUE)
            .expect("invariant: deoxygenated-blood preset satisfies optical constraints")
    }

    /// Tumor tissue optical properties (hypervascular, ~650 nm)
    #[must_use]
    pub fn tumor() -> Self {
        Self::from_si(10.0, 120.0, 0.85, REFRACTIVE_INDEX_SOFT_TISSUE)
            .expect("invariant: tumor preset satisfies optical constraints")
    }

    /// Brain tissue optical properties (gray matter, ~650 nm)
    #[must_use]
    pub fn brain_gray_matter() -> Self {
        Self::from_si(0.8, 150.0, 0.9, REFRACTIVE_INDEX_SOFT_TISSUE_NIR)
            .expect("invariant: gray-matter preset satisfies optical constraints")
    }

    /// Brain tissue optical properties (white matter, ~650 nm)
    #[must_use]
    pub fn brain_white_matter() -> Self {
        Self::from_si(1.0, 250.0, 0.92, REFRACTIVE_INDEX_SOFT_TISSUE_NIR)
            .expect("invariant: white-matter preset satisfies optical constraints")
    }

    /// Liver tissue optical properties (~650 nm)
    #[must_use]
    pub fn liver() -> Self {
        Self::from_si(2.0, 120.0, 0.88, 1.39)
            .expect("invariant: liver preset satisfies optical constraints")
    }

    /// Muscle tissue optical properties (~650 nm)
    #[must_use]
    pub fn muscle() -> Self {
        Self::from_si(0.8, 100.0, 0.85, 1.37)
            .expect("invariant: muscle preset satisfies optical constraints")
    }

    /// Skin (epidermis) optical properties (~650 nm)
    #[must_use]
    pub fn skin_epidermis() -> Self {
        Self::from_si(5.0, 300.0, 0.8, REFRACTIVE_INDEX_SOFT_TISSUE)
            .expect("invariant: epidermis preset satisfies optical constraints")
    }

    /// Skin (dermis) optical properties (~650 nm)
    #[must_use]
    pub fn skin_dermis() -> Self {
        Self::from_si(1.0, 200.0, 0.85, REFRACTIVE_INDEX_SOFT_TISSUE)
            .expect("invariant: dermis preset satisfies optical constraints")
    }

    /// Bone (cortical) optical properties (~650 nm)
    #[must_use]
    pub fn bone_cortical() -> Self {
        Self::from_si(5.0, 500.0, 0.9, 1.55)
            .expect("invariant: cortical-bone preset satisfies optical constraints")
    }

    /// Fat tissue optical properties (~650 nm)
    #[must_use]
    pub fn fat() -> Self {
        Self::from_si(0.3, 100.0, 0.9, 1.46)
            .expect("invariant: fat preset satisfies optical constraints")
    }
}

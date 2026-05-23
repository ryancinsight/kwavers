use super::OpticalPropertyData;
use crate::core::constants::optical::REFRACTIVE_INDEX_SOFT_TISSUE;

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
        if absorption_coefficient < 0.0 {
            return Err(format!(
                "Absorption coefficient must be non-negative, got {}",
                absorption_coefficient
            ));
        }
        if scattering_coefficient < 0.0 {
            return Err(format!(
                "Scattering coefficient must be non-negative, got {}",
                scattering_coefficient
            ));
        }
        if !(-1.0..=1.0).contains(&anisotropy) {
            return Err(format!(
                "Anisotropy factor must be in range [-1, 1], got {}",
                anisotropy
            ));
        }
        if refractive_index < 1.0 {
            return Err(format!(
                "Refractive index must be ≥ 1.0 (vacuum limit), got {}",
                refractive_index
            ));
        }
        Ok(Self {
            absorption_coefficient,
            scattering_coefficient,
            anisotropy,
            refractive_index,
        })
    }

    /// Water optical properties (visible spectrum, ~550 nm)
    #[must_use]
    pub fn water() -> Self {
        Self {
            absorption_coefficient: 0.01,
            scattering_coefficient: 0.001,
            anisotropy: 0.0,
            refractive_index: 1.33,
        }
    }

    /// Soft tissue optical properties (generic, ~650 nm)
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self {
            absorption_coefficient: 0.5,
            scattering_coefficient: 100.0,
            anisotropy: 0.9,
            refractive_index: REFRACTIVE_INDEX_SOFT_TISSUE,
        }
    }

    /// Blood optical properties (oxygenated, ~650 nm)
    #[must_use]
    pub fn blood_oxygenated() -> Self {
        Self {
            absorption_coefficient: 50.0,
            scattering_coefficient: 200.0,
            anisotropy: 0.95,
            refractive_index: REFRACTIVE_INDEX_SOFT_TISSUE,
        }
    }

    /// Blood optical properties (deoxygenated, ~650 nm)
    #[must_use]
    pub fn blood_deoxygenated() -> Self {
        Self {
            absorption_coefficient: 80.0,
            scattering_coefficient: 200.0,
            anisotropy: 0.95,
            refractive_index: REFRACTIVE_INDEX_SOFT_TISSUE,
        }
    }

    /// Tumor tissue optical properties (hypervascular, ~650 nm)
    #[must_use]
    pub fn tumor() -> Self {
        Self {
            absorption_coefficient: 10.0,
            scattering_coefficient: 120.0,
            anisotropy: 0.85,
            refractive_index: REFRACTIVE_INDEX_SOFT_TISSUE,
        }
    }

    /// Brain tissue optical properties (gray matter, ~650 nm)
    #[must_use]
    pub fn brain_gray_matter() -> Self {
        Self {
            absorption_coefficient: 0.8,
            scattering_coefficient: 150.0,
            anisotropy: 0.9,
            refractive_index: 1.38,
        }
    }

    /// Brain tissue optical properties (white matter, ~650 nm)
    #[must_use]
    pub fn brain_white_matter() -> Self {
        Self {
            absorption_coefficient: 1.0,
            scattering_coefficient: 250.0,
            anisotropy: 0.92,
            refractive_index: 1.38,
        }
    }

    /// Liver tissue optical properties (~650 nm)
    #[must_use]
    pub fn liver() -> Self {
        Self {
            absorption_coefficient: 2.0,
            scattering_coefficient: 120.0,
            anisotropy: 0.88,
            refractive_index: 1.39,
        }
    }

    /// Muscle tissue optical properties (~650 nm)
    #[must_use]
    pub fn muscle() -> Self {
        Self {
            absorption_coefficient: 0.8,
            scattering_coefficient: 100.0,
            anisotropy: 0.85,
            refractive_index: 1.37,
        }
    }

    /// Skin (epidermis) optical properties (~650 nm)
    #[must_use]
    pub fn skin_epidermis() -> Self {
        Self {
            absorption_coefficient: 5.0,
            scattering_coefficient: 300.0,
            anisotropy: 0.8,
            refractive_index: REFRACTIVE_INDEX_SOFT_TISSUE,
        }
    }

    /// Skin (dermis) optical properties (~650 nm)
    #[must_use]
    pub fn skin_dermis() -> Self {
        Self {
            absorption_coefficient: 1.0,
            scattering_coefficient: 200.0,
            anisotropy: 0.85,
            refractive_index: REFRACTIVE_INDEX_SOFT_TISSUE,
        }
    }

    /// Bone (cortical) optical properties (~650 nm)
    #[must_use]
    pub fn bone_cortical() -> Self {
        Self {
            absorption_coefficient: 5.0,
            scattering_coefficient: 500.0,
            anisotropy: 0.9,
            refractive_index: 1.55,
        }
    }

    /// Fat tissue optical properties (~650 nm)
    #[must_use]
    pub fn fat() -> Self {
        Self {
            absorption_coefficient: 0.3,
            scattering_coefficient: 100.0,
            anisotropy: 0.9,
            refractive_index: 1.46,
        }
    }
}

//! Optical properties for light diffusion calculations

use crate::domain::medium::properties::OpticalPropertyData;

/// Physics-layer optical properties bridge for photon diffusion calculations
///
/// This struct composes the domain SSOT `OpticalPropertyData` and provides
/// diffusion-specific accessor methods. The stored `reduced_scattering_coefficient`
/// is μₛ' = μₛ(1-g), pre-computed from the domain data.
#[derive(Debug, Clone, Copy)]
pub struct OpticalProperties {
    /// Absorption coefficient μₐ [m⁻¹] (from domain SSOT)
    pub absorption_coefficient: f64,
    /// Reduced scattering coefficient μₛ' [m⁻¹]
    /// Pre-computed as μₛ' = μₛ(1-g) where g is the anisotropy factor
    pub reduced_scattering_coefficient: f64,
    /// Refractive index n (dimensionless) (from domain SSOT)
    pub refractive_index: f64,
}

impl OpticalProperties {
    /// Create from canonical domain SSOT property data
    ///
    /// Automatically computes reduced scattering coefficient μₛ' = μₛ(1-g)
    #[must_use]
    pub fn from_domain(props: OpticalPropertyData) -> Self {
        Self {
            absorption_coefficient: props.absorption_coefficient,
            reduced_scattering_coefficient: props.reduced_scattering(),
            refractive_index: props.refractive_index,
        }
    }

    /// Create optical properties for a typical biological tissue
    #[must_use]
    pub fn biological_tissue() -> Self {
        Self::from_domain(OpticalPropertyData::soft_tissue())
    }

    /// Create optical properties for water
    #[must_use]
    pub fn water() -> Self {
        Self::from_domain(OpticalPropertyData::water())
    }

    /// Calculate diffusion coefficient from optical properties
    ///
    /// In the diffusion approximation: D = 1/(3(μₐ + μₛ'))
    /// where μₐ is absorption coefficient, μₛ' is reduced scattering coefficient
    #[must_use]
    pub fn diffusion_coefficient(&self) -> f64 {
        1.0 / (3.0 * (self.absorption_coefficient + self.reduced_scattering_coefficient))
    }

    /// Calculate the transport coefficient μ_tr = μₐ + μₛ'
    #[must_use]
    pub fn transport_coefficient(&self) -> f64 {
        self.absorption_coefficient + self.reduced_scattering_coefficient
    }

    /// Calculate albedo ω = μₛ' / μ_tr (single scattering albedo)
    #[must_use]
    pub fn single_scatter_albedo(&self) -> f64 {
        let mu_tr = self.transport_coefficient();
        if mu_tr > 0.0 {
            self.reduced_scattering_coefficient / mu_tr
        } else {
            0.0
        }
    }

    /// Check validity of diffusion approximation
    ///
    /// The diffusion approximation is valid when:
    /// 1. Reduced scattering dominates absorption: μₛ' ≫ μₐ
    /// 2. Optical depth is large enough for diffusion to develop
    #[must_use]
    pub fn diffusion_approximation_valid(&self) -> bool {
        // Require scattering to be at least 10x absorption for good diffusion approximation
        self.reduced_scattering_coefficient >= 10.0 * self.absorption_coefficient
    }
}

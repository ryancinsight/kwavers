use aequitas::systems::si::dimensions;
use kwavers_core::units::DimensionedField;
use kwavers_grid::GridDimensions;
use std::fmt;

/// Absorbed energy density samples, in `J/m3`.
pub type AbsorbedEnergyField<'a> = DimensionedField<&'a [f64], dimensions::EnergyPerVolume>;

/// Energy-fluence samples, in `J/m2`.
pub type EnergyFluenceField<'a> = DimensionedField<&'a [f64], dimensions::EnergyPerArea>;

/// Monte Carlo simulation result
#[derive(Debug)]
pub struct MCResult {
    pub(crate) dimensions: GridDimensions,
    pub(crate) absorbed_energy: Vec<f64>,
    pub(crate) fluence: Vec<f64>,
    pub(crate) num_photons: usize,
    /// Diffuse reflectance Rd = reflected photon weight / num_photons.
    ///
    /// A photon contributes to Rd when it exits the domain through the
    /// source surface (z < 0).  This matches the MCML convention
    /// (Wang et al. 1995 §2.7).
    pub(crate) diffuse_reflectance: f64,
}

impl MCResult {
    /// Absorbed energy density per voxel.
    ///
    /// The dimension tag is what distinguishes this from [`Self::fluence`];
    /// both were bare `&[f64]` with the unit stated only in this comment.
    #[must_use]
    pub fn absorbed_energy(&self) -> AbsorbedEnergyField<'_> {
        AbsorbedEnergyField::from_base(&self.absorbed_energy)
    }

    /// Energy fluence per voxel.
    #[must_use]
    pub fn fluence(&self) -> EnergyFluenceField<'_> {
        EnergyFluenceField::from_base(&self.fluence)
    }

    /// Get total absorbed energy (J)
    #[must_use]
    pub fn total_absorbed_energy(&self) -> f64 {
        self.absorbed_energy.iter().sum()
    }

    /// Get fluence normalized by number of photons
    #[must_use]
    pub fn normalized_fluence(&self) -> Vec<f64> {
        let norm = 1.0 / self.num_photons as f64;
        self.fluence.iter().map(|&f| f * norm).collect()
    }

    /// Get dimensions
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn dimensions(&self) -> GridDimensions {
        self.dimensions
    }

    /// Diffuse reflectance Rd (dimensionless, in [0, 1]).
    ///
    /// Fraction of incident photon weight that exits the domain from the
    /// source surface.  Matches MCML definition (Wang et al. 1995 §2.7).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn diffuse_reflectance(&self) -> f64 {
        self.diffuse_reflectance
    }
}

impl fmt::Display for MCResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_absorbed = self.total_absorbed_energy();
        let mean_fluence = self.fluence.iter().sum::<f64>() / self.fluence.len() as f64;
        write!(
            f,
            "MCResult(photons={}, absorbed={:.3e} J, mean_fluence={:.3e} J/m²)",
            self.num_photons, total_absorbed, mean_fluence
        )
    }
}

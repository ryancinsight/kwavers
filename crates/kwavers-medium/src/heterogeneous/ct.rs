//! Build a complete, tissue-varying simulation medium from a CT volume.
//!
//! A full forward simulation needs the power-law absorption **exponent** y and
//! the **nonlinearity** B/A to vary by tissue as well as density, sound speed,
//! and absorption. This builder produces a [`HeterogeneousMedium`] whose every
//! acoustic field — density, sound speed,
//! absorption prefactor α₀, exponent y, and B/A — is mapped per-voxel from
//! Hounsfield units via `HuAcousticModel`, with the non-acoustic fields
//! (thermal, optical, bubble, elastic, viscous) broadcast from a homogeneous
//! `background` medium. See book Ch4 §4.5.
//!
//! ## References
//! - Schneider U et al. (1996). *Phys. Med. Biol.* 41(1), 111–124. (ρ, c)
//! - Duck FA (1990). *Physical Properties of Tissue.* (y, B/A)
//! - Connor CW, Hynynen K (2002). *Phys. Med. Biol.* 47(12), 2213–2231. (skull y)

use crate::heterogeneous::HeterogeneousMedium;
use crate::HomogeneousMedium;
use kwavers_core::constants::hu_mapping::HuAcousticModel;
use kwavers_core::error::{KwaversError, KwaversResult, MediumError};
use kwavers_grid::Grid;
use leto::Array3;

/// Builder turning a Hounsfield-unit CT volume into a fully-specified,
/// tissue-varying [`HeterogeneousMedium`].
///
/// The HU→property calibration (`HuAcousticModel`) and the background medium
/// are both overridable; defaults are the Schneider (1996) calibration and a
/// water background.
#[derive(Debug)]
pub struct CtMediumBuilder<'a> {
    ct: &'a Array3<f64>,
    grid: &'a Grid,
    model: HuAcousticModel,
    background: HomogeneousMedium,
}

impl<'a> CtMediumBuilder<'a> {
    /// Start a builder over a CT volume (standard Hounsfield units) and the
    /// simulation grid. Defaults: Schneider calibration, water background.
    #[must_use]
    pub fn new(ct: &'a Array3<f64>, grid: &'a Grid) -> Self {
        Self {
            ct,
            grid,
            model: HuAcousticModel::default(),
            background: HomogeneousMedium::water(grid),
        }
    }

    /// Override the HU→acoustic-property calibration (e.g. a scanner-specific fit).
    #[must_use]
    pub fn with_model(mut self, model: HuAcousticModel) -> Self {
        self.model = model;
        self
    }

    /// Override the background medium whose non-acoustic fields (thermal,
    /// optical, bubble, elastic, viscous) are broadcast across the grid.
    #[must_use]
    pub fn with_background(mut self, background: HomogeneousMedium) -> Self {
        self.background = background;
        self
    }

    /// Assemble the medium. Every acoustic field is mapped per-voxel from HU;
    /// the result is validated against the grid before return.
    /// # Errors
    /// - Returns [`Err`] if the CT volume shape does not match the grid, or if
    ///   the assembled medium fails [`Medium`](crate::Medium) validation.
    pub fn build(self) -> KwaversResult<HeterogeneousMedium> {
        use crate::CoreMedium;

        let dims = self.grid.dimensions();
        if self.ct.shape() != [dims.0, dims.1, dims.2] {
            return Err(KwaversError::DimensionMismatch(format!(
                "CT volume shape {:?} does not match grid {:?}",
                self.ct.shape(),
                dims
            )));
        }

        let model = &self.model;
        let mut medium = HeterogeneousMedium::from_homogeneous(&self.background, self.grid)
            .map_err(|error| MediumError::InitializationFailed {
                reason: format!("invalid background optical properties: {error}"),
            })?;

        // Overwrite every acoustic field with the per-voxel HU mapping. These are
        // exactly the fields the solver's acoustic traits read: `absorption` is
        // the α₀ prefactor (alpha_coefficient), `alpha_power` the exponent y, and
        // `nonlinearity`/`b_a` the B/A parameter.
        medium.density = self.ct.mapv(|hu| model.density(hu));
        medium.sound_speed = self.ct.mapv(|hu| model.sound_speed(hu));
        medium.absorption = self.ct.mapv(|hu| model.absorption(hu));
        medium.alpha0 = medium.absorption.clone();
        medium.alpha_power = self.ct.mapv(|hu| model.power_law_exponent(hu));
        medium.nonlinearity = self.ct.mapv(|hu| model.nonlinearity(hu));
        medium.b_a = medium.nonlinearity.clone();

        medium.validate(self.grid)?;
        Ok(medium)
    }
}

#[cfg(test)]
mod tests {
    use super::CtMediumBuilder;
    use crate::{AcousticProperties, CoreMedium};
    use kwavers_core::constants::hu_mapping::HuAcousticModel;
    use kwavers_core::error::KwaversError;
    use kwavers_grid::Grid;
    use leto::Array3;

    fn grid_4x4x4() -> Grid {
        Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).expect("valid grid")
    }

    // A CT volume split into a soft-tissue half (HU=0) and a bone half (HU=1000)
    // must produce DISTINCT per-voxel ρ, c, α₀, y, and B/A — the whole point of a
    // complete tissue-varying medium.
    #[test]
    fn ct_medium_resolves_all_acoustic_fields_per_voxel() {
        let grid = grid_4x4x4();
        let mut ct = Array3::<f64>::zeros((4, 4, 4));
        for i in 2..4 {
            for j in 0..4 {
                for k in 0..4 {
                    ct[[i, j, k]] = 1000.0; // bone slab
                }
            }
        }
        let medium = CtMediumBuilder::new(&ct, &grid).build().unwrap();
        let m = HuAcousticModel::default();

        // Soft-tissue voxel (HU=0) vs bone voxel (HU=1000), point access.
        assert_eq!(medium.density(0, 0, 0), m.density(0.0));
        assert_eq!(medium.density(3, 0, 0), m.density(1000.0));
        assert!(medium.sound_speed(3, 0, 0) > medium.sound_speed(0, 0, 0));
        assert!(medium.absorption(3, 0, 0) > medium.absorption(0, 0, 0));
        assert!(medium.nonlinearity(3, 0, 0) > medium.nonlinearity(0, 0, 0));
        assert!(!medium.is_homogeneous());

        // Per-voxel power-law exponent y is exposed through the Medium trait:
        // soft tissue ≈ 1.1, bone ≈ 1.0. The bone half maps to the bone exponent.
        let y_soft = medium.alpha_power(0.0, 0.0, 0.0, &grid);
        let y_bone = medium.alpha_power(3.0e-3, 0.0, 0.0, &grid);
        assert_eq!(y_soft, m.exponent_soft);
        assert_eq!(y_bone, m.exponent_bone);
    }

    // Mismatched CT/grid shapes are rejected, not silently truncated.
    #[test]
    fn ct_grid_shape_mismatch_is_rejected() {
        let grid = grid_4x4x4();
        let ct = Array3::<f64>::zeros((4, 4, 3));
        let error = CtMediumBuilder::new(&ct, &grid)
            .build()
            .expect_err("mismatched CT and grid dimensions must fail");
        match error {
            KwaversError::DimensionMismatch(message) => assert_eq!(
                message,
                "CT volume shape [4, 4, 3] does not match grid (4, 4, 4)"
            ),
            other => panic!("expected dimension mismatch, got {other}"),
        }
    }

    // Frequency-resolved absorption uses the per-voxel (α₀, y): bone absorbs far
    // more than soft tissue at the same frequency.
    #[test]
    fn absorption_coefficient_is_frequency_and_tissue_dependent() {
        let grid = grid_4x4x4();
        let mut ct = Array3::<f64>::zeros((4, 4, 4));
        ct[[3, 3, 3]] = 1000.0;
        let medium = CtMediumBuilder::new(&ct, &grid).build().unwrap();
        let f = 1.0e6;
        let a_soft = medium.absorption_coefficient(0.0, 0.0, 0.0, &grid, f);
        let a_bone = medium.absorption_coefficient(3.0e-3, 3.0e-3, 3.0e-3, &grid, f);
        assert!(
            a_bone > a_soft && a_soft >= 0.0,
            "soft={a_soft} bone={a_bone}"
        );
    }
}

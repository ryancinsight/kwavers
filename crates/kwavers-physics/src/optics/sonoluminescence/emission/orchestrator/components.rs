//! Dimensioned sonoluminescence emission contributions.

use aequitas::systems::si::quantities::VolumetricPowerDensity;

/// Dimensioned emission contributions for one spatial cell.
///
/// Cherenkov threshold yield is intentionally excluded because the current
/// model exposes it in arbitrary spectral units rather than watts per cubic
/// metre. The private fields keep callers on the typed component boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EmissionComponents {
    blackbody: VolumetricPowerDensity<f64>,
    bremsstrahlung: VolumetricPowerDensity<f64>,
}

const _: () = assert!(
    core::mem::size_of::<EmissionComponents>()
        == 2 * core::mem::size_of::<VolumetricPowerDensity<f64>>()
);

impl EmissionComponents {
    /// Construct the dimensioned contributions for one cell.
    #[must_use]
    pub const fn new(
        blackbody: VolumetricPowerDensity<f64>,
        bremsstrahlung: VolumetricPowerDensity<f64>,
    ) -> Self {
        Self {
            blackbody,
            bremsstrahlung,
        }
    }

    /// Return the blackbody contribution.
    #[must_use]
    pub const fn blackbody(self) -> VolumetricPowerDensity<f64> {
        self.blackbody
    }

    /// Return the bremsstrahlung contribution.
    #[must_use]
    pub const fn bremsstrahlung(self) -> VolumetricPowerDensity<f64> {
        self.bremsstrahlung
    }

    /// Return the dimensioned total.
    #[must_use]
    pub const fn total(self) -> VolumetricPowerDensity<f64> {
        VolumetricPowerDensity::from_base(
            *self.blackbody.as_base() + *self.bremsstrahlung.as_base(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_preserves_dimensioned_components() {
        let components = EmissionComponents::new(
            VolumetricPowerDensity::from_base(2.5),
            VolumetricPowerDensity::from_base(7.5),
        );

        assert_eq!(components.total().into_base(), 10.0);
        assert_eq!(components.blackbody().into_base(), 2.5);
        assert_eq!(components.bremsstrahlung().into_base(), 7.5);
    }
}

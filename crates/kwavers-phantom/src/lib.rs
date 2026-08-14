//! Clinical Phantom Builders
//!
//! Provides realistic tissue phantom constructors for photoacoustic and optical imaging
//! validation, protocol development, and algorithm testing.

#![cfg_attr(test, expect(clippy::unwrap_used, reason = "ratchet KWAVERS-UNWRAP-1"))]

pub mod builder;
mod error;
pub mod presets;
pub mod properties;
pub mod scatterers;
pub mod shepp_logan;
pub mod types;

pub use builder::{
    BloodOxygenationPhantomBuilder, LayeredTissuePhantomBuilder, PhantomBuilder,
    TumorDetectionPhantomBuilder, VascularPhantomBuilder,
};
pub use error::PhantomError;
pub use presets::ClinicalPhantoms;
pub use scatterers::{PointScatterer, RfSynthesisConfig, ScattererCloud, TransmitWavefront};
pub use shepp_logan::{Ellipse, SheppLogan, SheppLoganVariant};
pub use types::{LayerSpec, PhantomTissueType, PhantomType, TumorSpec, VesselSpec};

#[cfg(test)]
mod tests {
    use super::*;
    use hyperion::TransportError;
    use kwavers_grid::GridDimensions;
    use kwavers_medium::properties::OpticalPropertyData;

    #[test]
    fn test_blood_oxygenation_phantom() -> Result<(), PhantomError> {
        let dims = GridDimensions::new(30, 30, 30, 0.001, 0.001, 0.001);
        let phantom = PhantomBuilder::blood_oxygenation()
            .dimensions(dims)
            .wavelength(800.0)
            .add_artery([0.015, 0.015, 0.015], 0.002, 0.98)
            .add_vein([0.015, 0.015, 0.020], 0.003, 0.65)
            .build()?;

        assert_eq!(phantom.properties().len(), 27000);
        // Vessels (oxy/deoxy hemoglobin) raise absorption above the background, so
        // the map must span a non-trivial range. Computed inline to keep phantom
        // fixtures free of any physics-layer dependency.
        let max = phantom
            .properties()
            .iter()
            .map(OpticalPropertyData::absorption_coefficient)
            .fold(f64::MIN, f64::max);
        let min = phantom
            .properties()
            .iter()
            .map(OpticalPropertyData::absorption_coefficient)
            .fold(f64::MAX, f64::min);
        assert!(max > min);
        Ok(())
    }

    #[test]
    fn test_layered_tissue_phantom() {
        let dims = GridDimensions::new(20, 20, 40, 0.001, 0.001, 0.001);
        let phantom = PhantomBuilder::layered_tissue()
            .dimensions(dims)
            .wavelength(800.0)
            .add_skin_layer(0.0, 0.002)
            .add_fat_layer(0.002, 0.010)
            .add_muscle_layer(0.010, 0.040)
            .build();

        assert_eq!(phantom.properties().len(), 16000);

        // Check first layer is skin (high absorption)
        let skin_props = phantom.get_properties(10, 10, 1).unwrap();
        assert!(skin_props.absorption_coefficient() > 3.0);

        // Check second layer is fat (low absorption)
        let fat_props = phantom.get_properties(10, 10, 6).unwrap();
        assert!(fat_props.absorption_coefficient() < 1.0);
    }

    #[test]
    fn test_tumor_detection_phantom() -> Result<(), PhantomError> {
        let dims = GridDimensions::new(25, 25, 25, 0.001, 0.001, 0.001);
        let phantom = PhantomBuilder::tumor_detection()
            .dimensions(dims)
            .wavelength(800.0)
            .background(OpticalPropertyData::soft_tissue())
            .add_tumor([0.0125, 0.0125, 0.0125], 0.003, 0.60)
            .build()?;

        assert_eq!(phantom.properties().len(), 15625);
        Ok(())
    }

    #[test]
    fn test_vascular_phantom() -> Result<(), PhantomError> {
        let dims = GridDimensions::new(20, 20, 30, 0.001, 0.001, 0.001);
        let phantom = PhantomBuilder::vascular()
            .dimensions(dims)
            .wavelength(800.0)
            .add_vessel([0.01, 0.01, 0.0], [0.01, 0.01, 0.03], 0.001, 0.95)
            .build()?;

        assert_eq!(phantom.properties().len(), 12000);
        Ok(())
    }

    #[test]
    fn test_standard_blood_oxygenation() -> Result<(), PhantomError> {
        let dims = GridDimensions::new(30, 30, 30, 0.001, 0.001, 0.001);
        let phantom = ClinicalPhantoms::standard_blood_oxygenation(dims)?;

        assert_eq!(phantom.properties().len(), 27000);
        // Non-trivial mean absorption (inlined to keep fixtures physics-free).
        let mean = phantom
            .properties()
            .iter()
            .map(OpticalPropertyData::absorption_coefficient)
            .sum::<f64>()
            / phantom.properties().len() as f64;
        assert!(mean > 0.0);
        Ok(())
    }

    #[test]
    fn test_skin_tissue_phantom() {
        let dims = GridDimensions::new(20, 20, 40, 0.001, 0.001, 0.001);
        let phantom = ClinicalPhantoms::skin_tissue(dims);

        assert_eq!(phantom.properties().len(), 16000);
    }

    #[test]
    fn test_breast_tumor_phantom() -> Result<(), PhantomError> {
        let dims = GridDimensions::new(30, 30, 30, 0.001, 0.001, 0.001);
        let phantom = ClinicalPhantoms::breast_tumor(dims, [0.015, 0.015, 0.015])?;

        assert_eq!(phantom.properties().len(), 27000);
        Ok(())
    }

    #[test]
    fn test_vascular_network_phantom() -> Result<(), PhantomError> {
        let dims = GridDimensions::new(25, 25, 30, 0.001, 0.001, 0.001);
        let phantom = ClinicalPhantoms::vascular_network(dims)?;

        assert_eq!(phantom.properties().len(), 18750);
        Ok(())
    }

    fn assert_wavelength_error<T>(result: Result<T, PhantomError>, wavelength: f64) {
        match result {
            Err(PhantomError::Hyperion(TransportError::WavelengthOutOfRange {
                value,
                minimum,
                maximum,
            })) => {
                assert_eq!(value, wavelength);
                assert!(wavelength < minimum || wavelength > maximum);
            }
            Err(error) => panic!("expected a wavelength error, got {error:?}"),
            Ok(_) => panic!("expected an invalid wavelength to be rejected"),
        }
    }

    #[test]
    fn invalid_wavelengths_are_rejected_by_property_helpers() {
        for wavelength in [440.0, 1_001.0] {
            assert_wavelength_error(
                properties::compute_blood_properties(wavelength, 0.7),
                wavelength,
            );
            assert_wavelength_error(
                properties::compute_tumor_properties(wavelength, 0.7),
                wavelength,
            );
        }
    }

    #[test]
    fn invalid_wavelengths_propagate_through_phantom_builders() {
        let dims = GridDimensions::new(1, 1, 1, 0.001, 0.001, 0.001);

        for wavelength in [440.0, 1_001.0] {
            assert_wavelength_error(
                PhantomBuilder::blood_oxygenation()
                    .dimensions(dims)
                    .wavelength(wavelength)
                    .add_artery([0.0, 0.0, 0.0], 0.0001, 0.7)
                    .build(),
                wavelength,
            );
            assert_wavelength_error(
                PhantomBuilder::blood_oxygenation()
                    .dimensions(dims)
                    .wavelength(wavelength)
                    .add_tumor([0.0, 0.0, 0.0], 0.0001, 0.7)
                    .build(),
                wavelength,
            );
            assert_wavelength_error(
                PhantomBuilder::tumor_detection()
                    .dimensions(dims)
                    .wavelength(wavelength)
                    .add_tumor([0.0, 0.0, 0.0], 0.0001, 0.7)
                    .build(),
                wavelength,
            );
            assert_wavelength_error(
                PhantomBuilder::vascular()
                    .dimensions(dims)
                    .wavelength(wavelength)
                    .add_vessel([0.0, 0.0, 0.0], [0.0, 0.0, 0.001], 0.0001, 0.7)
                    .build(),
                wavelength,
            );
        }
    }

    #[test]
    fn missing_dimensions_remains_a_distinct_builder_error() {
        assert!(matches!(
            PhantomBuilder::vascular().build(),
            Err(PhantomError::MissingDimensions)
        ));
    }
}

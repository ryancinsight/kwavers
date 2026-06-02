//! Clinical Phantom Builders
//!
//! Provides realistic tissue phantom constructors for photoacoustic and optical imaging
//! validation, protocol development, and algorithm testing.

pub mod builder;
pub mod presets;
pub mod types;
pub mod utils;

pub use builder::{
    BloodOxygenationPhantomBuilder, LayeredTissuePhantomBuilder, PhantomBuilder,
    TumorDetectionPhantomBuilder, VascularPhantomBuilder,
};
pub use presets::ClinicalPhantoms;
pub use types::{LayerSpec, PhantomTissueType, PhantomType, TumorSpec, VesselSpec};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::GridDimensions;
    use crate::medium::properties::OpticalPropertyData;

    #[test]
    fn test_blood_oxygenation_phantom() {
        let dims = GridDimensions::new(30, 30, 30, 0.001, 0.001, 0.001);
        let phantom = PhantomBuilder::blood_oxygenation()
            .dimensions(dims)
            .wavelength(800.0)
            .add_artery([0.015, 0.015, 0.015], 0.002, 0.98)
            .add_vein([0.015, 0.015, 0.020], 0.003, 0.65)
            .build();

        assert_eq!(phantom.mu_a.len(), 27000);
        // Vessels (oxy/deoxy hemoglobin) raise absorption above the background, so
        // the map must span a non-trivial range. Computed inline to keep phantom
        // fixtures free of any physics-layer dependency.
        let max = phantom.mu_a.iter().copied().fold(f64::MIN, f64::max);
        let min = phantom.mu_a.iter().copied().fold(f64::MAX, f64::min);
        assert!(max > min);
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

        assert_eq!(phantom.mu_a.len(), 16000);

        // Check first layer is skin (high absorption)
        let skin_props = phantom.get(10, 10, 1).unwrap();
        assert!(skin_props.absorption_coefficient > 3.0);

        // Check second layer is fat (low absorption)
        let fat_props = phantom.get(10, 10, 6).unwrap();
        assert!(fat_props.absorption_coefficient < 1.0);
    }

    #[test]
    fn test_tumor_detection_phantom() {
        let dims = GridDimensions::new(25, 25, 25, 0.001, 0.001, 0.001);
        let phantom = PhantomBuilder::tumor_detection()
            .dimensions(dims)
            .wavelength(800.0)
            .background(OpticalPropertyData::soft_tissue())
            .add_tumor([0.0125, 0.0125, 0.0125], 0.003, 0.60)
            .build();

        assert_eq!(phantom.mu_a.len(), 15625);
    }

    #[test]
    fn test_vascular_phantom() {
        let dims = GridDimensions::new(20, 20, 30, 0.001, 0.001, 0.001);
        let phantom = PhantomBuilder::vascular()
            .dimensions(dims)
            .wavelength(800.0)
            .add_vessel([0.01, 0.01, 0.0], [0.01, 0.01, 0.03], 0.001, 0.95)
            .build();

        assert_eq!(phantom.mu_a.len(), 12000);
    }

    #[test]
    fn test_standard_blood_oxygenation() {
        let dims = GridDimensions::new(30, 30, 30, 0.001, 0.001, 0.001);
        let phantom = ClinicalPhantoms::standard_blood_oxygenation(dims);

        assert_eq!(phantom.mu_a.len(), 27000);
        // Non-trivial mean absorption (inlined to keep fixtures physics-free).
        let mean = phantom.mu_a.iter().copied().sum::<f64>() / phantom.mu_a.len() as f64;
        assert!(mean > 0.0);
    }

    #[test]
    fn test_skin_tissue_phantom() {
        let dims = GridDimensions::new(20, 20, 40, 0.001, 0.001, 0.001);
        let phantom = ClinicalPhantoms::skin_tissue(dims);

        assert_eq!(phantom.mu_a.len(), 16000);
    }

    #[test]
    fn test_breast_tumor_phantom() {
        let dims = GridDimensions::new(30, 30, 30, 0.001, 0.001, 0.001);
        let phantom = ClinicalPhantoms::breast_tumor(dims, [0.015, 0.015, 0.015]);

        assert_eq!(phantom.mu_a.len(), 27000);
    }

    #[test]
    fn test_vascular_network_phantom() {
        let dims = GridDimensions::new(25, 25, 30, 0.001, 0.001, 0.001);
        let phantom = ClinicalPhantoms::vascular_network(dims);

        assert_eq!(phantom.mu_a.len(), 18750);
    }
}

//! `CEUSImagingParameters` — CEUS imaging parameters.

use aequitas::systems::si::quantities::{Frequency, Length};

/// CEUS imaging parameters
#[derive(Debug, Clone)]
pub struct CEUSImagingParameters {
    /// Transmit frequency (Hz)
    pub frequency: Frequency<f64>,
    /// Mechanical index
    pub mechanical_index: f64,
    /// Frame rate (Hz)
    pub frame_rate: Frequency<f64>,
    /// Dynamic range (dB)
    pub dynamic_range: f64,
    /// Field of view (mm)
    pub fov: (Length<f64>, Length<f64>),
    /// Imaging depth (mm)
    pub depth: Length<f64>,
}

impl Default for CEUSImagingParameters {
    fn default() -> Self {
        use aequitas::systems::si::units::{Hertz, Millimeter};
        use kwavers_core::constants::numerical::MHZ_TO_HZ;
        Self {
            frequency: Frequency::from_base(3.0 * MHZ_TO_HZ), // 3 MHz
            mechanical_index: 0.1,                            // Low MI for CEUS
            frame_rate: Frequency::from_unit::<Hertz>(10.0),  // 10 fps
            dynamic_range: 60.0,                              // 60 dB
            fov: (
                Length::from_unit::<Millimeter>(80.0),
                Length::from_unit::<Millimeter>(60.0),
            ), // 80x60 mm
            depth: Length::from_unit::<Millimeter>(150.0),    // 150 mm
        }
    }
}

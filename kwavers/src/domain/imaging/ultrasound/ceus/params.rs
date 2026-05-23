//! `CEUSImagingParameters` — CEUS imaging parameters.

/// CEUS imaging parameters
#[derive(Debug, Clone)]
pub struct CEUSImagingParameters {
    /// Transmit frequency (Hz)
    pub frequency: f64,
    /// Mechanical index
    pub mechanical_index: f64,
    /// Frame rate (Hz)
    pub frame_rate: f64,
    /// Dynamic range (dB)
    pub dynamic_range: f64,
    /// Field of view (mm)
    pub fov: (f64, f64),
    /// Imaging depth (mm)
    pub depth: f64,
}

impl Default for CEUSImagingParameters {
    fn default() -> Self {
        use crate::core::constants::numerical::MHZ_TO_HZ;
        Self {
            frequency: 3.0 * MHZ_TO_HZ, // 3 MHz
            mechanical_index: 0.1, // Low MI for CEUS
            frame_rate: 10.0,      // 10 fps
            dynamic_range: 60.0,   // 60 dB
            fov: (80.0, 60.0),     // 80x60 mm
            depth: 150.0,          // 150 mm
        }
    }
}

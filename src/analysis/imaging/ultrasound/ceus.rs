//! Contrast-Enhanced Ultrasound (CEUS) analysis types

use ndarray::Array3;

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
        Self {
            frequency: 3.0e6,
            mechanical_index: 0.1,
            frame_rate: 10.0,
            dynamic_range: 50.0,
            fov: (60.0, 60.0),
            depth: 80.0,
        }
    }
}

/// Microbubble agent properties
#[derive(Debug, Clone)]
pub struct Microbubble {
    /// Bubble radius (μm)
    pub radius: f64,
    /// Shell thickness (nm)
    pub shell_thickness: f64,
    /// Shell modulus (Pa)
    pub shell_modulus: f64,
    /// Resonance frequency (Hz)
    pub resonance_frequency: f64,
}

impl Default for Microbubble {
    fn default() -> Self {
        Self {
            radius: 2.0,
            shell_thickness: 10.0,
            shell_modulus: 1e7,
            resonance_frequency: 2e6,
        }
    }
}

/// Microbubble population statistics
#[derive(Debug, Clone)]
pub struct MicrobubblePopulation {
    /// Number of bubbles
    pub count: u32,
    /// Mean radius (μm)
    pub mean_radius: f64,
    /// Radius distribution std dev
    pub radius_std: f64,
}

impl Default for MicrobubblePopulation {
    fn default() -> Self {
        Self {
            count: 1_000_000,
            mean_radius: 2.0,
            radius_std: 0.5,
        }
    }
}

/// Perfusion map from CEUS data
#[derive(Debug, Clone)]
pub struct PerfusionMap {
    /// Perfusion intensity data
    pub intensity: Array3<f64>,
    /// Time-to-peak data
    pub time_to_peak: Array3<f64>,
    /// Mean transit time data
    pub mean_transit_time: Array3<f64>,
}

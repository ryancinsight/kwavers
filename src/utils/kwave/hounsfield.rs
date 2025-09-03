//! Hounsfield unit conversions for CT data
//!
//! Based on Mast (2000) empirical relationships for medical imaging

/// Hounsfield unit conversions for CT data
#[derive(Debug)]
pub struct HounsfieldUnits;

impl HounsfieldUnits {
    /// Convert Hounsfield units to density (kg/m³)
    #[must_use]
    pub fn to_density(hu: f64) -> f64 {
        // Linear relationship: density = 1000 * (1 + HU/1000)
        // Based on water = 0 HU = 1000 kg/m³
        1000.0 * (1.0 + hu / 1000.0)
    }

    /// Convert density to Hounsfield units
    #[must_use]
    pub fn from_density(density: f64) -> f64 {
        // Inverse of to_density
        1000.0 * (density / 1000.0 - 1.0)
    }

    /// Convert Hounsfield units to sound speed (m/s)
    /// Based on Mast (2000) empirical relationship
    #[must_use]
    pub fn to_sound_speed(hu: f64) -> f64 {
        // Mast's formula for different tissue types
        if hu < -100.0 {
            // Fat-like tissue
            1450.0 + 0.5 * hu
        } else if hu < 100.0 {
            // Soft tissue
            1540.0 + 0.3 * hu
        } else {
            // Bone-like tissue
            1580.0 + 1.6 * hu
        }
    }

    /// Convert Hounsfield units to acoustic impedance
    #[must_use]
    pub fn to_impedance(hu: f64) -> f64 {
        let density = Self::to_density(hu);
        let sound_speed = Self::to_sound_speed(hu);
        density * sound_speed
    }

    /// Get typical tissue properties from HU value
    #[must_use]
    pub fn classify_tissue(hu: f64) -> &'static str {
        match hu {
            h if h < -1000.0 => "Air",
            h if h < -100.0 => "Fat",
            h if h < -10.0 => "Water",
            h if h < 40.0 => "Soft Tissue",
            h if h < 100.0 => "Muscle",
            h if h < 300.0 => "Liver",
            h if h < 700.0 => "Trabecular Bone",
            _ => "Cortical Bone",
        }
    }
}

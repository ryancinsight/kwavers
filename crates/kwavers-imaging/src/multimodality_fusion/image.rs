use leto::Array3;

/// Medical image data with metadata
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Image modality type
    pub modality: ImageModality,
    /// 3D image array (nx, ny, nz), owned by Leto.
    pub data: Array3<f64>,
    /// Voxel spacing (dx, dy, dz) in mm
    pub voxel_spacing_mm: (f64, f64, f64),
    /// Image origin (x, y, z) in mm
    pub origin_mm: (f64, f64, f64),
    /// Intensity range (min, max)
    pub intensity_range: (f64, f64),
    /// Image dimensions
    pub dimensions: (usize, usize, usize),
}

/// Image modality type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageModality {
    /// Computed Tomography
    CT,
    /// Magnetic Resonance
    MR,
    /// Ultrasound
    Ultrasound,
    /// Positron Emission Tomography
    PET,
    /// Single Photon Emission Computed Tomography
    SPECT,
    /// Other modality
    Other,
}

impl std::fmt::Display for ImageModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CT => write!(f, "CT"),
            Self::MR => write!(f, "MR"),
            Self::Ultrasound => write!(f, "Ultrasound"),
            Self::PET => write!(f, "PET"),
            Self::SPECT => write!(f, "SPECT"),
            Self::Other => write!(f, "Other"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_modality_display() {
        assert_eq!(ImageModality::CT.to_string(), "CT");
        assert_eq!(ImageModality::MR.to_string(), "MR");
        assert_eq!(ImageModality::Ultrasound.to_string(), "Ultrasound");
    }
}

//! DICOM types: `DicomModality` and `DicomMetadata`.

/// DICOM imaging modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DicomModality {
    /// Computed Tomography.
    CT,
    /// Magnetic Resonance Imaging.
    MR,
    /// Ultrasound.
    US,
    /// Radiotherapy Dose.
    RD,
    /// Any other modality.
    Other,
}

impl std::fmt::Display for DicomModality {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::CT => write!(f, "CT"),
            Self::MR => write!(f, "MR"),
            Self::US => write!(f, "US"),
            Self::RD => write!(f, "RD"),
            Self::Other => write!(f, "Other"),
        }
    }
}

impl DicomModality {
    /// Parse modality from a DICOM string code.
    pub fn from_code(code: &str) -> Self {
        match code {
            "CT" => Self::CT,
            "MR" => Self::MR,
            "US" => Self::US,
            "RD" => Self::RD,
            _ => Self::Other,
        }
    }
}

/// Metadata extracted from a DICOM file header.
#[derive(Debug, Clone)]
pub struct DicomMetadata {
    /// Image dimensions (nx, ny, nz).
    pub dimensions: (usize, usize, usize),
    /// Voxel spacing in mm (dx, dy, dz).
    pub voxel_spacing_mm: (f64, f64, f64),
    /// Voxel spacing in metres (dx, dy, dz).
    pub voxel_spacing_m: (f64, f64, f64),
    /// Affine transformation matrix (4×4) — maps voxel indices to physical coordinates.
    pub affine: [[f64; 4]; 4],
    /// DICOM modality type.
    pub modality: DicomModality,
    /// Patient ID from DICOM header.
    pub patient_id: String,
    /// Patient name.
    pub patient_name: String,
    /// Patient birth date (YYYYMMDD format, if available).
    pub patient_birth_date: Option<String>,
    /// Patient sex (M, F, or O for other).
    pub patient_sex: Option<String>,
    /// Study date (YYYYMMDD format).
    pub study_date: String,
    /// Study time (HHMMSS.ffffff format).
    pub study_time: String,
    /// Study description.
    pub study_description: String,
    /// Series description.
    pub series_description: String,
    /// Series instance UID.
    pub series_instance_uid: String,
    /// Study instance UID.
    pub study_instance_uid: String,
    /// Number of slices in the series.
    pub num_slices: usize,
    /// Slice thickness in mm.
    pub slice_thickness_mm: f64,
    /// Image Position (Patient) — physical location of first voxel.
    pub image_position: Option<[f64; 3]>,
    /// Image Orientation (Patient) — direction cosines.
    pub image_orientation: Option<[f64; 6]>,
    /// Min/Max intensity values.
    pub intensity_range: (f64, f64),
    /// Window centre (for display).
    pub window_center: Option<f64>,
    /// Window width (for display).
    pub window_width: Option<f64>,
    /// CT Rescale Intercept (b in `HU = pixel_value × slope + intercept`).
    pub rescale_intercept: Option<f64>,
    /// CT Rescale Slope (m in `HU = pixel_value × slope + intercept`).
    pub rescale_slope: Option<f64>,
}

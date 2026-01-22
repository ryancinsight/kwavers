//! DICOM file format support for medical imaging data
//!
//! Implementation using the `dicom` crate for comprehensive
//! handling of medical imaging data, metadata, and clinical workflows.
//!
//! ## Features
//!
//! - **DICOM Parsing**: Full DICOM file parsing with tag-based access
//! - **Image Data**: Pixel data extraction for various transfer syntaxes
//! - **Metadata**: Comprehensive access to DICOM headers and attributes
//! - **Series Management**: Multi-image series handling and organization
//! - **Clinical Integration**: Patient data, study information, imaging parameters
//! - **Validation**: DICOM conformance checking and error handling

use crate::core::error::{DataError, KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use dicom::core::{DataElement, Tag, VR};
use dicom::object::{FileDicomObject, InMemDicomObject};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// DICOM file reader with comprehensive parsing capabilities
#[derive(Debug)]
pub struct DicomReader {
    /// Enable verbose logging
    verbose: bool,
    /// Maximum file size to prevent memory issues (MB)
    max_file_size_mb: usize,
}

impl Default for DicomReader {
    fn default() -> Self {
        Self::new()
    }
}

impl DicomReader {
    /// Create a new DICOM reader with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            verbose: false,
            max_file_size_mb: 100, // 100MB default limit
        }
    }

    /// Enable verbose logging during file operations
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set maximum file size limit in MB
    #[must_use]
    pub fn with_max_file_size(mut self, size_mb: usize) -> Self {
        self.max_file_size_mb = size_mb;
        self
    }

    /// Read a single DICOM file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, parsed, or contains invalid DICOM data
    pub fn read_file<P: AsRef<Path>>(&self, path: P) -> KwaversResult<DicomObject> {
        if self.verbose {
            println!("Reading DICOM file: {:?}", path.as_ref());
        }

        // Check file size
        let metadata = std::fs::metadata(&path)?;
        let file_size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        if file_size_mb > self.max_file_size_mb as f64 {
            return Err(KwaversError::Data(DataError::InvalidFormat {
                format: "DICOM".to_string(),
                reason: format!(
                    "File too large: {}MB, max allowed is {}MB",
                    file_size_mb, self.max_file_size_mb
                ),
            }));
        }

        // Open and parse DICOM file
        let dicom_obj = FileDicomObject::open_file(&path)?;

        // Validate DICOM conformance
        self.validate_dicom_object(&dicom_obj)?;

        // Extract metadata and pixel data
        let metadata = self.extract_metadata(&dicom_obj)?;
        let pixel_data = self.extract_pixel_data(&dicom_obj)?;

        Ok(DicomObject {
            file_path: path.as_ref().to_path_buf(),
            metadata,
            pixel_data,
        })
    }

    /// Read a directory containing DICOM files and organize into series
    ///
    /// # Errors
    ///
    /// Returns an error if files cannot be read or organized
    pub fn read_directory<P: AsRef<Path>>(&self, dir_path: P) -> KwaversResult<DicomStudy> {
        if self.verbose {
            println!("Reading DICOM directory: {:?}", dir_path.as_ref());
        }

        let mut dicom_files = Vec::new();
        let mut series_map: HashMap<String, Vec<DicomObject>> = HashMap::new();

        // Find all DICOM files in directory
        for entry in std::fs::read_dir(&dir_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                // Check if file is DICOM by trying to read it
                if let Ok(dicom_obj) = self.read_file(&path) {
                    dicom_files.push(dicom_obj);
                } else if self.verbose {
                    println!("Skipping non-DICOM file: {:?}", path);
                }
            }
        }

        // Group files by series
        for dicom_obj in dicom_files {
            let series_uid = dicom_obj
                .metadata
                .get("SeriesInstanceUID")
                .and_then(|v| v.as_string())
                .unwrap_or_else(|| "unknown".to_string());

            series_map
                .entry(series_uid)
                .or_insert_with(Vec::new)
                .push(dicom_obj);
        }

        // Convert to DicomSeries objects
        let mut series = Vec::new();
        for (series_uid, objects) in series_map {
            let metadata = self.extract_series_metadata(&objects);
            let series_obj = DicomSeries {
                series_instance_uid: series_uid,
                instances: objects,
                metadata,
            };
            series.push(series_obj);
        }

        // Extract study-level metadata
        let study_metadata = if !series.is_empty() {
            self.extract_study_metadata(&series[0])
        } else {
            DicomStudyMetadata::default()
        };

        Ok(DicomStudy {
            study_metadata,
            series,
        })
    }

    /// Validate DICOM object for required fields and conformance
    fn validate_dicom_object(&self, obj: &FileDicomObject<InMemDicomObject>) -> KwaversResult<()> {
        // Check for required DICOM headers
        let required_tags = vec![
            Tag(0x0008, 0x0016), // SOP Class UID
            Tag(0x0008, 0x0018), // SOP Instance UID
            Tag(0x0008, 0x0060), // Modality
        ];

        for tag in required_tags {
            if obj.element(tag).is_err() {
                return Err(KwaversError::Data(DataError::InvalidFormat {
                    format: "DICOM".to_string(),
                    reason: format!("Missing required tag: {:04X},{:04X}", tag.0, tag.1),
                }));
            }
        }

        // Check transfer syntax
        if let Ok(ts_element) = obj.element(Tag(0x0002, 0x0010)) {
            match ts_element.string() {
                Ok(ts_uid) => {
                    // For now, accept common transfer syntaxes
                    // Could be extended with a whitelist if needed
                    if self.verbose {
                        println!("Transfer syntax: {}", ts_uid);
                    }
                }
                Err(_) => {
                    // Non-string transfer syntax, skip validation
                }
            }
        }

        Ok(())
    }

    /// Extract metadata from DICOM object
    fn extract_metadata(
        &self,
        obj: &FileDicomObject<InMemDicomObject>,
    ) -> KwaversResult<HashMap<String, DicomValue>> {
        let mut metadata = HashMap::new();

        // Common DICOM tags to extract
        let tags_to_extract = vec![
            ("StudyInstanceUID", Tag(0x0020, 0x000D)),
            ("SeriesInstanceUID", Tag(0x0020, 0x000E)),
            ("SOPInstanceUID", Tag(0x0008, 0x0018)),
            ("SOPClassUID", Tag(0x0008, 0x0016)),
            ("Modality", Tag(0x0008, 0x0060)),
            ("PatientID", Tag(0x0010, 0x0020)),
            ("PatientName", Tag(0x0010, 0x0010)),
            ("StudyDate", Tag(0x0008, 0x0020)),
            ("StudyTime", Tag(0x0008, 0x0030)),
            ("SeriesDate", Tag(0x0008, 0x0020)),
            ("SeriesTime", Tag(0x0008, 0x0030)),
            ("InstitutionName", Tag(0x0008, 0x0080)),
            ("Manufacturer", Tag(0x0008, 0x0070)),
            ("ManufacturerModelName", Tag(0x0008, 0x1090)),
            ("SoftwareVersions", Tag(0x0018, 0x1020)),
            // Image dimensions
            ("Rows", Tag(0x0028, 0x0010)),
            ("Columns", Tag(0x0028, 0x0011)),
            ("BitsAllocated", Tag(0x0028, 0x0100)),
            ("BitsStored", Tag(0x0028, 0x0101)),
            ("HighBit", Tag(0x0028, 0x0102)),
            ("PixelRepresentation", Tag(0x0028, 0x0103)),
            ("SamplesPerPixel", Tag(0x0028, 0x0002)),
            ("PhotometricInterpretation", Tag(0x0028, 0x0004)),
            ("PixelSpacing", Tag(0x0028, 0x0030)),
            ("SliceThickness", Tag(0x0018, 0x0050)),
            ("SpacingBetweenSlices", Tag(0x0018, 0x0088)),
            // Ultrasound specific
            ("UltrasoundColorDataPresent", Tag(0x0028, 0x0014)),
            ("FrameTime", Tag(0x0018, 0x1063)),
            ("FrameTimeVector", Tag(0x0018, 0x1065)),
            ("StartTrim", Tag(0x0008, 0x9807)),
            ("StopTrim", Tag(0x0008, 0x9808)),
            ("RecommendedDisplayFrameRate", Tag(0x0008, 0x9458)),
            ("CineRate", Tag(0x0018, 0x0040)),
            ("EffectiveDuration", Tag(0x0018, 0x0072)),
            ("NumberOfFrames", Tag(0x0028, 0x0008)),
        ];

        for (name, tag) in tags_to_extract {
            if let Ok(element) = obj.element(tag) {
                let value = self.element_to_dicom_value(&element);
                metadata.insert(name.to_string(), value);
            }
        }

        Ok(metadata)
    }

    /// Extract series-level metadata
    fn extract_series_metadata(&self, objects: &[DicomObject]) -> HashMap<String, DicomValue> {
        let mut series_metadata = HashMap::new();

        if let Some(first_obj) = objects.first() {
            // Copy common metadata from first object
            for (key, value) in &first_obj.metadata {
                series_metadata.insert(key.clone(), value.clone());
            }

            // Add series-specific information
            series_metadata.insert(
                "NumberOfInstances".to_string(),
                DicomValue::Integer(objects.len() as i64),
            );

            // Calculate temporal information if available
            if let (Some(frame_time), Some(num_frames)) = (
                first_obj
                    .metadata
                    .get("FrameTime")
                    .and_then(|v| v.as_float()),
                first_obj
                    .metadata
                    .get("NumberOfFrames")
                    .and_then(|v| v.as_int()),
            ) {
                let total_duration = frame_time * num_frames as f64 / 1000.0; // Convert to seconds
                series_metadata.insert(
                    "TotalDuration".to_string(),
                    DicomValue::Float(total_duration),
                );
            }
        }

        series_metadata
    }

    /// Extract study-level metadata
    fn extract_study_metadata(&self, first_series: &DicomSeries) -> DicomStudyMetadata {
        DicomStudyMetadata {
            study_instance_uid: first_series
                .metadata
                .get("StudyInstanceUID")
                .and_then(|v| v.as_string())
                .unwrap_or_default(),
            patient_id: first_series
                .metadata
                .get("PatientID")
                .and_then(|v| v.as_string())
                .unwrap_or_default(),
            patient_name: first_series
                .metadata
                .get("PatientName")
                .and_then(|v| v.as_string())
                .unwrap_or_default(),
            study_date: first_series
                .metadata
                .get("StudyDate")
                .and_then(|v| v.as_string())
                .unwrap_or_default(),
            study_description: first_series
                .metadata
                .get("StudyDescription")
                .and_then(|v| v.as_string())
                .unwrap_or_default(),
            modality: first_series
                .metadata
                .get("Modality")
                .and_then(|v| v.as_string())
                .unwrap_or_default(),
            institution_name: first_series
                .metadata
                .get("InstitutionName")
                .and_then(|v| v.as_string()),
        }
    }

    /// Extract pixel data from DICOM object
    fn extract_pixel_data(
        &self,
        obj: &FileDicomObject<InMemDicomObject>,
    ) -> KwaversResult<Option<DicomPixelData>> {
        // Check if pixel data exists
        match obj.element(Tag(0x7FE0, 0x0010)) {
            Ok(pixel_element) => {
                // Extract basic image information
                let rows = obj
                    .element(Tag(0x0028, 0x0010))
                    .ok()
                    .and_then(|e| e.uint16().ok())
                    .unwrap_or(0) as usize;
                let cols = obj
                    .element(Tag(0x0028, 0x0011))
                    .ok()
                    .and_then(|e| e.uint16().ok())
                    .unwrap_or(0) as usize;
                let frames = obj
                    .element(Tag(0x0028, 0x0008))
                    .ok()
                    .and_then(|e| e.uint16().ok())
                    .unwrap_or(1) as usize;
                let samples_per_pixel = obj
                    .element(Tag(0x0028, 0x0002))
                    .ok()
                    .and_then(|e| e.uint16().ok())
                    .unwrap_or(1) as usize;
                let bits_allocated = obj
                    .element(Tag(0x0028, 0x0100))
                    .ok()
                    .and_then(|e| e.uint16().ok())
                    .unwrap_or(8) as usize;
                let bits_stored = obj
                    .element(Tag(0x0028, 0x0101))
                    .ok()
                    .and_then(|e| e.uint16().ok())
                    .unwrap_or(8) as usize;

                // Extract pixel data if available
                let pixel_data_raw = pixel_element.to_bytes().unwrap_or_default().to_vec();

                Ok(Some(DicomPixelData {
                    rows,
                    columns: cols,
                    frames,
                    samples_per_pixel,
                    bits_allocated,
                    bits_stored,
                    photometric_interpretation: obj
                        .element(Tag(0x0028, 0x0004))
                        .ok()
                        .and_then(|e| e.string().ok())
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "UNKNOWN".to_string()),
                    pixel_data_raw,
                }))
            }
            Err(_) => Ok(None), // No pixel data
        }
    }

    /// Convert DICOM element to our DicomValue type
    fn element_to_dicom_value(&self, element: &DataElement<InMemDicomObject>) -> DicomValue {
        match element.vr() {
            VR::AE
            | VR::AS
            | VR::CS
            | VR::DA
            | VR::DT
            | VR::LO
            | VR::LT
            | VR::PN
            | VR::SH
            | VR::ST
            | VR::TM
            | VR::UC
            | VR::UI
            | VR::UR
            | VR::UT => {
                DicomValue::String(element.string().map(|s| s.to_string()).unwrap_or_default())
            }
            VR::FL | VR::FD => DicomValue::Float(element.float64().unwrap_or(0.0)),
            VR::SL | VR::SS | VR::SV | VR::UL | VR::US | VR::UV => {
                DicomValue::Integer(element.int64().unwrap_or(0))
            }
            _ => DicomValue::String(format!("{:?}", element)),
        }
    }
}

/// DICOM value types - placeholder
#[derive(Debug, Clone, PartialEq)]
pub enum DicomValue {
    String(String),
    Integer(i64),
    Float(f64),
}

/// DICOM pixel data information - placeholder
#[derive(Debug, Clone)]
pub struct DicomPixelData {
    pub rows: usize,
    pub columns: usize,
    pub frames: usize,
    pub samples_per_pixel: usize,
    pub bits_allocated: usize,
    pub bits_stored: usize,
    pub photometric_interpretation: String,
    pub pixel_data_raw: Vec<u8>,
}

/// Individual DICOM object (single file) - placeholder
#[derive(Debug, Clone)]
pub struct DicomObject {
    pub file_path: PathBuf,
    pub metadata: HashMap<String, DicomValue>,
    pub pixel_data: Option<DicomPixelData>,
}

/// DICOM series (collection of related images) - placeholder
#[derive(Debug, Clone)]
pub struct DicomSeries {
    pub series_instance_uid: String,
    pub instances: Vec<DicomObject>,
    pub metadata: HashMap<String, DicomValue>,
}

/// DICOM study (collection of series) - placeholder
#[derive(Debug, Clone)]
pub struct DicomStudy {
    pub study_metadata: DicomStudyMetadata,
    pub series: Vec<DicomSeries>,
}

/// DICOM study metadata - placeholder
#[derive(Debug, Clone, Default)]
pub struct DicomStudyMetadata {
    pub study_instance_uid: String,
    pub patient_id: String,
    pub patient_name: String,
    pub study_date: String,
    pub study_description: String,
    pub modality: String,
    pub institution_name: Option<String>,
}

impl DicomStudy {
    /// Get all ultrasound series in the study
    #[must_use]
    pub fn ultrasound_series(&self) -> Vec<&DicomSeries> {
        self.series
            .iter()
            .filter(|s| {
                s.metadata
                    .get("Modality")
                    .and_then(|v| v.as_string())
                    .map_or(false, |modality| modality == "US")
            })
            .collect()
    }

    /// Get series by modality
    #[must_use]
    pub fn series_by_modality(&self, modality: &str) -> Vec<&DicomSeries> {
        self.series
            .iter()
            .filter(|s| {
                s.metadata
                    .get("Modality")
                    .and_then(|v| v.as_string())
                    .map_or(false, |m| m == modality)
            })
            .collect()
    }

    /// Convert DICOM study to kwavers Grid for simulation
    ///
    /// # Errors
    ///
    /// Returns an error if the DICOM data cannot be converted to a grid
    pub fn to_grid(&self) -> KwaversResult<Grid> {
        // Find the first ultrasound series
        let us_series = self.ultrasound_series();
        if us_series.is_empty() {
            return Err(KwaversError::Data(DataError::InvalidFormat {
                format: "DICOM".to_string(),
                reason: "No ultrasound data found in study".to_string(),
            }));
        }

        let first_series = us_series[0];
        if first_series.instances.is_empty() {
            return Err(KwaversError::Data(DataError::InvalidFormat {
                format: "DICOM".to_string(),
                reason: "Ultrasound series contains no instances".to_string(),
            }));
        }

        // Use first instance to determine grid dimensions
        let first_instance = &first_series.instances[0];
        if let Some(pixel_data) = &first_instance.pixel_data {
            // Create grid based on DICOM image dimensions
            // Note: This is a simplified conversion - real implementation would need
            // to handle spatial calibration, multiple frames, etc.
            Ok(Grid::new(
                pixel_data.rows,
                pixel_data.columns,
                pixel_data.frames.max(1), // At least 1 slice
                0.001,                    // Default 1mm spacing (would be read from DICOM)
                0.001,
                0.001,
            )?)
        } else {
            Err(KwaversError::Data(DataError::InvalidFormat {
                format: "DICOM".to_string(),
                reason: "DICOM instance contains no pixel data".to_string(),
            }))
        }
    }
}

impl DicomValue {
    /// Get value as string if possible
    #[must_use]
    pub fn as_string(&self) -> Option<String> {
        match self {
            DicomValue::String(s) => Some(s.clone()),
            DicomValue::Integer(i) => Some(i.to_string()),
            DicomValue::Float(f) => Some(f.to_string()),
        }
    }

    /// Get value as integer if possible
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            DicomValue::Integer(i) => Some(*i),
            DicomValue::Float(f) => Some(*f as i64),
            DicomValue::String(s) => s.parse().ok(),
        }
    }

    /// Get value as float if possible
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            DicomValue::Float(f) => Some(*f),
            DicomValue::Integer(i) => Some(*i as f64),
            DicomValue::String(s) => s.parse().ok(),
        }
    }
}

// DICOM-specific functionality implemented

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dicom_reader_creation() {
        let reader = DicomReader::new();
        assert!(!reader.verbose);
        assert_eq!(reader.max_file_size_mb, 100);
    }

    #[test]
    fn test_dicom_reader_with_options() {
        let reader = DicomReader::new()
            .with_verbose(true)
            .with_max_file_size(200);

        assert!(reader.verbose);
        assert_eq!(reader.max_file_size_mb, 200);
    }

    #[test]
    fn test_dicom_value_conversions() {
        let string_val = DicomValue::String("test".to_string());
        assert_eq!(string_val.as_string(), Some("test".to_string()));

        let int_val = DicomValue::Integer(42);
        assert_eq!(int_val.as_int(), Some(42));
        assert_eq!(int_val.as_string(), Some("42".to_string()));

        let float_val = DicomValue::Float(3.14);
        assert_eq!(float_val.as_float(), Some(3.14));
        assert_eq!(float_val.as_string(), Some("3.14".to_string()));
    }

    #[test]
    fn test_study_ultrasound_filtering() {
        let study = DicomStudy {
            study_metadata: DicomStudyMetadata::default(),
            series: vec![
                DicomSeries {
                    series_instance_uid: "us1".to_string(),
                    instances: vec![],
                    metadata: HashMap::from([(
                        "Modality".to_string(),
                        DicomValue::String("US".to_string()),
                    )]),
                },
                DicomSeries {
                    series_instance_uid: "ct1".to_string(),
                    instances: vec![],
                    metadata: HashMap::from([(
                        "Modality".to_string(),
                        DicomValue::String("CT".to_string()),
                    )]),
                },
            ],
        };

        let us_series = study.ultrasound_series();
        assert_eq!(us_series.len(), 1);
        assert_eq!(us_series[0].series_instance_uid, "us1");

        let ct_series = study.series_by_modality("CT");
        assert_eq!(ct_series.len(), 1);
        assert_eq!(ct_series[0].series_instance_uid, "ct1");
    }

    #[test]
    fn test_dicom_reader_invalid_file() {
        let reader = DicomReader::new();
        // Should return error for non-existent file
        let result = reader.read_file("nonexistent.dcm");
        assert!(result.is_err());
    }
}

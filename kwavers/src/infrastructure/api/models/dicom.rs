//! DICOM standards-compliance integration types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct DICOMIntegrationRequest {
    /// Study instance UID
    pub study_instance_uid: String,
    /// Series instance UID
    pub series_instance_uid: String,
    /// SOP instance UID
    pub sop_instance_uid: String,
    /// DICOM tags to extract
    pub requested_tags: Vec<String>,
    /// Include pixel data
    pub include_pixel_data: bool,
}

/// DICOM integration response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DICOMIntegrationResponse {
    /// Extracted DICOM metadata
    pub metadata: HashMap<String, DICOMValue>,
    /// Pixel data (if requested) as base64
    pub pixel_data: Option<String>,
    /// Study information
    pub study_info: DICOMStudyInfo,
}

/// DICOM value wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DICOMValue {
    String(String),
    Number(f64),
    Integer(i64),
    Sequence(Vec<HashMap<String, DICOMValue>>),
}

/// DICOM study information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DICOMStudyInfo {
    /// Patient ID
    pub patient_id: String,
    /// Study date
    pub study_date: String,
    /// Study description
    pub study_description: String,
    /// Modality
    pub modality: String,
    /// Institution name
    pub institution_name: Option<String>,
}

/// Mobile device optimization request

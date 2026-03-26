use crate::core::error::KwaversResult;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::Json as JsonResponse,
};
use base64::{engine::general_purpose, Engine as _};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

use crate::infrastructure::api::auth::AuthenticatedUser;
use crate::infrastructure::api::{APIError, DICOMIntegrationRequest, DICOMIntegrationResponse};

use super::state::ClinicalAppState;

/// DICOM integration service
#[derive(Debug)]
pub struct DICOMService {
    /// Connected DICOM nodes
    pub dicom_nodes: HashMap<String, DICOMNode>,
    /// Cache for parsed studies
    study_cache:
        std::sync::Mutex<HashMap<String, std::sync::Arc<crate::infrastructure::io::DicomStudy>>>,
}

impl DICOMService {
    pub fn new() -> Self {
        Self {
            dicom_nodes: HashMap::new(),
            study_cache: std::sync::Mutex::new(HashMap::new()),
        }
    }

    pub fn read_study(
        &self,
        study_uid: &str,
    ) -> KwaversResult<Option<std::sync::Arc<crate::infrastructure::io::DicomStudy>>> {
        if let Ok(cache) = self.study_cache.lock() {
            if let Some(study) = cache.get(study_uid) {
                return Ok(Some(study.clone()));
            }
        }

        let dicom_reader = crate::infrastructure::io::DicomReader::new();

        for node in self.dicom_nodes.values() {
            if let Some(storage_dir) = &node.storage_directory {
                let study_path = std::path::Path::new(storage_dir).join(study_uid);
                if study_path.exists() {
                    let study = dicom_reader.read_directory(&study_path)?;
                    let study_arc = std::sync::Arc::new(study);

                    if let Ok(mut cache) = self.study_cache.lock() {
                        if cache.len() >= 20 {
                            if let Some(key) = cache.keys().next().cloned() {
                                cache.remove(&key);
                            }
                        }
                        cache.insert(study_uid.to_string(), study_arc.clone());
                    }

                    return Ok(Some(study_arc));
                }
            }
        }

        Ok(None)
    }

    pub fn read_series(
        &self,
        study_uid: &str,
        series_uid: &str,
    ) -> KwaversResult<Option<crate::infrastructure::io::DicomSeries>> {
        if let Some(study) = self.read_study(study_uid)? {
            let series = study
                .series
                .iter()
                .find(|s| s.series_instance_uid == series_uid)
                .cloned();
            Ok(series)
        } else {
            Ok(None)
        }
    }

    pub fn read_instance(
        &self,
        study_uid: &str,
        series_uid: &str,
        instance_uid: &str,
    ) -> KwaversResult<Option<crate::infrastructure::io::DicomObject>> {
        if let Some(study) = self.read_study(study_uid)? {
            if let Some(series) = study
                .series
                .iter()
                .find(|s| s.series_instance_uid == series_uid)
            {
                let instance = series
                    .instances
                    .iter()
                    .find(|i| {
                        i.metadata
                            .get("SOPInstanceUID")
                            .and_then(|v| v.as_string())
                            .is_some_and(|uid| uid == instance_uid)
                    })
                    .cloned();
                return Ok(instance);
            }
        }
        Ok(None)
    }
}

impl Default for DICOMService {
    fn default() -> Self {
        Self::new()
    }
}

/// DICOM network node
#[derive(Debug, Clone)]
pub struct DICOMNode {
    pub node_id: String,
    pub ae_title: String,
    pub host: String,
    pub port: u16,
    pub last_seen: DateTime<Utc>,
    pub storage_directory: Option<String>,
}

/// DICOM integration endpoint
pub async fn dicom_integrate(
    State(state): State<ClinicalAppState>,
    _auth: AuthenticatedUser,
    Json(request): Json<DICOMIntegrationRequest>,
) -> Result<JsonResponse<DICOMIntegrationResponse>, (StatusCode, JsonResponse<APIError>)> {
    let dicom_service = state.dicom_service.read().await;

    let dicom_obj = dicom_service
        .read_instance(
            &request.study_instance_uid,
            &request.series_instance_uid,
            &request.sop_instance_uid,
        )
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                JsonResponse(APIError {
                    error: crate::api::APIErrorType::InternalError,
                    message: format!("Failed to read DICOM instance: {}", e),
                    details: None,
                }),
            )
        })?
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                JsonResponse(APIError {
                    error: crate::api::APIErrorType::ResourceNotFound,
                    message: format!(
                        "DICOM instance not found for study '{}' series '{}' instance '{}'",
                        request.study_instance_uid,
                        request.series_instance_uid,
                        request.sop_instance_uid
                    ),
                    details: None,
                }),
            )
        })?;

    let mut metadata: HashMap<String, crate::api::DICOMValue> = HashMap::new();
    let mut pixel_data = None;
    let mut study_info = crate::api::DICOMStudyInfo {
        patient_id: "UNKNOWN".to_string(),
        study_date: "UNKNOWN".to_string(),
        study_description: "DICOM Study".to_string(),
        modality: "UNKNOWN".to_string(),
        institution_name: None,
    };

    study_info.patient_id = dicom_obj
        .metadata
        .get("PatientID")
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "UNKNOWN".to_string());
    study_info.study_date = dicom_obj
        .metadata
        .get("StudyDate")
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "UNKNOWN".to_string());
    study_info.study_description = dicom_obj
        .metadata
        .get("StudyDescription")
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "DICOM Study".to_string());
    study_info.modality = dicom_obj
        .metadata
        .get("Modality")
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "UNKNOWN".to_string());
    study_info.institution_name = dicom_obj
        .metadata
        .get("InstitutionName")
        .and_then(|v| v.as_string());

    for (key, value) in &dicom_obj.metadata {
        let api_value = match value {
            crate::infrastructure::io::DicomValue::String(s) => {
                crate::api::DICOMValue::String(s.clone())
            }
            crate::infrastructure::io::DicomValue::Integer(i) => {
                crate::api::DICOMValue::Integer(*i)
            }
            crate::infrastructure::io::DicomValue::Float(f) => crate::api::DICOMValue::Number(*f),
        };
        metadata.insert(key.clone(), api_value);
    }

    if request.include_pixel_data {
        if let Some(pixel_info) = &dicom_obj.pixel_data {
            pixel_data = Some(general_purpose::STANDARD.encode(&pixel_info.pixel_data_raw));
        }
    }

    let extracted_metadata: HashMap<String, crate::api::DICOMValue> =
        if request.requested_tags.is_empty() {
            metadata.clone()
        } else {
            let mut extracted = HashMap::new();
            for tag in &request.requested_tags {
                if let Some(value) = metadata.get(tag) {
                    extracted.insert(tag.clone(), value.clone());
                }
            }
            extracted
        };

    let response = DICOMIntegrationResponse {
        metadata: extracted_metadata,
        pixel_data,
        study_info,
    };

    Ok(JsonResponse(response))
}

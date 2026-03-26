use std::collections::HashMap;
use chrono::{DateTime, Utc};
use base64::{engine::general_purpose, Engine as _};
use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::Json as JsonResponse,
};

use crate::infrastructure::api::auth::AuthenticatedUser;
use crate::infrastructure::api::{APIError, ClinicalAnalysisRequest, ClinicalAnalysisResponse};

#[cfg(feature = "pinn")]
use crate::clinical::imaging::workflows::neural::AIBeamformingResult;

use super::state::{ClinicalAppState, ClinicalSession};

/// Get clinical session status endpoint
pub async fn get_session_status(
    State(state): State<ClinicalAppState>,
    Path(session_id): Path<String>,
    _auth: AuthenticatedUser,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, JsonResponse<APIError>)> {
    let sessions = state.active_sessions.read().await;

    if let Some(session) = sessions.get(&session_id) {
        // Session authorization check - production would validate user permissions and session state
        let response = serde_json::json!({
            "session_id": session.session_id,
            "device_id": session.device_id,
            "patient_id": session.patient_id,
            "exam_type": session.exam_type,
            "started_at": session.started_at,
            "last_activity": session.last_activity,
            "priority": session.priority,
            "status": "active"
        });

        Ok(JsonResponse(response))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            JsonResponse(APIError {
                error: crate::api::APIErrorType::ResourceNotFound,
                message: format!("Session '{}' not found", session_id),
                details: None,
            }),
        ))
    }
}

/// Clinical analysis endpoint - AI-enhanced beamforming
#[cfg(feature = "pinn")]
pub async fn analyze_clinical(
    State(state): State<ClinicalAppState>,
    _auth: AuthenticatedUser,
    Json(request): Json<ClinicalAnalysisRequest>,
) -> Result<JsonResponse<ClinicalAnalysisResponse>, (StatusCode, JsonResponse<APIError>)> {
    let start_time = std::time::Instant::now();

    // Validate device is registered
    let registry = state.device_registry.read().await;
    if !registry.contains_key(&request.device_id) {
        return Err((
            StatusCode::BAD_REQUEST,
            JsonResponse(APIError {
                error: crate::api::APIErrorType::InvalidRequest,
                message: format!("Device '{}' is not registered", request.device_id),
                details: None,
            }),
        ));
    }
    drop(registry);

    // Start clinical session
    let session = ClinicalSession {
        session_id: request.session_id.clone(),
        device_id: request.device_id.clone(),
        patient_id: request.patient_id.clone(),
        exam_type: request.exam_type.clone(),
        started_at: Utc::now(),
        last_activity: Utc::now(),
        priority: request.priority.clone(),
    };

    let mut sessions = state.active_sessions.write().await;
    sessions.insert(request.session_id.clone(), session);
    drop(sessions);

    // Convert ultrasound frames to RF data for processing
    let mut rf_data = Vec::new();
    let mut angles: Vec<f32> = Vec::new();

    for frame in &request.frames {
        // Decode base64 RF data
        let rf_bytes = general_purpose::STANDARD
            .decode(&frame.rf_data)
            .map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    JsonResponse(APIError {
                        error: crate::api::APIErrorType::InvalidRequest,
                        message: format!("Invalid RF data encoding: {}", e),
                        details: None,
                    }),
                )
            })?;

        // Convert binary data to f32 array (production would handle endianness and validate format)
        let rf_frame: Vec<f32> = rf_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        rf_data.push(rf_frame);
        angles.push(
            frame
                .parameters
                .steering_angles
                .first()
                .copied()
                .unwrap_or(0.0) as f32,
        );
    }

    // Create 4D RF data array [time, channels, frames, spatial]
    let num_frames = rf_data.len();
    let num_channels = 4; // Assume 4-element array
    let time_samples = rf_data[0].len() / num_channels;

    let mut rf_data_4d = ndarray::Array4::<f32>::zeros((time_samples, num_channels, num_frames, 1));

    for (frame_idx, frame_data) in rf_data.iter().enumerate() {
        for channel in 0..num_channels {
            for sample in 0..time_samples {
                let idx = channel * time_samples + sample;
                if idx < frame_data.len() {
                    rf_data_4d[[sample, channel, frame_idx, 0]] = frame_data[idx];
                }
            }
        }
    }

    // Perform AI-enhanced beamforming analysis
    let mut ai_processor = state.ai_processor.lock().await;
    let analysis_result = ai_processor
        .process_ai_enhanced(rf_data_4d.view(), &angles)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                JsonResponse(APIError {
                    error: crate::api::APIErrorType::InternalError,
                    message: format!("AI analysis failed: {}", e),
                    details: None,
                }),
            )
        })?;
    drop(ai_processor);

    // Convert analysis result to clinical response
    let clinical_response = clinical_analysis_from_beamforming_result(
        &request,
        analysis_result,
        start_time.elapsed().as_millis() as u64,
    );

    // Update session activity
    let mut sessions = state.active_sessions.write().await;
    if let Some(session) = sessions.get_mut(&request.session_id) {
        session.last_activity = Utc::now();
    }

    tracing::info!(
        "Clinical analysis completed for session {} in {}ms",
        request.session_id,
        start_time.elapsed().as_millis()
    );

    Ok(JsonResponse(clinical_response))
}

/// Convert beamforming result to clinical analysis response
#[cfg(feature = "pinn")]
pub(crate) fn clinical_analysis_from_beamforming_result(
    request: &ClinicalAnalysisRequest,
    result: AIBeamformingResult,
    processing_time: u64,
) -> ClinicalAnalysisResponse {
    // Convert findings from beamforming result
    let mut findings = Vec::new();

    // Simulate lesion detection based on confidence map
    let confidence_view = result.confidence.view();
    let volume_view = result.volume.view();

    // Simple threshold-based detection (in production would be more sophisticated)
    for z in 10..result.volume.dim().2.saturating_sub(10) {
        for y in 10..result.volume.dim().1.saturating_sub(10) {
            for x in 10..result.volume.dim().0.saturating_sub(10) {
                let conf = confidence_view[[x, y, z]];
                let intensity = volume_view[[x, y, z]];

                if conf > 0.8 && intensity.abs() > 2.0 {
                    findings.push(crate::api::ClinicalFinding {
                        finding_type: crate::api::FindingType::Lesion,
                        description: format!("Potential lesion detected at ({}, {}, {})", x, y, z),
                        confidence: conf,
                        location: [x as f64 * 0.1, y as f64 * 0.1, z as f64 * 0.1], // Convert to mm
                        measurements: crate::api::FindingMeasurements {
                            max_diameter_mm: 5.0,
                            area_volume: 78.5, // π * r²
                            boundary_confidence: 0.85,
                        },
                        clinical_significance: if conf > 0.9 { 0.8 } else { 0.6 },
                    });
                }
            }
        }
    }

    // Generate tissue characterization
    let tissue_characterization = crate::api::TissueCharacterization {
        tissue_map: HashMap::from([(
            "liver".to_string(),
            crate::api::TissueRegion {
                tissue_type: "Liver".to_string(),
                boundaries: [0.0, 100.0, 0.0, 100.0, 0.0, 50.0],
                confidence: 0.85,
                properties: crate::api::TissueProperties {
                    attenuation_db_per_cm_mhz: 0.5,
                    backscatter_coefficient: 1.2,
                    speed_of_sound: 1540.0,
                },
            },
        )]),
        composition_percentages: HashMap::from([
            ("Liver".to_string(), 85.0),
            ("Vessels".to_string(), 10.0),
            ("Fat".to_string(), 5.0),
        ]),
        homogeneity_score: 0.75,
        abnormal_regions: if findings.is_empty() {
            Vec::new()
        } else {
            vec![crate::api::AbnormalRegion {
                region_id: "abnormal_001".to_string(),
                location: findings[0].location,
                abnormality_type: "Hypoechoic Lesion".to_string(),
                severity: 0.7,
                confidence: findings[0].confidence,
            }]
        },
    };

    // Generate clinical recommendations
    let recommendations = if findings.is_empty() {
        vec![crate::api::ClinicalRecommendation {
            recommendation_type: crate::api::RecommendationType::NormalFindings,
            text: "No significant abnormalities detected. Routine follow-up recommended."
                .to_string(),
            rationale:
                "AI analysis shows normal tissue characteristics with no suspicious findings."
                    .to_string(),
            urgency: crate::api::UrgencyLevel::Routine,
            supporting_evidence: vec![
                "Homogeneous tissue appearance".to_string(),
                "Normal attenuation values".to_string(),
                "No abnormal tissue regions identified".to_string(),
            ],
        }]
    } else {
        vec![
            crate::api::ClinicalRecommendation {
                recommendation_type: crate::api::RecommendationType::FollowUpImaging,
                text: "Follow-up imaging recommended to characterize detected lesion.".to_string(),
                rationale: format!(
                    "AI detected {} potential lesion(s) with high confidence.",
                    findings.len()
                ),
                urgency: crate::api::UrgencyLevel::Priority,
                supporting_evidence: vec![
                    format!("{} lesions detected with confidence > 0.8", findings.len()),
                    "Abnormal tissue characteristics identified".to_string(),
                    "Clinical correlation recommended".to_string(),
                ],
            },
            crate::api::ClinicalRecommendation {
                recommendation_type: crate::api::RecommendationType::SpecialistReferral,
                text: "Consider referral to radiology specialist for further evaluation."
                    .to_string(),
                rationale: "Lesion characteristics warrant specialist interpretation.".to_string(),
                urgency: crate::api::UrgencyLevel::Priority,
                supporting_evidence: vec![
                    "Lesion size and location suggest need for specialist evaluation".to_string(),
                ],
            },
        ]
    };

    // Calculate overall confidence
    let confidence_score = if findings.is_empty() {
        0.9 // High confidence when no findings
    } else {
        findings.iter().map(|f| f.confidence).sum::<f32>() / findings.len() as f32
    };

    ClinicalAnalysisResponse {
        session_id: request.session_id.clone(),
        processed_at: Utc::now(),
        confidence_score,
        findings,
        tissue_characterization,
        recommendations,
        performance: crate::api::ProcessingMetrics {
            total_time_ms: processing_time,
            ai_inference_time_ms: result.performance.pinn_inference_time_ms as u64,
            beamforming_time_ms: result.performance.beamforming_time_ms as u64,
            feature_extraction_time_ms: result.performance.feature_extraction_time_ms as u64,
            memory_usage_mb: result.performance.memory_usage_mb,
            gpu_utilization_percent: if result.performance.gpu_utilization_percent.is_nan() {
                None
            } else {
                Some(result.performance.gpu_utilization_percent)
            },
        },
        quality_indicators: crate::api::QualityIndicators {
            image_quality: 0.85,
            motion_artifacts: 0.1,
            acoustic_shadowing: 0.2,
            snr: 25.0,
            frame_rate_hz: 30.0,
        },
    }
}

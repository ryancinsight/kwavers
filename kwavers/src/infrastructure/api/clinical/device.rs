use axum::{
    extract::{Json, Path, Query, State},
    http::StatusCode,
    response::Json as JsonResponse,
};
use chrono::Utc;

use crate::infrastructure::api::auth::AuthenticatedUser;
use crate::infrastructure::api::{APIError, PaginationParams, UltrasoundDevice};

use super::state::ClinicalAppState;

/// Register ultrasound device endpoint
pub async fn register_device(
    State(state): State<ClinicalAppState>,
    auth: AuthenticatedUser,
    Json(device): Json<UltrasoundDevice>,
) -> Result<JsonResponse<UltrasoundDevice>, (StatusCode, JsonResponse<APIError>)> {
    let mut registry = state.device_registry.write().await;

    // Update device with registration info
    let mut registered_device = device.clone();
    registered_device.last_seen = Utc::now();

    // Store device
    registry.insert(device.device_id.clone(), registered_device.clone());

    tracing::info!(
        "Device registered: {} ({}) by user {}",
        device.device_id,
        device.model,
        auth.user_id
    );

    Ok(JsonResponse(registered_device))
}

/// Get device status endpoint
pub async fn get_device_status(
    State(state): State<ClinicalAppState>,
    Path(device_id): Path<String>,
    _auth: AuthenticatedUser,
) -> Result<JsonResponse<UltrasoundDevice>, (StatusCode, JsonResponse<APIError>)> {
    let registry = state.device_registry.read().await;

    if let Some(device) = registry.get(&device_id) {
        // Authorization check - production would validate JWT tokens and device ownership
        Ok(JsonResponse(device.clone()))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            JsonResponse(APIError {
                error: crate::api::APIErrorType::ResourceNotFound,
                message: format!("Device '{}' not found", device_id),
                details: None,
            }),
        ))
    }
}

/// List connected devices endpoint
pub async fn list_devices(
    State(state): State<ClinicalAppState>,
    _auth: AuthenticatedUser,
    Query(pagination): Query<PaginationParams>,
) -> Result<JsonResponse<serde_json::Value>, (StatusCode, JsonResponse<APIError>)> {
    let registry = state.device_registry.read().await;

    let devices: Vec<&UltrasoundDevice> = registry.values().collect();

    // Apply pagination
    let page = pagination.page.unwrap_or(1).max(1);
    let page_size = pagination.page_size.unwrap_or(50).min(100);
    let start_idx = (page - 1) * page_size;
    let end_idx = start_idx + page_size;

    let paginated_devices = if start_idx < devices.len() {
        devices[start_idx..end_idx.min(devices.len())].to_vec()
    } else {
        Vec::new()
    };

    let response = serde_json::json!({
        "devices": paginated_devices,
        "total_count": devices.len(),
        "page": page,
        "page_size": page_size
    });

    Ok(JsonResponse(response))
}

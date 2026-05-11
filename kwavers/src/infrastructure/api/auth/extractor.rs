//! Axum extractor implementations for AuthenticatedUser.

use super::{AuthMiddleware, AuthenticatedUser};
use crate::infrastructure::api::{APIError, APIErrorType};
use axum::extract::FromRequestParts;
use axum::http::{header::AUTHORIZATION, request::Parts, StatusCode};

impl AuthenticatedUser {
    /// Extract from middleware.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) async fn extract_from_middleware(
        parts: &mut Parts,
        auth_middleware: &std::sync::Arc<AuthMiddleware>,
    ) -> Result<Self, (StatusCode, axum::Json<APIError>)> {
        let auth_header = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|h| h.to_str().ok())
            .and_then(|h| h.strip_prefix("Bearer "));

        let token = match auth_header {
            Some(token) => token,
            None => {
                return Err((
                    StatusCode::UNAUTHORIZED,
                    axum::Json(APIError {
                        error: APIErrorType::AuthenticationFailed,
                        message: "Missing or invalid Authorization header".to_string(),
                        details: None,
                    }),
                ));
            }
        };

        match auth_middleware.authenticate_jwt(token) {
            Ok(user) => Ok(user),
            Err(error) => Err((StatusCode::UNAUTHORIZED, axum::Json(error))),
        }
    }
}

#[axum::async_trait]
impl FromRequestParts<crate::api::handlers::AppState> for AuthenticatedUser {
    type Rejection = (StatusCode, axum::Json<APIError>);

    async fn from_request_parts(
        parts: &mut Parts,
        state: &crate::api::handlers::AppState,
    ) -> Result<Self, Self::Rejection> {
        Self::extract_from_middleware(parts, &state.auth_middleware).await
    }
}

#[cfg(feature = "pinn")]
#[axum::async_trait]
impl FromRequestParts<crate::api::clinical_handlers::ClinicalAppState> for AuthenticatedUser {
    type Rejection = (StatusCode, axum::Json<APIError>);

    async fn from_request_parts(
        parts: &mut Parts,
        state: &crate::api::clinical_handlers::ClinicalAppState,
    ) -> Result<Self, Self::Rejection> {
        Self::extract_from_middleware(parts, &state.auth_middleware).await
    }
}

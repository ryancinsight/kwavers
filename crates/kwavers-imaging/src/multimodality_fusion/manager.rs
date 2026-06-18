use super::{FusionEngine, FusionParameters, ImageData, RegistrationTransform, TransformationType};
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use ritk_registration::{AffineTransform as RitkAffineTransform, ImageRegistration};
use std::collections::HashMap;

/// Multi-modality imaging session
#[derive(Debug, Clone)]
pub struct MultimodalitySession {
    /// Session identifier
    pub session_id: String,
    /// Reference image (CT or MR)
    pub reference_image: Option<ImageData>,
    /// Floating image (ultrasound, PET, or other)
    pub floating_image: Option<ImageData>,
    /// Registration transformation
    pub transformation: Option<RegistrationTransform>,
    /// Fusion parameters
    pub fusion_params: FusionParameters,
}

/// Multimodality fusion session manager
#[derive(Debug)]
pub struct MultimodalityFusionManager {
    /// Active sessions
    sessions: HashMap<String, MultimodalitySession>,
    /// Registration engine
    registration_engine: ImageRegistration,
    /// Fusion engine
    fusion_engine: FusionEngine,
}

impl MultimodalityFusionManager {
    /// Create new fusion manager
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            registration_engine: ImageRegistration::default(),
            fusion_engine: FusionEngine::new(FusionParameters::default()),
        }
    }

    /// Create new fusion session
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn create_session(&mut self, session_id: String) -> KwaversResult<()> {
        let session = MultimodalitySession {
            session_id: session_id.clone(),
            reference_image: None,
            floating_image: None,
            transformation: None,
            fusion_params: FusionParameters::default(),
        };
        self.sessions.insert(session_id, session);

        Ok(())
    }

    /// Load reference image into session
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn load_reference(&mut self, session_id: &str, image_data: ImageData) -> KwaversResult<()> {
        let session = self.sessions.get_mut(session_id).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Session {} not found", session_id))
        })?;

        session.reference_image = Some(image_data);
        Ok(())
    }

    /// Load floating image into session
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn load_floating(&mut self, session_id: &str, image_data: ImageData) -> KwaversResult<()> {
        let session = self.sessions.get_mut(session_id).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Session {} not found", session_id))
        })?;

        session.floating_image = Some(image_data);
        Ok(())
    }

    /// Register images in session
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn register(&mut self, session_id: &str) -> KwaversResult<()> {
        let session = self.sessions.get_mut(session_id).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Session {} not found", session_id))
        })?;

        let reference = session
            .reference_image
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Reference image not loaded".to_owned()))?;

        let floating = session
            .floating_image
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Floating image not loaded".to_owned()))?;

        let transform_result = self
            .registration_engine
            .rigid_registration_mutual_info(
                &reference.data,
                &floating.data,
                &RitkAffineTransform::IDENTITY,
            )
            .map_err(|e| {
                KwaversError::InvalidInput(format!("RITK Registration failed: {:?}", e))
            })?;

        let matrix =
            ndarray::Array2::from_shape_vec((4, 4), transform_result.transform.as_array().to_vec())
                .map_err(|e| {
                    KwaversError::InvalidInput(format!("RITK Transform generation failed: {:?}", e))
                })?;

        let transform = RegistrationTransform {
            transform_type: TransformationType::Rigid,
            matrix,
            registration_error_mm: 1.0 - transform_result.quality.normalized_cross_correlation,
            iterations: transform_result.quality.iterations,
        };
        session.transformation = Some(transform);

        Ok(())
    }

    /// Perform fusion in session
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn fuse(&self, session_id: &str) -> KwaversResult<Array3<f64>> {
        let session = self.sessions.get(session_id).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Session {} not found", session_id))
        })?;

        let reference = session
            .reference_image
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Reference image not loaded".to_owned()))?;

        let floating = session
            .floating_image
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Floating image not loaded".to_owned()))?;

        let transform = session
            .transformation
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Images not yet registered".to_owned()))?;

        self.fusion_engine.fuse(reference, floating, transform)
    }

    /// Get session
    #[must_use]
    pub fn get_session(&self, session_id: &str) -> Option<&MultimodalitySession> {
        self.sessions.get(session_id)
    }
}

impl Default for MultimodalityFusionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodality_fusion_manager() {
        let mut manager = MultimodalityFusionManager::new();
        manager.create_session("test_session".to_string()).unwrap();
        let session = manager.get_session("test_session").unwrap();
        assert_eq!(session.session_id, "test_session");
    }
}

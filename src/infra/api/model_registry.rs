//! Model registry for PINN model storage and retrieval
//!
//! Provides thread-safe storage and retrieval of trained PINN models
//! with proper serialization, versioning, and metadata management.

use crate::infra::api::{APIError, APIErrorType, ModelMetadata};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Stored model with metadata and serialized data
#[derive(Debug, Clone)]
pub struct StoredModel {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Serialized model data
    pub model_data: Vec<u8>,
    /// Storage timestamp
    pub stored_at: chrono::DateTime<chrono::Utc>,
    /// Model version for optimistic concurrency
    pub version: u64,
}

/// Model registry for storing and retrieving PINN models
#[derive(Clone, Debug)]
pub struct ModelRegistry {
    /// Stored models indexed by model ID
    models: Arc<RwLock<HashMap<String, StoredModel>>>,
    /// User-specific model lists
    user_models: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl ModelRegistry {
    /// Create a new empty model registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            user_models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Store a trained model
    pub fn store_model(
        &self,
        user_id: &str,
        model_id: &str,
        model_data: Vec<u8>,
        metadata: ModelMetadata,
    ) -> Result<(), APIError> {
        let stored_model = StoredModel {
            metadata,
            model_data,
            stored_at: chrono::Utc::now(),
            version: 1,
        };

        // Store the model
        {
            let mut models = self.models.write();
            models.insert(model_id.to_string(), stored_model);
        }

        // Update user's model list
        {
            let mut user_models = self.user_models.write();
            user_models
                .entry(user_id.to_string())
                .or_default()
                .push(model_id.to_string());
        }

        Ok(())
    }

    /// Retrieve a model by ID
    pub fn get_model(&self, model_id: &str) -> Option<StoredModel> {
        let models = self.models.read();
        models.get(model_id).cloned()
    }

    /// Get all models for a user
    pub fn get_user_models(&self, user_id: &str) -> Vec<ModelMetadata> {
        let user_models = self.user_models.read();
        let models = self.models.read();

        if let Some(model_ids) = user_models.get(user_id) {
            model_ids
                .iter()
                .filter_map(|id| models.get(id).map(|m| m.metadata.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Delete a model
    pub fn delete_model(&self, user_id: &str, model_id: &str) -> Result<(), APIError> {
        // Check if model exists and belongs to user
        {
            let models = self.models.read();
            let user_models = self.user_models.read();

            if let Some(model) = models.get(model_id) {
                if model.metadata.model_id != model_id {
                    return Err(APIError {
                        error: APIErrorType::ResourceNotFound,
                        message: "Model not found".to_string(),
                        details: None,
                    });
                }
            } else {
                return Err(APIError {
                    error: APIErrorType::ResourceNotFound,
                    message: "Model not found".to_string(),
                    details: None,
                });
            }

            // Check user ownership
            match user_models.get(user_id) {
                Some(user_model_ids) => {
                    if !user_model_ids.iter().any(|id| id == model_id) {
                        return Err(APIError {
                            error: APIErrorType::AuthorizationFailed,
                            message: "Not authorized to delete this model".to_string(),
                            details: None,
                        });
                    }
                }
                None => {
                    return Err(APIError {
                        error: APIErrorType::AuthorizationFailed,
                        message: "Not authorized to delete this model".to_string(),
                        details: None,
                    });
                }
            }
        }

        // Remove from storage
        {
            let mut models = self.models.write();
            models.remove(model_id);
        }

        // Remove from user's list
        {
            let mut user_models = self.user_models.write();
            if let Some(model_ids) = user_models.get_mut(user_id) {
                model_ids.retain(|id| id != model_id);
            }
        }

        Ok(())
    }

    /// Update model metadata
    pub fn update_metadata(
        &self,
        model_id: &str,
        new_metadata: ModelMetadata,
    ) -> Result<(), APIError> {
        let mut models = self.models.write();
        if let Some(model) = models.get_mut(model_id) {
            model.metadata = new_metadata;
            model.version += 1;
            Ok(())
        } else {
            Err(APIError {
                error: APIErrorType::ResourceNotFound,
                message: "Model not found".to_string(),
                details: None,
            })
        }
    }

    /// Get model statistics
    #[must_use]
    pub fn get_stats(&self) -> ModelRegistryStats {
        let models = self.models.read();
        let user_models = self.user_models.read();

        ModelRegistryStats {
            total_models: models.len(),
            total_users: user_models.len(),
            total_storage_bytes: models.values().map(|m| m.model_data.len()).sum(),
        }
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct ModelRegistryStats {
    /// Total number of stored models
    pub total_models: usize,
    /// Total number of users with models
    pub total_users: usize,
    /// Total storage used in bytes
    pub total_storage_bytes: usize,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infra::api::TrainingConfig;

    #[test]
    fn test_model_storage_and_retrieval() {
        let registry = ModelRegistry::new();

        let model_data = vec![1, 2, 3, 4, 5];
        let metadata = ModelMetadata {
            model_id: "test_model".to_string(),
            physics_domain: "acoustic_wave".to_string(),
            created_at: chrono::Utc::now(),
            training_config: TrainingConfig::default(),
            performance_metrics: crate::api::TrainingMetrics {
                final_loss: 0.001,
                best_loss: 0.0008,
                total_epochs: 100,
                training_time_seconds: 3600,
                convergence_epoch: Some(80),
                final_validation_error: Some(0.0012),
            },
            geometry_spec: crate::api::GeometrySpec {
                bounds: vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                obstacles: vec![],
                boundary_conditions: vec![],
            },
        };

        // Store model
        assert!(registry
            .store_model(
                "user123",
                "test_model",
                model_data.clone(),
                metadata.clone()
            )
            .is_ok());

        // Retrieve model
        let stored = registry.get_model("test_model").unwrap();
        assert_eq!(stored.metadata.model_id, "test_model");
        assert_eq!(stored.model_data, model_data);

        // Get user models
        let user_models = registry.get_user_models("user123");
        assert_eq!(user_models.len(), 1);
        assert_eq!(user_models[0].model_id, "test_model");
    }

    #[test]
    fn test_model_deletion() {
        let registry = ModelRegistry::new();

        let metadata = ModelMetadata {
            model_id: "test_model".to_string(),
            physics_domain: "test".to_string(),
            created_at: chrono::Utc::now(),
            training_config: TrainingConfig::default(),
            performance_metrics: crate::api::TrainingMetrics::default(),
            geometry_spec: crate::api::GeometrySpec::default(),
        };

        // Store and then delete
        assert!(registry
            .store_model("user123", "test_model", vec![1, 2, 3], metadata)
            .is_ok());
        assert!(registry.delete_model("user123", "test_model").is_ok());

        // Should not exist anymore
        assert!(registry.get_model("test_model").is_none());
        assert_eq!(registry.get_user_models("user123").len(), 0);
    }

    #[test]
    fn test_unauthorized_access() {
        let registry = ModelRegistry::new();

        let metadata = ModelMetadata {
            model_id: "test_model".to_string(),
            physics_domain: "test".to_string(),
            created_at: chrono::Utc::now(),
            training_config: TrainingConfig::default(),
            performance_metrics: crate::api::TrainingMetrics::default(),
            geometry_spec: crate::api::GeometrySpec::default(),
        };

        assert!(registry
            .store_model("user123", "test_model", vec![1, 2, 3], metadata)
            .is_ok());

        // Different user should not be able to delete
        assert!(registry.delete_model("user456", "test_model").is_err());
    }

    #[test]
    fn test_registry_stats() {
        let registry = ModelRegistry::new();

        let metadata = ModelMetadata {
            model_id: "test_model".to_string(),
            physics_domain: "test".to_string(),
            created_at: chrono::Utc::now(),
            training_config: TrainingConfig::default(),
            performance_metrics: crate::api::TrainingMetrics::default(),
            geometry_spec: crate::api::GeometrySpec::default(),
        };

        assert!(registry
            .store_model("user123", "test_model", vec![1, 2, 3, 4, 5], metadata)
            .is_ok());

        let stats = registry.get_stats();
        assert_eq!(stats.total_models, 1);
        assert_eq!(stats.total_users, 1);
        assert_eq!(stats.total_storage_bytes, 5);
    }
}

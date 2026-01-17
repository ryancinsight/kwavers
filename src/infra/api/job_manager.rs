//! Job management system for PINN training operations
//!
//! Provides asynchronous job queuing, execution, and monitoring for PINN training tasks.
//! Implements proper concurrency controls and resource management.

use crate::infra::api::{
    APIError, APIErrorType, JobStatus, PINNTrainingRequest, TrainingMetrics, TrainingProgress,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;

#[cfg(feature = "pinn")]
async fn execute_training_with_progress(
    request: &PINNTrainingRequest,
    progress_sender: mpsc::Sender<TrainingProgress>,
) -> Result<crate::analysis::ml::pinn::trainer::TrainingResult, APIError> {
    use crate::analysis::ml::pinn::trainer::{
        BoundaryCondition, Geometry, Obstacle, PINNConfig, PINNTrainer, PhysicsParams,
        TrainingConfig,
    };

    let geometry = Geometry {
        bounds: request.geometry.bounds.clone(),
        obstacles: request
            .geometry
            .obstacles
            .iter()
            .map(|o| Obstacle {
                shape: o.shape.clone(),
                center: o.center.clone(),
                parameters: o.parameters.clone(),
            })
            .collect(),
        boundary_conditions: request
            .geometry
            .boundary_conditions
            .iter()
            .map(|bc| BoundaryCondition {
                boundary: bc.boundary.clone(),
                condition_type: bc.condition_type.clone(),
                value: bc.value,
            })
            .collect(),
    };

    let physics_params = PhysicsParams {
        material_properties: request.physics_params.material_properties.clone(),
        boundary_values: request.physics_params.boundary_values.clone(),
        initial_values: request.physics_params.initial_values.clone(),
        domain_params: request.physics_params.domain_params.clone(),
    };

    let training_config = TrainingConfig {
        collocation_points: request.training_config.collocation_points,
        batch_size: request.training_config.batch_size,
        epochs: request.training_config.epochs,
        learning_rate: request.training_config.learning_rate,
        hidden_layers: request.training_config.hidden_layers.clone(),
        adaptive_sampling: request.training_config.adaptive_sampling,
        use_gpu: request.training_config.use_gpu,
    };

    let pinn_config = PINNConfig {
        physics_domain: request.physics_domain.clone(),
        geometry,
        physics_params,
        training_config,
        use_gpu: request.training_config.use_gpu,
    };

    let mut trainer = PINNTrainer::new(pinn_config).map_err(|e| APIError {
        error: APIErrorType::InternalError,
        message: e.to_string(),
        details: None,
    })?;

    trainer
        .train_with_progress(progress_sender)
        .await
        .map_err(|e| APIError {
            error: APIErrorType::InternalError,
            message: e.to_string(),
            details: None,
        })
}

/// Training job state
#[derive(Debug, Clone)]
pub struct TrainingJob {
    /// Unique job identifier
    pub id: String,
    /// User who submitted the job
    pub user_id: String,
    /// Training request parameters
    pub request: PINNTrainingRequest,
    /// Current job status
    pub status: JobStatus,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Start timestamp
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Completion timestamp
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Current training progress
    pub progress: Option<TrainingProgress>,
    /// Final training result
    pub result: Option<TrainingResult>,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Training result data
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final trained model
    pub model: Vec<u8>, // Serialized model data
    /// Training metrics
    pub metrics: TrainingMetrics,
    /// Model metadata
    pub model_metadata: crate::api::ModelMetadata,
}

/// Stub training result for when PINN is not available
#[cfg(not(feature = "pinn"))]
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Placeholder for non-PINN builds
    pub placeholder: (),
}

/// Job manager for coordinating PINN training tasks
#[derive(Clone, Debug)]
pub struct JobManager {
    /// Active and completed jobs
    jobs: Arc<RwLock<HashMap<String, TrainingJob>>>,
    /// Job queue sender
    job_sender: mpsc::UnboundedSender<String>,
    /// Maximum concurrent jobs
    max_concurrent_jobs: usize,
    /// Active job count
    active_jobs: Arc<RwLock<usize>>,
}

impl JobManager {
    /// Create a new job manager
    #[must_use]
    pub fn new(max_concurrent_jobs: usize) -> Self {
        let (job_sender, job_receiver) = mpsc::unbounded_channel();

        let manager = Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            job_sender,
            max_concurrent_jobs,
            active_jobs: Arc::new(RwLock::new(0)),
        };

        // Start job processing task
        tokio::spawn(manager.clone().process_jobs(job_receiver));

        manager
    }

    /// Submit a new training job
    pub async fn submit_job(
        &self,
        user_id: &str,
        request: PINNTrainingRequest,
    ) -> Result<String, APIError> {
        let job_id = format!("pinn_job_{}", uuid::Uuid::new_v4().simple());

        let job = TrainingJob {
            id: job_id.clone(),
            user_id: user_id.to_string(),
            request,
            status: JobStatus::Queued,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            progress: None,
            result: None,
            error_message: None,
        };

        // Check concurrent job limit
        {
            let active_count = *self.active_jobs.read();
            if active_count >= self.max_concurrent_jobs {
                return Err(APIError {
                    error: APIErrorType::InternalError,
                    message: "Too many concurrent training jobs. Please try again later."
                        .to_string(),
                    details: Some(HashMap::from([
                        (
                            "max_concurrent_jobs".to_string(),
                            serde_json::json!(self.max_concurrent_jobs),
                        ),
                        (
                            "current_active_jobs".to_string(),
                            serde_json::json!(active_count),
                        ),
                    ])),
                });
            }
        }

        // Store job
        {
            let mut jobs = self.jobs.write();
            jobs.insert(job_id.clone(), job);
        }

        // Queue job for processing
        self.job_sender.send(job_id.clone()).map_err(|_| APIError {
            error: APIErrorType::InternalError,
            message: "Failed to queue training job".to_string(),
            details: None,
        })?;

        Ok(job_id)
    }

    /// Get job information
    pub fn get_job(&self, job_id: &str) -> Option<TrainingJob> {
        let jobs = self.jobs.read();
        jobs.get(job_id).cloned()
    }

    /// Get all jobs for a user
    pub fn get_user_jobs(&self, user_id: &str) -> Vec<TrainingJob> {
        let jobs = self.jobs.read();
        jobs.values()
            .filter(|job| job.user_id == user_id)
            .cloned()
            .collect()
    }

    /// Cancel a job (if not already running)
    pub fn cancel_job(&self, job_id: &str, user_id: &str) -> Result<(), APIError> {
        let mut jobs = self.jobs.write();
        if let Some(job) = jobs.get_mut(job_id) {
            if job.user_id != user_id {
                return Err(APIError {
                    error: APIErrorType::AuthorizationFailed,
                    message: "Not authorized to cancel this job".to_string(),
                    details: None,
                });
            }

            if job.status == JobStatus::Queued {
                job.status = JobStatus::Cancelled;
                job.completed_at = Some(chrono::Utc::now());
                Ok(())
            } else {
                Err(APIError {
                    error: APIErrorType::InvalidRequest,
                    message: "Job cannot be cancelled (already running or completed)".to_string(),
                    details: None,
                })
            }
        } else {
            Err(APIError {
                error: APIErrorType::ResourceNotFound,
                message: "Job not found".to_string(),
                details: None,
            })
        }
    }

    /// Process jobs from the queue
    async fn process_jobs(self, mut receiver: mpsc::UnboundedReceiver<String>) {
        while let Some(job_id) = receiver.recv().await {
            // Check if we can start another job
            {
                let active_count = *self.active_jobs.read();
                if active_count >= self.max_concurrent_jobs {
                    // Re-queue the job
                    let _ = self.job_sender.send(job_id);
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                    continue;
                }
            }

            // Start job processing
            let manager_clone = self.clone();
            tokio::spawn(async move {
                manager_clone.process_single_job(&job_id).await;
            });
        }
    }

    /// Process a single training job
    async fn process_single_job(&self, job_id: &str) {
        // Mark job as running
        {
            let mut jobs = self.jobs.write();
            if let Some(job) = jobs.get_mut(job_id) {
                job.status = JobStatus::Running;
                job.started_at = Some(chrono::Utc::now());
                *self.active_jobs.write() += 1;
            }
        }

        // Execute training
        let result = self.execute_training(job_id).await;

        // Update job with result
        {
            let mut jobs = self.jobs.write();
            if let Some(job) = jobs.get_mut(job_id) {
                job.completed_at = Some(chrono::Utc::now());

                match result {
                    Ok(training_result) => {
                        job.status = JobStatus::Completed;
                        job.result = Some(training_result);
                    }
                    Err(error) => {
                        job.status = JobStatus::Failed;
                        job.error_message = Some(error.message);
                    }
                }

                *self.active_jobs.write() -= 1;
            }
        }
    }

    /// Execute the actual PINN training
    #[cfg(feature = "pinn")]
    async fn execute_training(&self, job_id: &str) -> Result<TrainingResult, APIError> {
        let job = {
            let jobs = self.jobs.read();
            jobs.get(job_id).cloned().ok_or_else(|| APIError {
                error: APIErrorType::InternalError,
                message: "Job not found during execution".to_string(),
                details: None,
            })?
        };

        // Execute training with progress updates
        let progress_sender = {
            let jobs = self.jobs.clone();
            let job_id = job_id.to_string();

            // Create a channel for progress updates
            let (tx, mut rx) = mpsc::channel(100);

            // Spawn progress update handler
            tokio::spawn(async move {
                while let Some(progress) = rx.recv().await {
                    let mut jobs = jobs.write();
                    if let Some(job) = jobs.get_mut(&job_id) {
                        job.progress = Some(progress);
                    }
                }
            });

            tx
        };

        let training_result = execute_training_with_progress(&job.request, progress_sender).await?;

        // Create model metadata
        let model_metadata = crate::api::ModelMetadata {
            model_id: format!("model_{}", uuid::Uuid::new_v4().simple()),
            physics_domain: job.request.physics_domain.clone(),
            created_at: chrono::Utc::now(),
            training_config: job.request.training_config.clone(),
            performance_metrics: training_result.metrics.clone().into(),
            geometry_spec: job.request.geometry.clone(),
        };

        // Serialize model (interface-level payload for now)
        let model_data = serde_json::to_vec(&training_result).map_err(|e| APIError {
            error: APIErrorType::InternalError,
            message: format!("Failed to serialize model: {}", e),
            details: None,
        })?;

        Ok(TrainingResult {
            model: model_data,
            metrics: training_result.metrics.into(),
            model_metadata,
        })
    }

    /// Stub implementation when PINN feature is not available
    #[cfg(not(feature = "pinn"))]
    async fn execute_training(&self, _job_id: &str) -> Result<TrainingResult, APIError> {
        Err(APIError {
            error: APIErrorType::InternalError,
            message: "PINN training not available - feature not enabled".to_string(),
            details: None,
        })
    }
}

impl Default for JobManager {
    fn default() -> Self {
        Self::new(5) // Default to 5 concurrent jobs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infra::api::{GeometrySpec, PhysicsParameters, TrainingConfig};

    #[tokio::test]
    async fn test_job_submission() {
        let manager = JobManager::new(2);

        let request = PINNTrainingRequest {
            physics_domain: "acoustic_wave".to_string(),
            geometry: GeometrySpec {
                bounds: vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                obstacles: vec![],
                boundary_conditions: vec![],
            },
            physics_params: PhysicsParameters {
                material_properties: HashMap::new(),
                boundary_values: HashMap::new(),
                initial_values: HashMap::new(),
                domain_params: HashMap::new(),
            },
            training_config: TrainingConfig::default(),
            callback_url: None,
            metadata: None,
        };

        let job_id = manager.submit_job("user123", request).await.unwrap();
        assert!(job_id.starts_with("pinn_job_"));

        let job = manager.get_job(&job_id).unwrap();
        assert_eq!(job.status, JobStatus::Queued);
        assert_eq!(job.user_id, "user123");
    }

    #[tokio::test]
    async fn test_job_cancellation() {
        let manager = JobManager::new(2);

        let request = PINNTrainingRequest {
            physics_domain: "acoustic_wave".to_string(),
            geometry: GeometrySpec {
                bounds: vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                obstacles: vec![],
                boundary_conditions: vec![],
            },
            physics_params: PhysicsParameters {
                material_properties: HashMap::new(),
                boundary_values: HashMap::new(),
                initial_values: HashMap::new(),
                domain_params: HashMap::new(),
            },
            training_config: TrainingConfig::default(),
            callback_url: None,
            metadata: None,
        };

        let job_id = manager.submit_job("user123", request).await.unwrap();

        // Should be able to cancel queued job
        assert!(manager.cancel_job(&job_id, "user123").is_ok());

        let job = manager.get_job(&job_id).unwrap();
        assert_eq!(job.status, JobStatus::Cancelled);
    }

    #[test]
    fn test_user_jobs_filtering() {
        let manager = JobManager::new(2);

        // Add a job manually for testing
        let job = TrainingJob {
            id: "test_job".to_string(),
            user_id: "user123".to_string(),
            request: PINNTrainingRequest {
                physics_domain: "test".to_string(),
                geometry: GeometrySpec::default(),
                physics_params: PhysicsParameters::default(),
                training_config: TrainingConfig::default(),
                callback_url: None,
                metadata: None,
            },
            status: JobStatus::Queued,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            progress: None,
            result: None,
            error_message: None,
        };

        {
            let mut jobs = manager.jobs.write();
            jobs.insert("test_job".to_string(), job);
        }

        let user_jobs = manager.get_user_jobs("user123");
        assert_eq!(user_jobs.len(), 1);
        assert_eq!(user_jobs[0].id, "test_job");

        let other_user_jobs = manager.get_user_jobs("other_user");
        assert_eq!(other_user_jobs.len(), 0);
    }
}

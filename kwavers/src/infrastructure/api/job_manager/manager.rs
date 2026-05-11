//! JobManager implementation.

use crate::infrastructure::api::{APIError, APIErrorType, PINNTrainingRequest};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;

use super::types::{TrainingExecutor, TrainingJob, TrainingResult};
use crate::infrastructure::api::JobStatus;

/// Job manager for coordinating PINN training tasks
#[derive(Clone, Debug)]
pub struct JobManager {
    /// Active and completed jobs
    pub(super) jobs: Arc<RwLock<HashMap<String, TrainingJob>>>,
    /// Job queue sender
    job_sender: mpsc::UnboundedSender<String>,
    /// Maximum concurrent jobs
    max_concurrent_jobs: usize,
    /// Active job count
    active_jobs: Arc<RwLock<usize>>,
    training_executor: Arc<dyn TrainingExecutor>,
}

impl JobManager {
    /// Create a new job manager
    #[must_use]
    pub fn new(max_concurrent_jobs: usize, training_executor: Arc<dyn TrainingExecutor>) -> Self {
        let (job_sender, job_receiver) = mpsc::unbounded_channel();

        let manager = Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            job_sender,
            max_concurrent_jobs,
            active_jobs: Arc::new(RwLock::new(0)),
            training_executor,
        };

        // Start job processing task
        tokio::spawn(manager.clone().process_jobs(job_receiver));

        manager
    }

    /// Submit a new training job
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn get_job(&self, job_id: &str) -> Option<TrainingJob> {
        let jobs = self.jobs.read();
        jobs.get(job_id).cloned()
    }

    /// Get all jobs for a user
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn get_user_jobs(&self, user_id: &str) -> Vec<TrainingJob> {
        let jobs = self.jobs.read();
        jobs.values()
            .filter(|job| job.user_id == user_id)
            .cloned()
            .collect()
    }

    /// Cancel a job (if not already running)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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

        let training_output = self
            .training_executor
            .execute(job.request.clone(), progress_sender)
            .await?;

        // Create model metadata
        let model_metadata = crate::api::ModelMetadata {
            model_id: format!("model_{}", uuid::Uuid::new_v4().simple()),
            physics_domain: job.request.physics_domain.clone(),
            created_at: chrono::Utc::now(),
            training_config: job.request.training_config.clone(),
            performance_metrics: training_output.metrics.clone(),
            geometry_spec: job.request.geometry.clone(),
        };

        Ok(TrainingResult {
            model: training_output.model,
            metrics: training_output.metrics,
            model_metadata,
        })
    }

    /// Stub implementation when PINN feature is not available
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[cfg(not(feature = "pinn"))]
    async fn execute_training(&self, _job_id: &str) -> Result<TrainingResult, APIError> {
        Err(APIError {
            error: APIErrorType::InternalError,
            message: "PINN training not available - feature not enabled".to_string(),
            details: None,
        })
    }
}

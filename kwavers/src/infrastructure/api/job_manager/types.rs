//! Types for job management.

use crate::infrastructure::api::{
    APIError, JobStatus, PINNTrainingRequest, PinnApiTrainingMetrics, TrainingProgress,
};
use std::future::Future;
use std::pin::Pin;
use tokio::sync::mpsc;

#[derive(Debug)]
pub struct TrainingOutput {
    pub model: Vec<u8>,
    pub metrics: PinnApiTrainingMetrics,
}

pub type TrainingFuture = Pin<Box<dyn Future<Output = Result<TrainingOutput, APIError>> + Send>>;

pub trait TrainingExecutor: Send + Sync + std::fmt::Debug {
    fn execute(
        &self,
        request: PINNTrainingRequest,
        progress_sender: mpsc::Sender<TrainingProgress>,
    ) -> TrainingFuture;
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
    pub result: Option<JobManagerTrainingResult>,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Training result data
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct JobManagerTrainingResult {
    /// Final trained model
    pub model: Vec<u8>, // Serialized model data
    /// Training metrics
    pub metrics: PinnApiTrainingMetrics,
    /// Model metadata
    pub model_metadata: crate::api::ModelMetadata,
}

/// Stub training result for when PINN is not available.
///
/// This is a unit struct since PINN training always returns an error
/// when the `pinn` feature is disabled.
#[cfg(not(feature = "pinn"))]
#[derive(Debug, Clone)]
pub struct JobManagerTrainingResult;

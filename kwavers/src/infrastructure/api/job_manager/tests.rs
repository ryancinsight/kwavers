use super::manager::JobManager;
use super::types::{TrainingExecutor, TrainingFuture, TrainingOutput};
use crate::infrastructure::api::{
    APIError, APIErrorType, GeometrySpec, JobStatus, PINNTrainingRequest, PhysicsParameters,
    TrainingConfig, TrainingProgress,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Debug)]
struct TestTrainingExecutor;

impl TrainingExecutor for TestTrainingExecutor {
    fn execute(
        &self,
        _request: PINNTrainingRequest,
        _progress_sender: mpsc::Sender<TrainingProgress>,
    ) -> TrainingFuture {
        Box::pin(async move {
            Err(APIError {
                error: APIErrorType::InternalError,
                message: "Test training executor invoked".to_string(),
                details: None,
            })
        })
    }
}

#[tokio::test]
async fn test_job_submission() {
    let manager = JobManager::new(2, Arc::new(TestTrainingExecutor));

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
    let manager = JobManager::new(2, Arc::new(TestTrainingExecutor));

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
    use super::types::TrainingJob;

    let manager = JobManager::new(2, Arc::new(TestTrainingExecutor));

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

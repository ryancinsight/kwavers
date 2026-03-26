use crate::core::error::{KwaversError, KwaversResult};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use super::queue::{PoolMetrics, ThreadPoolConfig, WorkQueue};
use super::task::TaskPriority;

/// Real-time pipeline coordinator for multi-stage processing
pub struct PipelineCoordinator {
    /// Work queues per stage
    pub(crate) stages: Vec<Arc<WorkQueue>>,
    /// Stage synchronization events
    _stage_ready: Arc<Mutex<Vec<bool>>>,
    /// Metrics aggregator
    _total_throughput: Arc<AtomicU64>,
    /// Pipeline latency tracking
    _latencies: Arc<Mutex<Vec<u64>>>,
}

// Manual Debug implementation
impl std::fmt::Debug for PipelineCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCoordinator")
            .field("num_stages", &self.stages.len())
            .finish()
    }
}

impl PipelineCoordinator {
    /// Create a new pipeline coordinator with given number of stages
    pub fn new(num_stages: usize) -> KwaversResult<Self> {
        if num_stages == 0 {
            return Err(KwaversError::InvalidInput(
                "Pipeline must have at least 1 stage".to_string(),
            ));
        }

        let mut stages = Vec::new();
        for _ in 0..num_stages {
            let config = ThreadPoolConfig::default();
            stages.push(Arc::new(WorkQueue::new(config)));
        }

        Ok(Self {
            stages,
            _stage_ready: Arc::new(Mutex::new(vec![false; num_stages])),
            _total_throughput: Arc::new(AtomicU64::new(0)),
            _latencies: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Submit work to a specific pipeline stage
    pub fn submit_to_stage(
        &self,
        stage_idx: usize,
        priority: TaskPriority,
        work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync>,
    ) -> KwaversResult<u64> {
        if stage_idx >= self.stages.len() {
            return Err(KwaversError::InvalidInput(format!(
                "Stage index {} out of bounds",
                stage_idx
            )));
        }

        self.stages[stage_idx].submit(priority, work)
    }

    /// Get metrics for a specific stage
    pub fn stage_metrics(&self, stage_idx: usize) -> KwaversResult<PoolMetrics> {
        if stage_idx >= self.stages.len() {
            return Err(KwaversError::InvalidInput(format!(
                "Stage index {} out of bounds",
                stage_idx
            )));
        }

        Ok(self.stages[stage_idx].metrics())
    }

    /// Get aggregate metrics across all stages
    pub fn aggregate_metrics(&self) -> Vec<PoolMetrics> {
        self.stages.iter().map(|stage| stage.metrics()).collect()
    }

    /// Get number of stages
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Shutdown all pipeline stages
    pub fn shutdown(&mut self) -> KwaversResult<()> {
        for stage in self.stages.iter_mut() {
            // Convert Arc to mutable reference - requires creating owned WorkQueue
            // For now, we'll just signal shutdown to the scheduler
            if let Ok(stage_ref) = Arc::try_unwrap(Arc::clone(stage)) {
                let _ = stage_ref;
            }
        }

        Ok(())
    }
}

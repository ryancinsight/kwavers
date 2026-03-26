use crate::core::error::{KwaversError, KwaversResult};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use super::task::{TaskMetrics, TaskPriority, WorkItem};

/// Get current Unix timestamp in milliseconds
pub(crate) fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Real-time task scheduler
#[derive(Debug)]
pub struct RealTimeScheduler {
    /// Queue of pending tasks (using priority ordering)
    queue: Arc<Mutex<Vec<WorkItem>>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Total tasks submitted
    pub(crate) submitted: Arc<AtomicU64>,
    /// Total tasks completed
    completed: Arc<AtomicU64>,
    /// Total tasks failed
    failed: Arc<AtomicU64>,
    /// Total execution time (ms)
    total_execution_time: Arc<AtomicU64>,
    /// Peak queue depth
    peak_queue_depth: Arc<AtomicU64>,
    /// Total wait time (ms)
    total_wait_time: Arc<AtomicU64>,
}

impl RealTimeScheduler {
    /// Create a new real-time scheduler
    pub fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(Vec::new())),
            shutdown: Arc::new(AtomicBool::new(false)),
            submitted: Arc::new(AtomicU64::new(0)),
            completed: Arc::new(AtomicU64::new(0)),
            failed: Arc::new(AtomicU64::new(0)),
            total_execution_time: Arc::new(AtomicU64::new(0)),
            peak_queue_depth: Arc::new(AtomicU64::new(0)),
            total_wait_time: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Submit a task to the scheduler
    pub fn submit(
        &self,
        priority: TaskPriority,
        work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync>,
    ) -> KwaversResult<u64> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(KwaversError::InvalidInput(
                "Scheduler is shutting down".to_string(),
            ));
        }

        let task_id = self.submitted.fetch_add(1, Ordering::SeqCst);
        let item = WorkItem::new(task_id, priority, work, current_timestamp());

        let mut queue = self.queue.lock().unwrap();
        queue.push(item);

        // Sort by priority (highest first)
        queue.sort_by_key(|item| std::cmp::Reverse(item.priority));

        // Update peak queue depth
        let current_depth = queue.len() as u64;
        let peak = self.peak_queue_depth.load(Ordering::Relaxed);
        if current_depth > peak {
            self.peak_queue_depth
                .store(current_depth, Ordering::Relaxed);
        }

        Ok(task_id)
    }

    /// Add a pre-configured work item
    pub fn add_item(&self, item: WorkItem) {
        let mut queue = self.queue.lock().unwrap();
        queue.push(item);
        queue.sort_by_key(|item| std::cmp::Reverse(item.priority));

        // Update peak queue depth
        let current_depth = queue.len() as u64;
        let peak = self.peak_queue_depth.load(Ordering::Relaxed);
        if current_depth > peak {
            self.peak_queue_depth
                .store(current_depth, Ordering::Relaxed);
        }
    }

    /// Get next pending task
    pub fn next_task(&self) -> Option<WorkItem> {
        let mut queue = self.queue.lock().unwrap();
        if queue.is_empty() {
            None
        } else {
            let item = queue.remove(0);
            Some(item)
        }
    }

    /// Execute a task and record metrics
    pub fn execute_task(&self, item: WorkItem) -> KwaversResult<()> {
        let start = Instant::now();
        let wait_time = item.age_ms(current_timestamp());

        // Execute work
        let result = (item.work)();

        let execution_time = start.elapsed().as_millis() as u64;

        // Update metrics
        self.total_execution_time
            .fetch_add(execution_time, Ordering::Relaxed);
        self.total_wait_time.fetch_add(wait_time, Ordering::Relaxed);

        match result {
            Ok(()) => {
                self.completed.fetch_add(1, Ordering::SeqCst);
            }
            Err(_) => {
                self.failed.fetch_add(1, Ordering::SeqCst);
            }
        }

        result
    }

    /// Get current metrics
    pub fn metrics(&self) -> TaskMetrics {
        let submitted = self.submitted.load(Ordering::Relaxed);
        let completed = self.completed.load(Ordering::Relaxed);
        let failed = self.failed.load(Ordering::Relaxed);
        let total_exec = self.total_execution_time.load(Ordering::Relaxed);
        let total_wait = self.total_wait_time.load(Ordering::Relaxed);
        let peak_queue = self.peak_queue_depth.load(Ordering::Relaxed);

        let queue = self.queue.lock().unwrap();
        let current_queue = queue.len() as u64;

        let total_done = completed + failed;
        let avg_exec = if total_done > 0 {
            total_exec as f64 / total_done as f64
        } else {
            0.0
        };

        let avg_wait = if total_done > 0 {
            total_wait as f64 / total_done as f64
        } else {
            0.0
        };

        TaskMetrics {
            total_submitted: submitted,
            total_completed: completed,
            total_failed: failed,
            avg_execution_time_ms: avg_exec,
            peak_queue_depth: peak_queue,
            current_queue_depth: current_queue,
            avg_wait_time_ms: avg_wait,
        }
    }

    /// Shutdown the scheduler
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Check if scheduler is shutdown
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// Clear all pending tasks
    pub fn clear(&self) {
        let mut queue = self.queue.lock().unwrap();
        queue.clear();
    }

    /// Get queue depth
    pub fn queue_depth(&self) -> usize {
        let queue = self.queue.lock().unwrap();
        queue.len()
    }
}

impl Default for RealTimeScheduler {
    fn default() -> Self {
        Self::new()
    }
}

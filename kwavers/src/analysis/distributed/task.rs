use crate::core::error::KwaversResult;
use std::sync::Arc;

/// Task priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Low priority, can be deferred
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// High priority, should run soon
    High = 2,
    /// Critical priority, must run immediately
    Critical = 3,
}

impl TaskPriority {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Normal => "Normal",
            Self::High => "High",
            Self::Critical => "Critical",
        }
    }
}

/// Work item to be processed by the thread pool
#[derive(Clone)]
pub struct WorkItem {
    /// Unique task ID
    pub task_id: u64,
    /// Task priority
    pub priority: TaskPriority,
    /// Task work function
    pub work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync>,
    /// Timestamp when task was queued
    pub queued_time: u64,
    /// Optional deadline for task completion
    pub deadline: Option<u64>,
}

// Manual Debug implementation since dyn Fn is not Debug
impl std::fmt::Debug for WorkItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkItem")
            .field("task_id", &self.task_id)
            .field("priority", &self.priority)
            .field("queued_time", &self.queued_time)
            .field("deadline", &self.deadline)
            .field("work", &"<function>")
            .finish()
    }
}

impl WorkItem {
    /// Create a new work item
    pub fn new(
        task_id: u64,
        priority: TaskPriority,
        work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync>,
        queued_time: u64,
    ) -> Self {
        Self {
            task_id,
            priority,
            work,
            queued_time,
            deadline: None,
        }
    }

    /// Set a deadline for the task
    pub fn with_deadline(mut self, deadline_ms: u64, current_timestamp: u64) -> Self {
        self.deadline = Some(current_timestamp + deadline_ms);
        self
    }

    /// Check if task has exceeded deadline
    pub fn is_overdue(&self, current_timestamp: u64) -> bool {
        if let Some(deadline) = self.deadline {
            current_timestamp > deadline
        } else {
            false
        }
    }

    /// Get task age in milliseconds
    pub fn age_ms(&self, current_timestamp: u64) -> u64 {
        current_timestamp.saturating_sub(self.queued_time)
    }
}

/// Task execution metrics
#[derive(Debug, Clone)]
pub struct TaskMetrics {
    /// Total tasks submitted
    pub total_submitted: u64,
    /// Total tasks completed
    pub total_completed: u64,
    /// Total tasks failed
    pub total_failed: u64,
    /// Average task execution time (ms)
    pub avg_execution_time_ms: f64,
    /// Peak queue depth
    pub peak_queue_depth: u64,
    /// Current queue depth
    pub current_queue_depth: u64,
    /// Average wait time (ms)
    pub avg_wait_time_ms: f64,
}

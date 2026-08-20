use kwavers_core::error::{KwaversError, KwaversResult};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
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
    /// Pending tasks and the task claims held by workers.
    state: Mutex<SchedulerState>,
    /// Notifies workers and waiters when scheduler state changes.
    state_changed: Condvar,
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

#[derive(Debug, Default)]
struct SchedulerState {
    queue: Vec<WorkItem>,
    active_tasks: HashSet<u64>,
}

struct TaskCompletionGuard<'scheduler> {
    scheduler: &'scheduler RealTimeScheduler,
    task_id: u64,
}

impl Drop for TaskCompletionGuard<'_> {
    fn drop(&mut self) {
        self.scheduler.finish_task(self.task_id);
    }
}

impl RealTimeScheduler {
    /// Create a new real-time scheduler
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(SchedulerState::default()),
            state_changed: Condvar::new(),
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
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn submit(
        &self,
        priority: TaskPriority,
        work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync>,
    ) -> KwaversResult<u64> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(KwaversError::InvalidInput(
                "Scheduler is shutting down".to_owned(),
            ));
        }

        let task_id = self.submitted.fetch_add(1, Ordering::SeqCst);
        let item = WorkItem::new(task_id, priority, work, current_timestamp());

        self.add_item(item);

        Ok(task_id)
    }

    /// Add a pre-configured work item
    pub fn add_item(&self, item: WorkItem) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.queue.push(item);
        state
            .queue
            .sort_by_key(|item| std::cmp::Reverse(item.priority));

        // Update peak queue depth
        let current_depth = state.queue.len() as u64;
        let peak = self.peak_queue_depth.load(Ordering::Relaxed);
        if current_depth > peak {
            self.peak_queue_depth
                .store(current_depth, Ordering::Relaxed);
        }
        drop(state);
        self.state_changed.notify_all();
    }

    /// Get next pending task
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn next_task(&self) -> Option<WorkItem> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.queue.is_empty() {
            None
        } else {
            let item = state.queue.remove(0);
            state.active_tasks.insert(item.task_id);
            Some(item)
        }
    }

    /// Wait for a pending task or for shutdown when no task remains.
    pub(crate) fn wait_for_task(&self) -> Option<WorkItem> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if !state.queue.is_empty() {
                let item = state.queue.remove(0);
                state.active_tasks.insert(item.task_id);
                return Some(item);
            }

            if self.is_shutdown() {
                return None;
            }

            state = self
                .state_changed
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Execute a task and record metrics
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn execute_task(&self, item: WorkItem) -> KwaversResult<()> {
        let _completion = if self.is_active(item.task_id) {
            Some(TaskCompletionGuard {
                scheduler: self,
                task_id: item.task_id,
            })
        } else {
            None
        };
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
    #[must_use]
    pub fn metrics(&self) -> TaskMetrics {
        let submitted = self.submitted.load(Ordering::Relaxed);
        let completed = self.completed.load(Ordering::Relaxed);
        let failed = self.failed.load(Ordering::Relaxed);
        let total_exec = self.total_execution_time.load(Ordering::Relaxed);
        let total_wait = self.total_wait_time.load(Ordering::Relaxed);
        let peak_queue = self.peak_queue_depth.load(Ordering::Relaxed);

        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let current_queue = state.queue.len() as u64;

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
        self.state_changed.notify_all();
    }

    /// Check if scheduler is shutdown
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// Clear all pending tasks
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.queue.clear();
        drop(state);
        self.state_changed.notify_all();
    }

    /// Get queue depth
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.queue.len()
    }

    /// Wait until no queued or executing task remains.
    pub fn wait_all(&self) {
        self.wait_until_idle(|| {}, || {});
    }

    /// Return the number of tasks currently executing.
    #[must_use]
    pub(crate) fn active_task_count(&self) -> usize {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.active_tasks.len()
    }

    fn is_active(&self, task_id: u64) -> bool {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.active_tasks.contains(&task_id)
    }

    fn finish_task(&self, task_id: u64) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.active_tasks.remove(&task_id);
        drop(state);
        self.state_changed.notify_all();
    }

    fn wait_until_idle<F, G>(&self, mut on_wait: F, on_idle: G)
    where
        F: FnMut(),
        G: FnOnce(),
    {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while !state.queue.is_empty() || !state.active_tasks.is_empty() {
            on_wait();
            state = self
                .state_changed
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
        on_idle();
    }
}

impl Default for RealTimeScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::sync_channel;

    #[test]
    fn wait_all_waits_for_a_claimed_task() {
        let scheduler = Arc::new(RealTimeScheduler::new());
        let work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync> = Arc::new(|| Ok(()));
        scheduler
            .submit(TaskPriority::Normal, work)
            .expect("scheduler accepts test task");
        let item = scheduler.next_task().expect("task is claimable");

        let (waiting_tx, waiting_rx) = sync_channel(0);
        let (idle_tx, idle_rx) = sync_channel(0);
        let waiter_scheduler = Arc::clone(&scheduler);
        let waiter = std::thread::spawn(move || {
            waiter_scheduler.wait_until_idle(
                || {
                    waiting_tx
                        .send(())
                        .expect("test receiver remains connected")
                },
                || idle_tx.send(()).expect("test receiver remains connected"),
            );
        });

        waiting_rx.recv().expect("waiter observes the active task");
        scheduler
            .execute_task(item)
            .expect("claimed task executes successfully");
        idle_rx.recv().expect("waiter observes task completion");
        waiter.join().expect("waiter exits after completion");
    }
}

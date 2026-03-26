use crate::core::error::{KwaversError, KwaversResult};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use super::scheduler::{current_timestamp, RealTimeScheduler};
use super::task::{TaskPriority, WorkItem};

/// Performance metrics for thread pool
#[derive(Debug, Clone)]
pub struct PoolMetrics {
    /// Number of active threads
    pub active_threads: usize,
    /// Number of idle threads
    pub idle_threads: usize,
    /// Total work time (ms)
    pub total_work_time_ms: u64,
    /// Total idle time (ms)
    pub total_idle_time_ms: u64,
    /// Thread utilization percentage
    pub utilization_percent: f64,
    /// Tasks processed per second
    pub throughput_tps: f64,
}

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of worker threads
    pub num_threads: usize,
    /// Stack size per thread (bytes)
    pub stack_size: Option<usize>,
    /// Enable work stealing
    pub work_stealing_enabled: bool,
    /// Thread naming prefix
    pub thread_name_prefix: String,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            num_threads,
            stack_size: None,
            work_stealing_enabled: true,
            thread_name_prefix: "kwavers-worker".to_string(),
        }
    }
}

/// Multi-threaded work queue for real-time processing
pub struct WorkQueue {
    /// Scheduler instance
    scheduler: Arc<RealTimeScheduler>,
    /// Configuration
    pub(crate) config: ThreadPoolConfig,
    /// Worker threads
    threads: Vec<thread::JoinHandle<()>>,
    /// Thread local work time tracking
    work_time: Arc<AtomicU64>,
    /// Thread local idle time tracking
    idle_time: Arc<AtomicU64>,
}

// Manual Debug implementation since JoinHandle is not Debug
impl std::fmt::Debug for WorkQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkQueue")
            .field("scheduler", &self.scheduler)
            .field("config", &self.config)
            .field("num_threads", &self.threads.len())
            .field("work_time", &self.work_time)
            .field("idle_time", &self.idle_time)
            .finish()
    }
}

impl WorkQueue {
    /// Create a new work queue with the given configuration
    pub fn new(config: ThreadPoolConfig) -> Self {
        let scheduler = Arc::new(RealTimeScheduler::new());
        let work_time = Arc::new(AtomicU64::new(0));
        let idle_time = Arc::new(AtomicU64::new(0));

        let mut threads = Vec::new();

        // Spawn worker threads
        for _i in 0..config.num_threads {
            let scheduler_clone = Arc::clone(&scheduler);
            let work_time_clone = Arc::clone(&work_time);
            let idle_time_clone = Arc::clone(&idle_time);
            let _prefix = config.thread_name_prefix.clone();

            let thread = thread::spawn(move || {
                // Set thread name
                let _ = thread::current().id();

                // Worker loop
                loop {
                    if scheduler_clone.is_shutdown() && scheduler_clone.queue_depth() == 0 {
                        break;
                    }

                    // Try to get next task
                    if let Some(item) = scheduler_clone.next_task() {
                        let start = Instant::now();

                        // Execute task
                        let _ = scheduler_clone.execute_task(item);

                        let elapsed = start.elapsed().as_millis() as u64;
                        work_time_clone.fetch_add(elapsed, Ordering::Relaxed);
                    } else {
                        // No work available, sleep briefly
                        let idle_start = Instant::now();
                        thread::sleep(Duration::from_micros(100));
                        let idle_elapsed = idle_start.elapsed().as_millis() as u64;
                        idle_time_clone.fetch_add(idle_elapsed, Ordering::Relaxed);
                    }
                }
            });

            threads.push(thread);
        }

        Self {
            scheduler,
            config,
            threads,
            work_time,
            idle_time,
        }
    }

    /// Submit a task with given priority
    pub fn submit(
        &self,
        priority: TaskPriority,
        work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync>,
    ) -> KwaversResult<u64> {
        self.scheduler.submit(priority, work)
    }

    /// Submit a task with deadline
    pub fn submit_with_deadline(
        &self,
        priority: TaskPriority,
        deadline_ms: u64,
        work: Arc<dyn Fn() -> KwaversResult<()> + Send + Sync>,
    ) -> KwaversResult<u64> {
        if self.scheduler.is_shutdown() {
            return Err(KwaversError::InvalidInput(
                "Queue is shutting down".to_string(),
            ));
        }

        let task_id = self.scheduler.submitted.fetch_add(1, Ordering::SeqCst);
        let item = WorkItem::new(task_id, priority, work, current_timestamp())
            .with_deadline(deadline_ms, current_timestamp());

        self.scheduler.add_item(item);

        Ok(task_id)
    }

    /// Get current metrics
    pub fn metrics(&self) -> PoolMetrics {
        let task_metrics = self.scheduler.metrics();
        let work = self.work_time.load(Ordering::Relaxed);
        let idle = self.idle_time.load(Ordering::Relaxed);
        let total_time = work + idle;

        let utilization = if total_time > 0 {
            (work as f64 / total_time as f64) * 100.0
        } else {
            0.0
        };

        let active_threads = if task_metrics.current_queue_depth > 0 {
            (task_metrics.current_queue_depth as usize).min(self.config.num_threads)
        } else {
            0
        };

        let throughput = if work > 0 {
            (task_metrics.total_completed as f64 / work as f64) * 1000.0
        } else {
            0.0
        };

        PoolMetrics {
            active_threads,
            idle_threads: self.config.num_threads - active_threads,
            total_work_time_ms: work,
            total_idle_time_ms: idle,
            utilization_percent: utilization,
            throughput_tps: throughput,
        }
    }

    /// Wait for all pending tasks to complete
    pub fn wait_all(&self) -> KwaversResult<()> {
        loop {
            if self.scheduler.queue_depth() == 0 {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }

    /// Shutdown the work queue and wait for threads to finish
    pub fn shutdown(&mut self) -> KwaversResult<()> {
        self.scheduler.shutdown();

        // Wait for all threads to finish
        for thread in self.threads.drain(..) {
            if thread.join().is_err() {
                return Err(KwaversError::InvalidInput(
                    "Failed to join worker thread".to_string(),
                ));
            }
        }

        Ok(())
    }
}

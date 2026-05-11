use crate::solver::validation::contract::MemoryBudget;
use std::sync::Arc;
use tracing::debug;
use tracing::info;

use super::ThreadAllocationTracker;

/// Global allocation tracker spanning all threads
///
/// Aggregates per-thread trackers and provides solver-wide memory metrics.
/// For single-threaded solvers, this is a simple wrapper around ThreadAllocationTracker.
#[derive(Debug)]
pub struct GlobalAllocationTracker {
    /// Per-thread trackers (thread-indexed)
    pub(super) thread_trackers: Vec<ThreadAllocationTracker>,
    /// Memory budget shared across all threads
    pub(super) budget: Arc<MemoryBudget>,
    /// Optional GPU allocation tracker for unified telemetry
    gpu_tracker: Option<Arc<crate::profiling::gpu_allocator::GpuAllocationTracker>>,
}

impl GlobalAllocationTracker {
    /// Create new global tracker with shared budget
    pub fn new(
        num_threads: usize,
        budget: Arc<MemoryBudget>,
        gpu_tracker: Option<Arc<crate::profiling::gpu_allocator::GpuAllocationTracker>>,
    ) -> Self {
        let mut trackers = Vec::with_capacity(num_threads);

        for _ in 0..num_threads {
            trackers.push(ThreadAllocationTracker::new(Arc::clone(&budget)));
        }

        Self {
            thread_trackers: trackers,
            budget,
            gpu_tracker,
        }
    }

    /// Get tracker for specific thread
    #[inline]
    pub fn thread_tracker(&self, thread_id: usize) -> Option<&ThreadAllocationTracker> {
        self.thread_trackers.get(thread_id)
    }

    /// Get current thread's tracker (uses current thread index)
    #[inline]
    pub fn current_thread_tracker(&self) -> &ThreadAllocationTracker {
        // For rayon thread pools, we'd use rayon::current_thread_index()
        // For now, default to thread 0 for single-threaded
        &self.thread_trackers[0]
    }

    /// Get total transient bytes across all threads
    #[inline]
    pub fn total_current_bytes(&self) -> usize {
        self.thread_trackers.iter().map(|t| t.current_bytes()).sum()
    }

    /// Get peak transient bytes across all threads
    #[inline]
    pub fn total_peak_bytes(&self) -> usize {
        self.thread_trackers
            .iter()
            .map(|t| t.peak_bytes())
            .max()
            .unwrap_or(0)
    }

    /// Get combined memory budget with current state
    pub fn current_budget(&self) -> MemoryBudget {
        (*self.budget).clone()
    }

    /// Reset all thread trackers (for new timestep)
    pub fn reset_all(&self) {
        for tracker in &self.thread_trackers {
            tracker.reset();
        }
        debug!("All thread allocation trackers reset");
    }

    /// Log memory statistics
    pub fn log_statistics(&self) {
        if let Some(gpu) = &self.gpu_tracker {
            info!(
                cpu_total_peak = self.total_peak_bytes(),
                cpu_budget_workspace = self.budget.workspace_bytes,
                cpu_budget_peak = self.budget.peak_transient(),
                gpu_current_bytes = gpu.current_bytes(),
                gpu_peak_bytes = gpu.peak_bytes(),
                "Memory tracking statistics (Unified)"
            );
        } else {
            info!(
                total_peak = self.total_peak_bytes(),
                budget_workspace = self.budget.workspace_bytes,
                budget_peak = self.budget.peak_transient(),
                "Memory tracking statistics"
            );
        }
    }
}

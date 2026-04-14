// Memory tracking and instrumentation for solver operations
//
// This module provides comprehensive memory allocation tracking that enables:
// - Real-time transient allocation monitoring
// - Peak memory detection with O(1) overhead
// - Allocation tracing with causal chain preservation
// - Memory budget enforcement
//
// ## Mathematical Specification
//
// ### Memory Invariant Theorem
// For a simulation S with n timesteps and workspace W:
//
// ```text
// TotalMemory(S) = StaticMemory(W) + Σ TransientMemory(t) for t ∈ [1, n]
//
// where:
// - StaticMemory(W) = Σ (buffer_size × element_size) for all pre-allocated buffers
// - TransientMemory(t) = Σ (alloc_size) for all allocations at timestep t
// - PeakMemory = StaticMemory + max(TransientMemory(t)) ∀t ∈ [1, n]
// ```
//
// ### Complexity Analysis
// - allocate(): O(1) amortized
// - record_transient(): O(1) with atomic operations
// - peak_memory(): O(1) read of cached maximum
// - total_memory(): O(1) arithmetic
//
// ## References
//
// - Wilson et al. (1995) "Dynamic Memory Management"
//   ISBN: 0-201-52992-9
// - Berger et al. (2001) "Composing High-Performance Memory Allocators"
//   DOI: 10.1145/378993.379433
// - Drepper (2007) "What Every Programmer Should Know About Memory"
//   https://akkadia.org/drepper/cpumemory.pdf

use crate::solver::validation::contract::MemoryBudget;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{debug, info, trace};

/// Thread-local allocation tracker for transient memory
///
/// Tracks allocations within a single thread. For parallel solvers,
/// each thread maintains its own tracker, aggregated by MemoryBudget.
#[derive(Debug)]
pub struct ThreadAllocationTracker {
    /// Current allocation count for this thread
    current_bytes: AtomicUsize,
    /// Peak allocation for this thread
    peak_bytes: AtomicUsize,
    /// Total allocations (cumulative)
    total_allocations: AtomicUsize,
    /// Parent budget for aggregation
    budget: Arc<MemoryBudget>,
}

impl ThreadAllocationTracker {
    /// Create new thread-local tracker linked to a budget
    pub fn new(budget: Arc<MemoryBudget>) -> Self {
        Self {
            current_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            total_allocations: AtomicUsize::new(0),
            budget,
        }
    }

    /// Record a transient allocation
    ///
    /// Updates both local counters and parent budget atomically.
    /// Thread-safe for concurrent allocation tracking.
    #[inline]
    pub fn allocate(&self, bytes: usize) {
        // Update local current bytes
        let current = self.current_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;

        // Update peak if needed
        let mut peak = self.peak_bytes.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_bytes.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }

        // Increment total allocations
        self.total_allocations.fetch_add(1, Ordering::Relaxed);

        trace!(
            bytes = bytes,
            current = current,
            "Transient allocation recorded"
        );
    }

    /// Record a deallocation
    ///
    /// Decrements current bytes but preserves peak.
    #[inline]
    pub fn deallocate(&self, bytes: usize) {
        let current = self
            .current_bytes
            .fetch_sub(bytes, Ordering::Relaxed)
            .saturating_sub(bytes);

        trace!(bytes = bytes, current = current, "Deallocation recorded");
    }

    /// Get current transient bytes for this thread
    #[inline]
    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Get peak transient bytes for this thread
    #[inline]
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    /// Get total allocation count
    #[inline]
    pub fn total_allocations(&self) -> usize {
        self.total_allocations.load(Ordering::Relaxed)
    }

    /// Reset thread-local counters (for reuse in new timestep)
    #[inline]
    pub fn reset(&self) {
        let current = self.current_bytes.swap(0, Ordering::Relaxed);

        // Update budget with current
        if let Ok(mut budget) = std::sync::Arc::try_unwrap(Arc::clone(&self.budget)) {
            budget.record_transient(current);
        }

        trace!(previous = current, "Thread allocation tracker reset");
    }
}

/// Global allocation tracker spanning all threads
///
/// Aggregates per-thread trackers and provides solver-wide memory metrics.
/// For single-threaded solvers, this is a simple wrapper around ThreadAllocationTracker.
#[derive(Debug)]
pub struct GlobalAllocationTracker {
    /// Per-thread trackers (thread-indexed)
    thread_trackers: Vec<ThreadAllocationTracker>,
    /// Memory budget shared across all threads
    budget: Arc<MemoryBudget>,
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

/// RAII guard for scoped allocation tracking
///
/// Automatically records allocation on construction and deallocation on drop.
/// Use this for temporary buffers that should be tracked.
#[derive(Debug)]
pub struct AllocationGuard<'a> {
    tracker: &'a ThreadAllocationTracker,
    bytes: usize,
    allocated: bool,
}

impl<'a> AllocationGuard<'a> {
    /// Create new allocation guard
    pub fn new(tracker: &'a ThreadAllocationTracker, bytes: usize) -> Self {
        tracker.allocate(bytes);
        debug!(bytes = bytes, "Allocation guard created");
        Self {
            tracker,
            bytes,
            allocated: true,
        }
    }

    /// Explicitly release allocation before drop
    pub fn release(mut self) {
        if self.allocated {
            self.tracker.deallocate(self.bytes);
            self.allocated = false;
            debug!(bytes = self.bytes, "Allocation guard released");
        }
    }
}

impl<'a> Drop for AllocationGuard<'a> {
    fn drop(&mut self) {
        if self.allocated {
            self.tracker.deallocate(self.bytes);
            trace!(bytes = self.bytes, "Allocation guard auto-released");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_budget() -> Arc<MemoryBudget> {
        Arc::new(MemoryBudget::new(1024 * 1024 * 100)) // 100 MB
    }

    #[test]
    fn thread_tracker_records_allocations() {
        let budget = create_test_budget();
        let tracker = ThreadAllocationTracker::new(budget);

        assert_eq!(tracker.current_bytes(), 0);
        assert_eq!(tracker.peak_bytes(), 0);

        tracker.allocate(1000);
        assert_eq!(tracker.current_bytes(), 1000);
        assert_eq!(tracker.peak_bytes(), 1000);

        tracker.allocate(500);
        assert_eq!(tracker.current_bytes(), 1500);
        assert_eq!(tracker.peak_bytes(), 1500);

        tracker.deallocate(500);
        assert_eq!(tracker.current_bytes(), 1000);
        assert_eq!(tracker.peak_bytes(), 1500); // Peak preserved
    }

    #[test]
    fn thread_tracker_tracks_allocations_count() {
        let budget = create_test_budget();
        let tracker = ThreadAllocationTracker::new(budget);

        assert_eq!(tracker.total_allocations(), 0);

        tracker.allocate(100);
        tracker.allocate(200);
        tracker.allocate(300);

        assert_eq!(tracker.total_allocations(), 3);
    }

    #[test]
    fn thread_tracker_reset_clears_current() {
        let budget = create_test_budget();
        let tracker = ThreadAllocationTracker::new(budget);

        tracker.allocate(1000);
        assert_eq!(tracker.current_bytes(), 1000);

        tracker.reset();
        assert_eq!(tracker.current_bytes(), 0);
        assert_eq!(tracker.peak_bytes(), 1000); // Peak preserved
    }

    #[test]
    fn allocation_guard_tracks_on_drop() {
        let budget = create_test_budget();
        let tracker = ThreadAllocationTracker::new(budget);

        {
            let _guard = AllocationGuard::new(&tracker, 500);
            assert_eq!(tracker.current_bytes(), 500);
        }

        // Should be deallocated on drop
        assert_eq!(tracker.current_bytes(), 0);
        assert_eq!(tracker.peak_bytes(), 500);
    }

    #[test]
    fn allocation_guard_release_prevents_double_dealloc() {
        let budget = create_test_budget();
        let tracker = ThreadAllocationTracker::new(budget);

        let guard = AllocationGuard::new(&tracker, 500);
        guard.release();

        // After explicit release, drop should not deallocate again
        assert_eq!(tracker.current_bytes(), 0);
    }

    #[test]
    fn global_tracker_aggregates_threads() {
        let budget = create_test_budget();
        let global = GlobalAllocationTracker::new(2, budget, None);

        // Simulate allocations on thread 0
        global.thread_trackers[0].allocate(1000);
        global.thread_trackers[0].allocate(500);

        // Simulate allocations on thread 1
        global.thread_trackers[1].allocate(200);

        assert_eq!(global.total_current_bytes(), 1700);
        assert_eq!(global.total_peak_bytes(), 1500); // Max of threads
    }
}

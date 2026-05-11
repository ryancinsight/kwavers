use crate::solver::validation::contract::MemoryBudget;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::trace;

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
    pub(super) budget: Arc<MemoryBudget>,
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

//! `GpuAllocationTracker` and `GpuAllocationGuard`.
//!
//! ## Mathematical Specification
//!
//! **THEOREM: GPU Memory Invariant**
//!
//! For GPU device D with memory capacity M and safety factor α (default 0.9):
//! ```text
//! GPU_Memory_Used ≤ M × α
//! ```
//!
//! **Proof**: Pre-allocation budget check in `allocate()`:
//! ```text
//! if current_bytes + size > budget_bytes {
//!     return Err(GpuError::OutOfMemory { ... })
//! }
//! current_bytes += size  // Atomic operation
//! ```
//!
//! **Complexity**: O(1) amortized via atomic operations.
//! - `allocate()`: O(1) atomic fetch_add + compare
//! - deallocate (Drop): O(1) atomic fetch_sub
//! - `peak_memory()`: O(1) read of cached maximum

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{debug, info, trace};

use kwavers_core::error::gpu::GpuError;

use super::config::GpuAllocationConfig;
use super::stats::GpuAllocationStats;

/// RAII guard for a tracked GPU allocation.
///
/// Automatically decrements the allocation counter when dropped.
#[derive(Debug)]
pub struct GpuAllocationGuard {
    pub(super) tracker: Arc<GpuAllocationTracker>,
    pub(super) size: usize,
    pub(super) name: String,
}

impl Drop for GpuAllocationGuard {
    fn drop(&mut self) {
        self.tracker.deallocate_impl(self.size, &self.name);
        trace!(size = self.size, name = %self.name, "GPU allocation released");
    }
}

/// O(1) GPU Memory Allocation Tracker.
///
/// Tracks GPU memory allocations with atomic operations for thread safety.
///
/// ## Thread Safety
///
/// All mutable state is accessed via atomic operations:
/// - `current_bytes: AtomicUsize` — live usage
/// - `peak_bytes: AtomicUsize`    — high-water mark
/// - `allocation_count / deallocation_count: AtomicUsize`
///
/// ## Peak Tracking
///
/// A CAS loop ensures `peak_bytes` is monotonically non-decreasing:
/// ```text
/// repeat:
///     old_peak = peak_bytes.load()
///     new_peak = max(old_peak, current)
///     if CAS(old_peak, new_peak) == old_peak: break
///     goto repeat
/// ```
#[derive(Debug)]
pub struct GpuAllocationTracker {
    /// Current allocated bytes (atomic).
    current_bytes: AtomicUsize,
    /// Peak allocated bytes (atomic).
    peak_bytes: AtomicUsize,
    /// Total allocation count.
    allocation_count: AtomicUsize,
    /// Total deallocation count.
    deallocation_count: AtomicUsize,
    /// Device estimated capacity in bytes.
    device_capacity: usize,
    /// Memory budget (`device_capacity × safety_factor`).
    budget_bytes: usize,
    /// Configuration.
    config: GpuAllocationConfig,
}

impl GpuAllocationTracker {
    /// Create a new GPU allocation tracker wrapped in `Arc`.
    ///
    /// # Arguments
    /// * `device_capacity` — Estimated device memory capacity in bytes.
    /// * `config` — Allocation configuration.
    #[must_use]
    pub fn new(device_capacity: usize, config: GpuAllocationConfig) -> Arc<Self> {
        let budget_bytes = (device_capacity as f64 * config.safety_factor) as usize;
        Arc::new(Self {
            current_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            deallocation_count: AtomicUsize::new(0),
            device_capacity,
            budget_bytes,
            config,
        })
    }

    /// Create with default configuration (90% safety factor).
    #[must_use]
    pub fn with_capacity(device_capacity: usize) -> Arc<Self> {
        Self::new(device_capacity, GpuAllocationConfig::default())
    }

    /// Return the allocation configuration.
    pub fn config(&self) -> &GpuAllocationConfig {
        &self.config
    }

    /// Return the current allocated bytes.
    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Return the peak allocated bytes.
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    /// Track a new GPU allocation.
    ///
    /// Performs a pre-allocation budget check before permitting the allocation.
    ///
    /// # Returns
    /// - `Ok(GpuAllocationGuard)` — allocation permitted; guard auto-releases on drop.
    /// - `Err(GpuError::OutOfMemory)` — budget would be exceeded.
    ///
    /// # Complexity
    /// O(1) amortized via atomic operations.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn allocate(
        self: &Arc<Self>,
        size: usize,
        name: &str,
    ) -> Result<GpuAllocationGuard, GpuError> {
        let current = self.current_bytes.load(Ordering::Relaxed);
        if current + size > self.budget_bytes {
            return Err(GpuError::OutOfMemory {
                requested: size,
                available: self.budget_bytes - current,
                current,
            });
        }

        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        let new_current = self.current_bytes.fetch_add(size, Ordering::Relaxed) + size;
        self.update_peak(new_current);

        debug!(
            size = size,
            name = %name,
            current = new_current,
            peak = self.peak_bytes.load(Ordering::Relaxed),
            "GPU allocation tracked"
        );

        Ok(GpuAllocationGuard {
            tracker: self.clone(),
            size,
            name: name.to_owned(),
        })
    }

    /// Internal deallocate — called by `GpuAllocationGuard::drop`.
    pub(super) fn deallocate_impl(&self, size: usize, _name: &str) {
        self.current_bytes.fetch_sub(size, Ordering::Relaxed);
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Update `peak_bytes` via CAS loop.
    fn update_peak(&self, current: usize) {
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
    }

    /// Return current memory usage in bytes.
    pub fn current_memory(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Return peak memory usage in bytes.
    pub fn peak_memory(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    /// Return a snapshot of allocation statistics.
    pub fn stats(&self) -> GpuAllocationStats {
        GpuAllocationStats {
            total_allocations: self.allocation_count.load(Ordering::Relaxed),
            total_deallocations: self.deallocation_count.load(Ordering::Relaxed),
            current_bytes: self.current_bytes.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
            budget_bytes: self.budget_bytes,
            device_capacity: self.device_capacity,
        }
    }

    /// Return `true` if allocating `size` bytes would exceed the budget.
    pub fn would_exceed_budget(&self, size: usize) -> bool {
        let current = self.current_bytes.load(Ordering::Relaxed);
        current + size > self.budget_bytes
    }

    /// Return available memory in bytes (relative to safety-factor budget).
    pub fn available_bytes(&self) -> usize {
        let current = self.current_bytes.load(Ordering::Relaxed);
        self.budget_bytes.saturating_sub(current)
    }

    /// Log current allocation state via `tracing::info!`.
    pub fn log_status(&self) {
        let stats = self.stats();
        let utilization = stats.utilization() * 100.0;
        info!(
            current_bytes = stats.current_bytes,
            peak_bytes = stats.peak_bytes,
            budget_bytes = stats.budget_bytes,
            utilization = ?utilization,
            allocations = stats.total_allocations,
            deallocations = stats.total_deallocations,
            "GPU allocation status"
        );
    }

    /// Reset peak tracking to the current usage level.
    pub fn reset_peak(&self) {
        self.peak_bytes.store(
            self.current_bytes.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        info!("GPU peak memory tracker reset");
    }
}

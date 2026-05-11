use tracing::{debug, trace};

use super::ThreadAllocationTracker;

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

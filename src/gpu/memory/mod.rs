//! GPU memory management
//!
//! Provides efficient memory allocation and transfer for GPU operations

use crate::KwaversResult;

/// Memory statistics
#[derive(Debug, Default)]
#[derive(Debug)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
    pub transfer_count: usize,
    pub transfer_bytes: usize,
}

impl MemoryStats {
    /// Create memory statistics
    pub fn create() -> Self {
        Self::default()
    }

    /// Record allocation
    pub fn record_allocation(&mut self, bytes: usize) {
        self.allocated_bytes += bytes;
        self.peak_bytes = self.peak_bytes.max(self.allocated_bytes);
    }

    /// Record deallocation
    pub fn record_deallocation(&mut self, bytes: usize) {
        self.allocated_bytes = self.allocated_bytes.saturating_sub(bytes);
    }

    /// Record transfer
    pub fn record_transfer(&mut self, bytes: usize) {
        self.transfer_count += 1;
        self.transfer_bytes += bytes;
    }
}

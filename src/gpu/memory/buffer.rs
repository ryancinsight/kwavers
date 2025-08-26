//! GPU buffer management

use std::time::Instant;

/// GPU memory buffer descriptor
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    pub id: usize,
    pub size_bytes: usize,
    pub device_ptr: Option<u64>,
    pub host_ptr: Option<*mut u8>,
    pub is_pinned: bool,
    pub allocation_time: Instant,
    pub last_access_time: Instant,
}

impl GpuBuffer {
    /// Update last access time
    pub fn touch(&mut self) {
        self.last_access_time = Instant::now();
    }

    /// Check if buffer is valid
    pub fn is_valid(&self) -> bool {
        self.device_ptr.is_some() || self.host_ptr.is_some()
    }

    /// Get age in seconds
    pub fn age_seconds(&self) -> f64 {
        self.allocation_time.elapsed().as_secs_f64()
    }
}

/// Buffer descriptor for allocation requests
#[derive(Debug, Clone)]
pub struct BufferDescriptor {
    pub size_bytes: usize,
    pub pinned: bool,
    pub zero_init: bool,
}

//! Frame pool for zero-allocation frame buffer reuse.

use ndarray::Array3;
use parking_lot::Mutex;
use tracing::{trace, warn};

/// Frame pool for zero-allocation frame reuse.
///
/// Implements a bump allocator pattern for frame buffers to avoid
/// repeated heap allocations during high-frequency simulation.
#[derive(Debug)]
pub struct FramePool {
    /// Pre-allocated buffer storage.
    buffers: Mutex<Vec<Array3<f32>>>,
    /// Template dimensions for allocation.
    dimensions: (usize, usize, usize),
    /// Maximum pool size.
    max_size: usize,
}

impl FramePool {
    /// Create a new frame pool with specified dimensions.
    pub fn new(nx: usize, ny: usize, nz: usize, max_size: usize) -> Self {
        let mut buffers = Vec::with_capacity(max_size);
        for _ in 0..max_size {
            buffers.push(Array3::<f32>::zeros((nx, ny, nz)));
        }

        Self {
            buffers: Mutex::new(buffers),
            dimensions: (nx, ny, nz),
            max_size,
        }
    }

    /// Acquire a buffer from the pool.
    /// Returns a zeroed array if pool is exhausted.
    pub fn acquire(&self) -> Array3<f32> {
        let mut buffers = self.buffers.lock();
        if let Some(buffer) = buffers.pop() {
            trace!(pool_size = buffers.len(), "Acquired buffer from pool");
            buffer
        } else {
            warn!("Frame pool exhausted, allocating new buffer");
            Array3::<f32>::zeros(self.dimensions)
        }
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&self, mut buffer: Array3<f32>) {
        buffer.fill(0.0);
        let mut buffers = self.buffers.lock();
        if buffers.len() < self.max_size {
            buffers.push(buffer);
            trace!(pool_size = buffers.len(), "Returned buffer to pool");
        } else {
            trace!("Pool full, dropping buffer");
        }
    }
}

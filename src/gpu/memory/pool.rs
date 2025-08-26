//! GPU memory pool management

use super::buffer::GpuBuffer;
use crate::error::{KwaversError, KwaversResult};
use std::collections::VecDeque;

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub initial_size: usize,
    pub max_size: usize,
    pub chunk_size: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 100 << 20, // 100MB
            max_size: 1 << 30,       // 1GB
            chunk_size: 4 << 20,     // 4MB chunks
        }
    }
}

/// GPU memory pool
#[derive(Debug)]
pub struct MemoryPool {
    config: PoolConfig,
    free_chunks: VecDeque<GpuBuffer>,
    allocated_size: usize,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(config: PoolConfig) -> KwaversResult<Self> {
        Ok(Self {
            config,
            free_chunks: VecDeque::new(),
            allocated_size: 0,
        })
    }

    /// Allocate from pool
    pub fn allocate(&mut self, size_bytes: usize) -> KwaversResult<GpuBuffer> {
        // Find suitable chunk
        for i in 0..self.free_chunks.len() {
            if self.free_chunks[i].size_bytes >= size_bytes {
                return Ok(self.free_chunks.remove(i).unwrap());
            }
        }

        Err(KwaversError::ResourceExhausted(
            "No suitable chunk in pool".to_string(),
        ))
    }

    /// Check if buffer can be recycled
    pub fn can_recycle(&self, buffer: &GpuBuffer) -> bool {
        buffer.size_bytes == self.config.chunk_size
            && self.allocated_size + buffer.size_bytes <= self.config.max_size
    }

    /// Recycle buffer back to pool
    pub fn recycle(&mut self, buffer: GpuBuffer) -> KwaversResult<()> {
        if !self.can_recycle(&buffer) {
            return Err(KwaversError::InvalidParameter(
                "Buffer cannot be recycled".to_string(),
            ));
        }

        self.free_chunks.push_back(buffer);
        Ok(())
    }

    /// Get pool size in bytes
    pub fn size_bytes(&self) -> usize {
        self.allocated_size
    }

    /// Clear pool
    pub fn clear(&mut self) {
        self.free_chunks.clear();
        self.allocated_size = 0;
    }
}

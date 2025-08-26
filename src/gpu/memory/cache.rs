//! GPU memory caching

use super::buffer::GpuBuffer;
use std::collections::{HashMap, VecDeque};

/// Cache strategy
#[derive(Debug, Clone, Copy)]
pub enum CacheStrategy {
    /// Least Recently Used
    Lru,
    /// First In First Out
    Fifo,
    /// Size-based
    SizeBased,
}

/// Memory cache for reusing buffers
#[derive(Debug)]
pub struct MemoryCache {
    strategy: CacheStrategy,
    buffers: HashMap<usize, VecDeque<GpuBuffer>>,
    total_size: usize,
    max_size: usize,
}

impl MemoryCache {
    /// Create new cache
    pub fn new(strategy: CacheStrategy) -> Self {
        const MAX_CACHE_SIZE: usize = 1 << 30; // 1GB max cache

        Self {
            strategy,
            buffers: HashMap::new(),
            total_size: 0,
            max_size: MAX_CACHE_SIZE,
        }
    }

    /// Get buffer from cache
    pub fn get(&mut self, size_bytes: usize) -> Option<GpuBuffer> {
        self.buffers
            .get_mut(&size_bytes)?
            .pop_front()
            .map(|mut buffer| {
                buffer.touch();
                self.total_size -= buffer.size_bytes;
                buffer
            })
    }

    /// Add buffer to cache
    pub fn put(&mut self, mut buffer: GpuBuffer) {
        if self.total_size + buffer.size_bytes > self.max_size {
            self.evict(buffer.size_bytes);
        }

        buffer.touch();
        self.total_size += buffer.size_bytes;
        self.buffers
            .entry(buffer.size_bytes)
            .or_insert_with(VecDeque::new)
            .push_back(buffer);
    }

    /// Evict buffers to make space
    fn evict(&mut self, needed_bytes: usize) {
        let mut freed = 0;

        // Simple eviction: remove oldest buffers
        for buffers in self.buffers.values_mut() {
            while !buffers.is_empty() && freed < needed_bytes {
                if let Some(buffer) = buffers.pop_front() {
                    freed += buffer.size_bytes;
                    self.total_size -= buffer.size_bytes;
                }
            }
            if freed >= needed_bytes {
                break;
            }
        }
    }

    /// Get cache size in bytes
    pub fn size_bytes(&self) -> usize {
        self.total_size
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.total_size = 0;
    }
}

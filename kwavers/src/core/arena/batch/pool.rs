use std::sync::atomic::{AtomicUsize, Ordering};

/// Pool allocator for temporary computation buffers.
///
/// Pre-allocates buffers in three size classes to eliminate allocation in hot
/// loops.  Uses LIFO (stack) discipline for cache efficiency.
#[derive(Debug)]
pub struct TempBufferPool {
    small: Vec<Box<[f64]>>,  // ≤256 elements
    medium: Vec<Box<[f64]>>, // ≤4096 elements
    large: Vec<Box<[f64]>>,  // ≤65536 elements
    stat_allocated: AtomicUsize,
    stat_reused: AtomicUsize,
}

/// Buffer size classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferSize {
    /// ~2 KB (256 × f64)
    Small,
    /// ~32 KB (4096 × f64)
    Medium,
    /// ~512 KB (65536 × f64)
    Large,
    /// Custom size (not pooled)
    Custom(usize),
}

impl BufferSize {
    /// Classify a required element count into a pool tier.
    #[must_use]
    pub fn classify(elements: usize) -> Self {
        if elements <= 256 {
            Self::Small
        } else if elements <= 4096 {
            Self::Medium
        } else if elements <= 65536 {
            Self::Large
        } else {
            Self::Custom(elements)
        }
    }

    /// Capacity in elements for this tier.
    #[must_use]
    pub fn capacity(&self) -> usize {
        match self {
            Self::Small => 256,
            Self::Medium => 4096,
            Self::Large => 65536,
            Self::Custom(n) => *n,
        }
    }
}

impl Default for TempBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl TempBufferPool {
    /// Create an empty pool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            small: Vec::new(),
            medium: Vec::new(),
            large: Vec::new(),
            stat_allocated: AtomicUsize::new(0),
            stat_reused: AtomicUsize::new(0),
        }
    }

    /// Pre-allocate buffers in each size class.
    pub fn preallocate(&mut self, small_count: usize, medium_count: usize, large_count: usize) {
        for _ in 0..small_count {
            self.small
                .push(vec![0.0; BufferSize::Small.capacity()].into_boxed_slice());
        }
        for _ in 0..medium_count {
            self.medium
                .push(vec![0.0; BufferSize::Medium.capacity()].into_boxed_slice());
        }
        for _ in 0..large_count {
            self.large
                .push(vec![0.0; BufferSize::Large.capacity()].into_boxed_slice());
        }
    }

    /// Acquire a buffer with at least `min_elements` capacity.
    pub fn acquire(&mut self, min_elements: usize) -> Vec<f64> {
        let size_class = BufferSize::classify(min_elements);

        let buffer = match size_class {
            BufferSize::Small => self.small.pop(),
            BufferSize::Medium => self.medium.pop(),
            BufferSize::Large => self.large.pop(),
            BufferSize::Custom(_) => None,
        };

        match buffer {
            Some(b) => {
                self.stat_reused.fetch_add(1, Ordering::Relaxed);
                let mut vec = b.into_vec();
                vec.clear();
                vec.resize(min_elements, 0.0);
                vec
            }
            None => {
                self.stat_allocated.fetch_add(1, Ordering::Relaxed);
                vec![0.0; size_class.capacity().max(min_elements)]
            }
        }
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&mut self, mut buffer: Vec<f64>) {
        let cap = buffer.capacity();
        buffer.clear();

        if cap == BufferSize::Large.capacity() {
            self.large.push(buffer.into_boxed_slice());
        } else if cap == BufferSize::Medium.capacity() {
            self.medium.push(buffer.into_boxed_slice());
        } else if cap == BufferSize::Small.capacity() {
            self.small.push(buffer.into_boxed_slice());
        }
        // Custom-sized buffers are dropped (not pooled).
    }

    /// Current pool sizes per tier (small, medium, large).
    pub fn pool_sizes(&self) -> (usize, usize, usize) {
        (self.small.len(), self.medium.len(), self.large.len())
    }

    /// Total new allocations (excluding pre-allocation).
    pub fn total_allocated(&self) -> usize {
        self.stat_allocated.load(Ordering::Relaxed)
    }

    /// Total reused buffers from pool.
    pub fn total_reused(&self) -> usize {
        self.stat_reused.load(Ordering::Relaxed)
    }

    /// Efficiency ratio: `reused / (reused + allocated)`.
    pub fn efficiency_ratio(&self) -> f64 {
        let reused = self.total_reused() as f64;
        let allocated = self.total_allocated() as f64;
        if reused + allocated > 0.0 {
            reused / (reused + allocated)
        } else {
            0.0
        }
    }

    /// Clear all pools.
    pub fn clear(&mut self) {
        self.small.clear();
        self.medium.clear();
        self.large.clear();
        self.stat_allocated.store(0, Ordering::Relaxed);
        self.stat_reused.store(0, Ordering::Relaxed);
    }
}

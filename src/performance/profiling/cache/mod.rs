//! Cache profiling infrastructure
//!
//! Analyzes cache behavior and memory access patterns.

use std::sync::{Arc, Mutex};

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    /// L1 cache hits
    pub l1_hits: usize,
    /// L1 cache misses
    pub l1_misses: usize,
    /// L2 cache hits
    pub l2_hits: usize,
    /// L2 cache misses
    pub l2_misses: usize,
    /// L3 cache hits
    pub l3_hits: usize,
    /// L3 cache misses
    pub l3_misses: usize,
    /// TLB hits
    pub tlb_hits: usize,
    /// TLB misses
    pub tlb_misses: usize,
}

impl CacheStatistics {
    /// Get L1 cache hit rate
    #[must_use]
    pub fn l1_hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l1_misses;
        if total == 0 {
            0.0
        } else {
            self.l1_hits as f64 / total as f64
        }
    }

    /// Get L2 cache hit rate
    #[must_use]
    pub fn l2_hit_rate(&self) -> f64 {
        let total = self.l2_hits + self.l2_misses;
        if total == 0 {
            0.0
        } else {
            self.l2_hits as f64 / total as f64
        }
    }

    /// Get L3 cache hit rate
    #[must_use]
    pub fn l3_hit_rate(&self) -> f64 {
        let total = self.l3_hits + self.l3_misses;
        if total == 0 {
            0.0
        } else {
            self.l3_hits as f64 / total as f64
        }
    }

    /// Get TLB hit rate
    #[must_use]
    pub fn tlb_hit_rate(&self) -> f64 {
        let total = self.tlb_hits + self.tlb_misses;
        if total == 0 {
            0.0
        } else {
            self.tlb_hits as f64 / total as f64
        }
    }
}

/// Cache profile with detailed metrics
#[derive(Debug, Clone)]
pub struct CacheProfile {
    /// Cache line size in bytes
    pub cache_line_size: usize,
    /// L1 cache size in bytes
    pub l1_size: usize,
    /// L2 cache size in bytes
    pub l2_size: usize,
    /// L3 cache size in bytes
    pub l3_size: usize,
    /// Cache statistics
    pub statistics: CacheStatistics,
}

impl CacheProfile {
    /// Create a new cache profile with typical `x86_64` values
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache_line_size: 64,
            l1_size: 32 * 1024,       // 32 KB
            l2_size: 256 * 1024,      // 256 KB
            l3_size: 8 * 1024 * 1024, // 8 MB
            statistics: CacheStatistics::default(),
        }
    }

    /// Estimate cache efficiency based on hit rates
    #[must_use]
    pub fn efficiency(&self) -> f64 {
        // Weighted average of cache hit rates
        let l1_weight = 0.5;
        let l2_weight = 0.3;
        let l3_weight = 0.2;

        self.statistics.l1_hit_rate() * l1_weight
            + self.statistics.l2_hit_rate() * l2_weight
            + self.statistics.l3_hit_rate() * l3_weight
    }

    /// Estimate memory bandwidth utilization
    #[must_use]
    pub fn bandwidth_utilization(&self, bytes_transferred: usize, time_seconds: f64) -> f64 {
        if time_seconds <= 0.0 {
            return 0.0;
        }

        let bandwidth_gb_s = (bytes_transferred as f64 / 1e9) / time_seconds;

        // Assume typical memory bandwidth of 50 GB/s for modern systems
        const TYPICAL_BANDWIDTH_GB_S: f64 = 50.0;

        (bandwidth_gb_s / TYPICAL_BANDWIDTH_GB_S).min(1.0)
    }
}

impl Default for CacheProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache profiler for analyzing memory access patterns
#[derive(Debug, Clone)]
pub struct CacheProfiler {
    profile: Arc<Mutex<CacheProfile>>,
}

impl CacheProfiler {
    /// Create a new cache profiler
    #[must_use]
    pub fn new() -> Self {
        Self {
            profile: Arc::new(Mutex::new(CacheProfile::new())),
        }
    }

    /// Record a cache hit at level
    pub fn record_hit(&self, level: usize) {
        if let Ok(mut profile) = self.profile.lock() {
            match level {
                1 => profile.statistics.l1_hits += 1,
                2 => profile.statistics.l2_hits += 1,
                3 => profile.statistics.l3_hits += 1,
                _ => {}
            }
        }
    }

    /// Record a cache miss at level
    pub fn record_miss(&self, level: usize) {
        if let Ok(mut profile) = self.profile.lock() {
            match level {
                1 => profile.statistics.l1_misses += 1,
                2 => profile.statistics.l2_misses += 1,
                3 => profile.statistics.l3_misses += 1,
                _ => {}
            }
        }
    }

    /// Get current cache profile
    #[must_use]
    pub fn profile(&self) -> CacheProfile {
        self.profile.lock().map(|p| p.clone()).unwrap_or_else(|_| CacheProfile::new())
    }

    /// Clear cache statistics
    pub fn clear(&self) {
        if let Ok(mut profile) = self.profile.lock() {
            profile.statistics = CacheStatistics::default();
        }
    }
}

impl Default for CacheProfiler {
    fn default() -> Self {
        Self::new()
    }
}

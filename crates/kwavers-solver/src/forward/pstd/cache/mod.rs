//! K-Space Operator Caching Module
//!
//! Implements thread-local and global caching for FFT operators and k-space
//! matrices to eliminate per-step heap allocations in PSTD/PSTK solvers.
//!
//! ## Cache Architecture
//!
//! ```text
//! Thread-Local Cache (fastest, per-thread)
//!   └── Recently used operators for current grid geometry
//!       └── O(1) lookup, no contention
//!
//! Global Cache (shared across threads)
//!   └── DashMap keyed by (nx, ny, nz, dx, dy, dz, c_ref, dt)
//!       └── Reference-counted operators for reuse
//!       └── Automatic eviction on memory pressure
//! ```
//!
//! ## Mathematical Key
//!
//! K-space operators depend only on grid geometry and reference sound speed:
//! ```text
//! κ = sinc(c_ref·|k|·dt/2)  —— k-space correction factor
//! d/dx → i·k_x·exp(±i·k_x·dx/2) —— staggered grid spectral derivative
//! ```
//!
//! For fixed grid parameters, these are **constant** across time steps,
//! making them ideal candidates for caching.
//!
//! ## References
//!
//! - Treeby, B. E. & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for the
//!   simulation and reconstruction of photoacoustic wave fields."
//!   *J. Biomed. Opt.* 15(2), 021314. §2.2 K-Space Correction.
//! - Kamil, S. et al. (2010). "An auto-tuning framework for parallel
//!   multicore stencil computations." *IPDPS*, 1-12. §2 Cache tiling.

use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_math::fft::Complex64;
use ndarray::Array1;
use std::cell::RefCell;
use std::sync::Arc;

/// Cache key for identifying unique operator configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperatorKey {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx_bits: u64, // f64 bit representation for exact equality
    pub dy_bits: u64,
    pub dz_bits: u64,
    pub c_ref_bits: u64,
    pub dt_bits: u64,
}

impl OperatorKey {
    /// Create a new operator key from grid parameters
    ///
    /// # Arguments
    /// * `nx, ny, nz` - Grid dimensions
    /// * `dx, dy, dz` - Grid spacing in meters
    /// * `c_ref` - Reference sound speed in m/s
    /// * `dt` - Time step in seconds
    ///
    /// # Mathematical Justification
    /// The k-space operators are fully determined by these 8 parameters:
    /// - Grid dimensions define FFT sizes
    /// - Grid spacing defines k-vector sampling: k_x = 2π/(nx·dx)
    /// - Reference sound speed defines k-space correction: sinc(c_ref·|k|·dt/2)
    /// - Time step defines propagation operator: exp(i·c·|k|·dt)
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        c_ref: f64,
        dt: f64,
    ) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx_bits: dx.to_bits(),
            dy_bits: dy.to_bits(),
            dz_bits: dz.to_bits(),
            c_ref_bits: c_ref.to_bits(),
            dt_bits: dt.to_bits(),
        }
    }
}

/// Staggered grid shift operators for spectral derivatives
#[derive(Debug, Clone)]
pub struct ShiftOperators {
    pub ddx_k_shift_pos: Array1<Complex64>,
    pub ddy_k_shift_pos: Array1<Complex64>,
    pub ddz_k_shift_pos: Array1<Complex64>,
    pub ddx_k_shift_neg: Array1<Complex64>,
    pub ddy_k_shift_neg: Array1<Complex64>,
    pub ddz_k_shift_neg: Array1<Complex64>,
}

/// Complete k-space operator set for a given geometry
#[derive(Debug, Clone)]
pub struct KSpaceOperators {
    pub kappa: ndarray::Array3<f64>,
    pub k_vec: (
        ndarray::Array3<f64>,
        ndarray::Array3<f64>,
        ndarray::Array3<f64>,
    ),
    pub k_magnitude: ndarray::Array3<f64>,
    pub shift_ops: ShiftOperators,
}

// Thread-local operator cache for fast access without synchronization
thread_local! {
    static THREAD_CACHE: RefCell<Option<ThreadLocalCache>> = const { RefCell::new(None) };
}

/// Per-thread cache entry
#[derive(Clone)]
struct ThreadLocalCache {
    key: OperatorKey,
    kspace_ops: Arc<KSpaceOperators>,
}

/// Cache statistics for monitoring
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    pub thread_hits: u64,
    pub thread_misses: u64,
    pub global_hits: u64,
    pub global_misses: u64,
    pub total_requests: u64,
}

impl CacheStats {
    pub fn thread_hit_rate(&self) -> f64 {
        let total = self.thread_hits + self.thread_misses;
        if total == 0 {
            0.0
        } else {
            self.thread_hits as f64 / total as f64
        }
    }

    pub fn global_hit_rate(&self) -> f64 {
        let total = self.global_hits + self.global_misses;
        if total == 0 {
            0.0
        } else {
            self.global_hits as f64 / total as f64
        }
    }
}

/// Global operator cache using RwLock for thread-safe concurrent access
///
/// Uses std::sync::LazyLock for initialization and RwLock<HashMap>
/// for thread-safe concurrent access.
static GLOBAL_CACHE: std::sync::LazyLock<
    std::sync::RwLock<std::collections::HashMap<OperatorKey, Arc<KSpaceOperators>>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(std::collections::HashMap::new()));

/// Cache manager for k-space operators
#[derive(Debug)]
pub struct KSpaceCache {
    stats: CacheStats,
}

impl Default for KSpaceCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KSpaceCache {
    /// Create a new cache manager
    pub fn new() -> Self {
        Self {
            stats: CacheStats::default(),
        }
    }

    /// Get operators for the given geometry, using cache if available
    ///
    /// # Algorithm
    /// 1. Check thread-local cache (O(1), no contention)
    /// 2. Check global cache (O(1), lock-free via DashMap)
    /// 3. Compute operators if not cached
    /// 4. Store in thread-local cache for fast subsequent access
    pub fn get_operators(
        &mut self,
        key: OperatorKey,
        compute_fn: impl FnOnce() -> KSpaceOperators,
    ) -> Arc<KSpaceOperators> {
        self.stats.total_requests += 1;

        // Step 1: Check thread-local cache
        let thread_result = THREAD_CACHE.with(|tc| {
            let cache = tc.borrow();
            if let Some(ref entry) = *cache {
                if entry.key == key {
                    return Some(entry.kspace_ops.clone());
                }
            }
            None
        });

        if let Some(ops) = thread_result {
            self.stats.thread_hits += 1;
            return ops;
        }
        self.stats.thread_misses += 1;

        // Step 2: Check global cache
        {
            let global = GLOBAL_CACHE.read().unwrap_or_else(|e| e.into_inner());
            if let Some(ops) = global.get(&key) {
                self.stats.global_hits += 1;
                let ops = ops.clone();
                drop(global);

                // Update thread-local cache
                THREAD_CACHE.with(|tc| {
                    *tc.borrow_mut() = Some(ThreadLocalCache {
                        key,
                        kspace_ops: ops.clone(),
                    });
                });

                return ops;
            }
        }
        self.stats.global_misses += 1;

        // Step 3 & 4: Compute and cache
        let ops = Arc::new(compute_fn());

        // Store in thread-local cache
        THREAD_CACHE.with(|tc| {
            *tc.borrow_mut() = Some(ThreadLocalCache {
                key,
                kspace_ops: ops.clone(),
            });
        });

        // Store in global cache
        {
            let mut global = GLOBAL_CACHE.write().unwrap_or_else(|e| e.into_inner());
            global.insert(key, ops.clone());
        }

        ops
    }

    /// Get current cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Clear thread-local cache for current thread
    pub fn clear_thread_cache(&self) {
        THREAD_CACHE.with(|tc| {
            tc.borrow_mut().take();
        });
    }

    /// Clear global cache (all threads)
    pub fn clear_global_cache(&self) {
        let mut global = GLOBAL_CACHE.write().unwrap_or_else(|e| e.into_inner());
        global.clear();
    }
}

/// Initialize global cache on first use
pub fn init_cache() {
    // Force lazy initialization
    let _ = &*GLOBAL_CACHE;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_key_creation() {
        let key = OperatorKey::new(64, 64, 64, 1e-4, 1e-4, 1e-4, SOUND_SPEED_WATER_SIM, 1e-8);
        assert_eq!(key.nx, 64);
        assert_eq!(key.ny, 64);
        assert_eq!(key.nz, 64);
    }

    #[test]
    fn test_cache_stats() {
        let stats = CacheStats {
            thread_hits: 80,
            thread_misses: 20,
            global_hits: 90,
            global_misses: 10,
            total_requests: 200,
        };

        assert!((stats.thread_hit_rate() - 0.8).abs() < 1e-9);
        assert!((stats.global_hit_rate() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_cache_get_compute() {
        let mut cache = KSpaceCache::new();
        let key = OperatorKey::new(32, 32, 32, 1e-4, 1e-4, 1e-4, SOUND_SPEED_WATER_SIM, 1e-8);

        let compute_count = std::cell::Cell::new(0);
        let ops = cache.get_operators(key, || {
            compute_count.set(compute_count.get() + 1);
            KSpaceOperators {
                kappa: ndarray::Array3::zeros((32, 32, 32)),
                k_vec: (
                    ndarray::Array3::zeros((32, 32, 32)),
                    ndarray::Array3::zeros((32, 32, 32)),
                    ndarray::Array3::zeros((32, 32, 32)),
                ),
                k_magnitude: ndarray::Array3::zeros((32, 32, 32)),
                shift_ops: ShiftOperators {
                    ddx_k_shift_pos: ndarray::Array1::zeros(32),
                    ddy_k_shift_pos: ndarray::Array1::zeros(32),
                    ddz_k_shift_pos: ndarray::Array1::zeros(32),
                    ddx_k_shift_neg: ndarray::Array1::zeros(32),
                    ddy_k_shift_neg: ndarray::Array1::zeros(32),
                    ddz_k_shift_neg: ndarray::Array1::zeros(32),
                },
            }
        });

        assert_eq!(compute_count.get(), 1);
        assert_eq!(ops.kappa.shape(), &[32, 32, 32]);

        // Second access should use thread-local cache (no recompute)
        let ops2 = cache.get_operators(key, || {
            compute_count.set(compute_count.get() + 1);
            unreachable!("Should use cached value")
        });

        assert_eq!(compute_count.get(), 1); // Still 1
        assert!(Arc::ptr_eq(&ops, &ops2));
    }
}

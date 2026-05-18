//! Parallel execution optimization

use crate::core::error::KwaversResult;
use rayon::prelude::*;
use std::sync::Arc;

/// Parallel execution optimizer
#[derive(Debug)]
pub struct ParallelOptimizer {
    num_threads: usize,
    chunk_size: usize,
}

impl Default for ParallelOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelOptimizer {
    /// Create a new parallel optimizer
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
            chunk_size: 1024, // Default chunk size for parallel iteration
        }
    }

    /// Set the number of threads for parallel execution
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn set_num_threads(&mut self, threads: usize) -> KwaversResult<()> {
        self.num_threads = threads;
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|e| {
                crate::core::error::KwaversError::System(
                    crate::core::error::SystemError::ThreadCreation {
                        reason: e.to_string(),
                    },
                )
            })?;
        Ok(())
    }

    /// Execute a function in parallel over a 3D domain
    pub fn parallel_3d<F>(&self, nx: usize, ny: usize, nz: usize, f: F)
    where
        F: Fn(usize, usize, usize) + Sync + Send,
    {
        let f = Arc::new(f);

        // Parallelize over the outermost dimension
        (0..nz).into_par_iter().for_each(|k| {
            for j in 0..ny {
                for i in 0..nx {
                    f(i, j, k);
                }
            }
        });
    }

    /// Parallel reduction operation
    pub fn parallel_reduce<T, F, G, R>(&self, data: &[T], identity: R, op: F, combine: G) -> R
    where
        T: Sync,
        R: Send + Sync + Clone,
        F: Fn(R, &T) -> R + Sync + Send,
        G: Fn(R, R) -> R + Sync + Send,
    {
        data.par_chunks(self.chunk_size)
            .map(|chunk| chunk.iter().fold(identity.clone(), &op))
            .reduce(|| identity.clone(), combine)
    }

    /// Parallel map operation on arrays
    pub fn parallel_map<T, U, F>(&self, input: &[T], f: F) -> Vec<U>
    where
        T: Sync,
        U: Send,
        F: Fn(&T) -> U + Sync + Send,
    {
        input
            .par_chunks(self.chunk_size)
            .flat_map(|chunk| chunk.iter().map(&f).collect::<Vec<_>>())
            .collect()
    }

    /// Get optimal chunk size based on data size and thread count
    #[must_use]
    pub fn optimal_chunk_size(&self, data_size: usize) -> usize {
        // Aim for at least 4 chunks per thread for load balancing
        let min_chunks = self.num_threads * 4;
        let chunk_size = (data_size / min_chunks).max(1);

        // But not smaller than cache line to avoid false sharing
        chunk_size.max(8) // 8 f64s = 64 bytes = typical cache line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── ParallelOptimizer exact value tests ──────────────────────────────────

    /// `parallel_reduce` computes exact sum over integer slice.
    ///
    /// [1, 2, 3, 4, 5] with identity=0, op=add, combine=add → 15.
    #[test]
    fn parallel_optimizer_reduce_sum_exact() {
        let optimizer = ParallelOptimizer::new();
        let data = vec![1i32, 2, 3, 4, 5];
        let sum = optimizer.parallel_reduce(&data, 0i32, |acc, &x| acc + x, |a, b| a + b);
        assert_eq!(sum, 15, "parallel_reduce sum of [1..5] must be 15");
    }

    /// `parallel_reduce` with multiplication over [1,2,3,4] gives 24.
    ///
    /// identity=1, op=mul, combine=mul → 1×2×3×4 = 24.
    #[test]
    fn parallel_optimizer_reduce_product_exact() {
        let optimizer = ParallelOptimizer::new();
        let data = vec![1i32, 2, 3, 4];
        let product = optimizer.parallel_reduce(&data, 1i32, |acc, &x| acc * x, |a, b| a * b);
        assert_eq!(
            product, 24,
            "parallel_reduce product of [1,2,3,4] must be 24"
        );
    }

    /// `parallel_map` doubles each element exactly.
    ///
    /// [1, 2, 3] → [2, 4, 6].
    #[test]
    fn parallel_optimizer_map_double_exact() {
        let optimizer = ParallelOptimizer::new();
        let data = vec![1i32, 2, 3];
        let result = optimizer.parallel_map(&data, |&x| x * 2);
        assert_eq!(
            result,
            vec![2i32, 4, 6],
            "parallel_map double must yield [2,4,6]"
        );
    }

    /// `parallel_map` on empty input returns empty output.
    #[test]
    fn parallel_optimizer_map_empty_input_returns_empty() {
        let optimizer = ParallelOptimizer::new();
        let data: Vec<i32> = vec![];
        let result: Vec<i32> = optimizer.parallel_map(&data, |&x| x);
        assert!(
            result.is_empty(),
            "map of empty slice must return empty Vec"
        );
    }

    /// `parallel_3d` visits every (i,j,k) in [0,nx)×[0,ny)×[0,nz) exactly once.
    ///
    /// Uses an atomic counter array; every cell must be incremented exactly once.
    #[test]
    fn parallel_3d_visits_every_cell_exactly_once() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let nx = 3usize;
        let ny = 4usize;
        let nz = 5usize;
        let optimizer = ParallelOptimizer::new();

        let counts = Arc::new(
            (0..nx * ny * nz)
                .map(|_| AtomicU32::new(0))
                .collect::<Vec<_>>(),
        );
        let counts_clone = Arc::clone(&counts);

        optimizer.parallel_3d(nx, ny, nz, move |i, j, k| {
            counts_clone[i * ny * nz + j * nz + k].fetch_add(1, Ordering::Relaxed);
        });

        for (idx, cell) in counts.iter().enumerate() {
            assert_eq!(
                cell.load(Ordering::Relaxed),
                1,
                "cell {idx} visited {} times (expected 1)",
                cell.load(Ordering::Relaxed)
            );
        }
    }

    /// `optimal_chunk_size` returns value ≥ 8 (cache-line guard) for any input.
    #[test]
    fn parallel_optimizer_chunk_size_at_least_eight() {
        let optimizer = ParallelOptimizer::new();
        // Even tiny data sizes must return >= 8.
        for size in [0, 1, 7, 8, 100, 10_000] {
            let chunk = optimizer.optimal_chunk_size(size);
            assert!(
                chunk >= 8,
                "optimal_chunk_size({size}) = {chunk} (expected >= 8)"
            );
        }
    }
}

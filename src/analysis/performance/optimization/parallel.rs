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
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
            chunk_size: 1024, // Default chunk size for parallel iteration
        }
    }

    /// Set the number of threads for parallel execution
    pub fn set_num_threads(&mut self, threads: usize) -> KwaversResult<()> {
        self.num_threads = threads;
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|e| {
                crate::domain::core::error::KwaversError::System(
                    crate::domain::core::error::SystemError::ThreadCreation {
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

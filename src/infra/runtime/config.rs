//! Performance tuning configuration

use serde::{Deserialize, Serialize};

/// Performance tuning parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceParameters {
    /// Number of threads for parallel execution
    pub num_threads: Option<usize>,
    /// Enable GPU acceleration
    pub use_gpu: bool,
    /// GPU device index
    pub gpu_device: usize,
    /// Cache size in MB
    pub cache_size: usize,
    /// Chunk size for parallel operations
    pub chunk_size: usize,
    /// Enable SIMD optimizations
    pub use_simd: bool,
    /// Memory pool size in MB
    pub memory_pool: usize,
}

impl PerformanceParameters {
    /// Validate performance parameters
    pub fn validate(&self) -> crate::domain::core::error::KwaversResult<()> {
        if let Some(threads) = self.num_threads {
            if threads == 0 {
                return Err(crate::domain::core::error::ConfigError::InvalidValue {
                    parameter: "num_threads".to_string(),
                    value: "0".to_string(),
                    constraint: "Must be positive".to_string(),
                }
                .into());
            }
        }

        if self.cache_size == 0 {
            return Err(crate::domain::core::error::ConfigError::InvalidValue {
                parameter: "cache_size".to_string(),
                value: "0".to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        if self.chunk_size == 0 {
            return Err(crate::domain::core::error::ConfigError::InvalidValue {
                parameter: "chunk_size".to_string(),
                value: "0".to_string(),
                constraint: "Must be positive".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for PerformanceParameters {
    fn default() -> Self {
        Self {
            num_threads: None, // Use all available
            use_gpu: false,
            gpu_device: 0,
            cache_size: 256, // MB
            chunk_size: 1024,
            use_simd: true,
            memory_pool: 1024, // MB
        }
    }
}

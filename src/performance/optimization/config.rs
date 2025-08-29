//! Optimization configuration

/// Performance optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable SIMD vectorization
    pub enable_simd: bool,
    /// SIMD instruction set to use
    pub simd_level: SimdLevel,
    /// Enable cache blocking
    pub cache_blocking: bool,
    /// Cache block size (in elements)
    pub cache_block_size: usize,
    /// Enable memory prefetching
    pub prefetching: bool,
    /// Prefetch distance (in cache lines)
    pub prefetch_distance: usize,
    /// Enable kernel fusion on GPU
    pub kernel_fusion: bool,
    /// Enable asynchronous execution
    pub async_execution: bool,
    /// Number of GPU streams
    pub gpu_streams: usize,
    /// Enable multi-GPU
    pub multi_gpu: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            simd_level: SimdLevel::detect_best(),
            cache_blocking: true,
            cache_block_size: 64, // Typical L1 cache line
            prefetching: true,
            prefetch_distance: 8,
            kernel_fusion: true,
            async_execution: true,
            gpu_streams: 4,
            multi_gpu: true,
        }
    }
}

/// SIMD instruction set level
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimdLevel {
    /// No SIMD
    None,
    /// SSE2 (128-bit)
    Sse2,
    /// AVX (256-bit)
    Avx,
    /// AVX2 (256-bit with FMA)
    Avx2,
    /// AVX-512 (512-bit)
    Avx512,
    /// ARM NEON
    Neon,
}

impl SimdLevel {
    /// Detect the best available SIMD level for the current CPU
    pub fn detect_best() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Self::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
            if is_x86_feature_detected!("avx") {
                return Self::Avx;
            }
            if is_x86_feature_detected!("sse2") {
                return Self::Sse2;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Self::Neon;
            }
        }

        Self::None
    }

    /// Get the vector width in f64 elements
    pub fn vector_width(&self) -> usize {
        match self {
            Self::None => 1,
            Self::Sse2 => 2,   // 128-bit / 64-bit = 2
            Self::Avx => 4,    // 256-bit / 64-bit = 4
            Self::Avx2 => 4,   // 256-bit / 64-bit = 4
            Self::Avx512 => 8, // 512-bit / 64-bit = 8
            Self::Neon => 2,   // 128-bit / 64-bit = 2
        }
    }
}

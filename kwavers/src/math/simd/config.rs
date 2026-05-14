//! SIMD capability detection and configuration record.

/// SIMD capability detection and configuration
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// SIMD instruction set level
    pub level: SimdLevel,
    /// Vector width in elements
    pub vector_width: usize,
    /// Alignment requirement in bytes
    pub alignment: usize,
    /// Whether SIMD is available and enabled
    pub enabled: bool,
}

/// SIMD instruction set levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// No SIMD (scalar operations)
    Scalar,
    /// SSE/SSE2 (128-bit vectors)
    Sse2,
    /// AVX/AVX2 (256-bit vectors)
    Avx2,
    /// AVX-512 (512-bit vectors)
    Avx512,
    /// ARM NEON
    Neon,
    /// Portable SIMD (std::simd)
    Portable,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self::detect()
    }
}

impl SimdConfig {
    /// Detect available SIMD capabilities
    #[must_use]
    pub fn detect() -> Self {
        // Check for x86 SIMD features
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Self {
                    level: SimdLevel::Avx512,
                    vector_width: 16, // 512 bits / 32 bits per f32
                    alignment: 64,
                    enabled: true,
                };
            }

            if is_x86_feature_detected!("avx2") {
                return Self {
                    level: SimdLevel::Avx2,
                    vector_width: 8, // 256 bits / 32 bits per f32
                    alignment: 32,
                    enabled: true,
                };
            }

            if is_x86_feature_detected!("sse2") {
                return Self {
                    level: SimdLevel::Sse2,
                    vector_width: 4, // 128 bits / 32 bits per f32
                    alignment: 16,
                    enabled: true,
                };
            }
        }

        // Check for ARM NEON
        #[cfg(target_arch = "aarch64")]
        {
            return Self {
                level: SimdLevel::Neon,
                vector_width: 4, // NEON 128-bit
                alignment: 16,
                enabled: true,
            };
        }

        // Fallback to scalar operations
        Self {
            level: SimdLevel::Scalar,
            vector_width: 1,
            alignment: std::mem::align_of::<f32>(),
            enabled: false,
        }
    }
}

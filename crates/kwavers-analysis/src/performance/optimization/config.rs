//! Optimization configuration

/// SIMD instruction set level
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PerfOptSimdLevel {
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

impl PerfOptSimdLevel {
    /// Detect the best available SIMD level for the current CPU
    #[must_use]
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
    #[must_use]
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

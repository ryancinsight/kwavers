//! SIMD performance estimation and metrics record.

use super::config::{MathSimdLevel, SimdConfig};

/// SIMD performance utilities
#[derive(Debug)]
pub struct SimdPerformance;

impl SimdPerformance {
    /// Get SIMD performance metrics
    #[must_use]
    pub fn get_metrics() -> SimdMetrics {
        let config = SimdConfig::detect();

        SimdMetrics {
            detected_level: config.level,
            vector_width: config.vector_width,
            alignment_bytes: config.alignment,
            estimated_speedup: Self::estimate_speedup(config.level),
        }
    }

    /// Estimate performance speedup for given SIMD level
    fn estimate_speedup(level: MathSimdLevel) -> f64 {
        match level {
            MathSimdLevel::Scalar => 1.0,
            MathSimdLevel::Sse2 => 2.5,
            MathSimdLevel::Avx2 => 4.0,
            MathSimdLevel::Avx512 => 8.0,
            MathSimdLevel::Neon => 3.0,
            MathSimdLevel::Portable => 4.0,
        }
    }
}

/// SIMD performance metrics
#[derive(Debug, Clone)]
pub struct SimdMetrics {
    pub detected_level: MathSimdLevel,
    pub vector_width: usize,
    pub alignment_bytes: usize,
    pub estimated_speedup: f64,
}

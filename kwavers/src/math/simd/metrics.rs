//! SIMD performance estimation and metrics record.

use super::config::{SimdConfig, SimdLevel};

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
    fn estimate_speedup(level: SimdLevel) -> f64 {
        match level {
            SimdLevel::Scalar => 1.0,
            SimdLevel::Sse2 => 2.5,
            SimdLevel::Avx2 => 4.0,
            SimdLevel::Avx512 => 8.0,
            SimdLevel::Neon => 3.0,
            SimdLevel::Portable => 4.0,
        }
    }
}

/// SIMD performance metrics
#[derive(Debug, Clone)]
pub struct SimdMetrics {
    pub detected_level: SimdLevel,
    pub vector_width: usize,
    pub alignment_bytes: usize,
    pub estimated_speedup: f64,
}

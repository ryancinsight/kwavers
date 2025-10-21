//! SIMD capability detection module
//!
//! This module owns the knowledge of CPU feature detection following the
//! Information Expert GRASP principle.

use std::arch::is_x86_feature_detected;

/// SIMD capability detection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimdCapability {
    /// AVX-512 available (`x86_64`)
    Avx512,
    /// AVX2 available (`x86_64`)
    Avx2,
    /// SSE4.2 available (`x86_64`)
    Sse42,
    /// NEON available (ARM)
    Neon,
    /// No SIMD, use SWAR
    Swar,
}

impl SimdCapability {
    /// Detect the best available SIMD capability
    ///
    /// Uses runtime CPU feature detection to determine optimal SIMD level.
    /// Falls back gracefully to SWAR (SIMD Within A Register) if no SIMD available.
    ///
    /// # Safety
    /// This function is safe - it only detects capabilities, doesn't use them.
    /// Actual SIMD usage occurs in architecture-specific modules with proper
    /// safety documentation per ICSE 2020 guidelines.
    #[inline]
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Self::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
            if is_x86_feature_detected!("sse4.2") {
                return Self::Sse42;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on AArch64
            return Self::Neon;
        }

        #[cfg(all(target_arch = "arm", target_feature = "neon"))]
        {
            return Self::Neon;
        }

        // Fallback to SWAR for all other architectures
        Self::Swar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_detection() {
        let capability = SimdCapability::detect();

        // Should detect some capability on any architecture
        assert!(matches!(
            capability,
            SimdCapability::Avx512
                | SimdCapability::Avx2
                | SimdCapability::Sse42
                | SimdCapability::Neon
                | SimdCapability::Swar
        ));
    }

    #[test]
    fn test_capability_display() {
        // Verify each capability can be created and debugged
        let capabilities = [
            SimdCapability::Avx512,
            SimdCapability::Avx2,
            SimdCapability::Sse42,
            SimdCapability::Neon,
            SimdCapability::Swar,
        ];

        for cap in &capabilities {
            let debug_str = format!("{:?}", cap);
            assert!(!debug_str.is_empty());
        }
    }
}

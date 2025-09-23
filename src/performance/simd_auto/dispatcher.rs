//! SIMD operation dispatcher
//!
//! This module owns the dispatching logic, following the Information Expert
//! GRASP principle by encapsulating dispatch knowledge.

use super::capability::SimdCapability;
use super::x86_64;
use ndarray::Array3;

/// Automatic SIMD dispatcher
#[derive(Debug)]
pub struct SimdAuto {
    capability: SimdCapability,
}

impl SimdAuto {
    /// Create new SIMD dispatcher with auto-detected capability
    #[must_use]
    pub fn new() -> Self {
        Self {
            capability: SimdCapability::detect(),
        }
    }

    /// Add two arrays in-place with optimal SIMD dispatch
    ///
    /// # Safety
    /// This method dispatches to unsafe SIMD implementations but maintains
    /// safety invariants through careful bounds checking and validation.
    pub fn add_inplace(&self, a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        match self.capability {
            SimdCapability::Avx512 => x86_64::avx512::add_arrays(a, b, out),
            SimdCapability::Avx2 => x86_64::avx2::add_arrays(a, b, out),
            SimdCapability::Sse42 => x86_64::sse42::add_arrays(a, b, out),
            SimdCapability::Neon => {
                #[cfg(target_arch = "aarch64")]
                {
                    super::aarch64::neon::add_arrays(a, b, out);
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    self.fallback_add(a, b, out);
                }
            }
            SimdCapability::Swar => self.fallback_add(a, b, out),
        }
    }

    /// Scale array in-place with optimal SIMD dispatch
    pub fn scale_inplace(&self, array: &mut Array3<f64>, scalar: f64) {
        match self.capability {
            SimdCapability::Avx512 => x86_64::avx512::scale_array(array, scalar),
            SimdCapability::Avx2 => x86_64::avx2::scale_array(array, scalar),
            SimdCapability::Sse42 => x86_64::sse42::scale_array(array, scalar),
            SimdCapability::Neon => {
                #[cfg(target_arch = "aarch64")]
                {
                    super::aarch64::neon::scale_array(array, scalar);
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    self.fallback_scale(array, scalar);
                }
            }
            SimdCapability::Swar => self.fallback_scale(array, scalar),
        }
    }

    /// Fused multiply-add operation with optimal SIMD dispatch
    pub fn fma_inplace(
        &self,
        a: &Array3<f64>,
        b: &Array3<f64>,
        c: &mut Array3<f64>,
        multiplier: f64,
    ) {
        match self.capability {
            SimdCapability::Avx512 => x86_64::avx512::fma_arrays(a, b, c, multiplier),
            SimdCapability::Avx2 => x86_64::avx2::fma_arrays(a, b, c, multiplier),
            SimdCapability::Sse42 => x86_64::sse42::fma_arrays(a, b, c, multiplier),
            SimdCapability::Neon => {
                #[cfg(target_arch = "aarch64")]
                {
                    super::aarch64::neon::fma_arrays(a, b, c, multiplier);
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    self.fallback_fma(a, b, c, multiplier);
                }
            }
            SimdCapability::Swar => self.fallback_fma(a, b, c, multiplier),
        }
    }

    // Fallback implementations using standard operations
    fn fallback_add(&self, a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        ndarray::Zip::from(out).and(a).and(b).for_each(|out, &a, &b| {
            *out = a + b;
        });
    }

    fn fallback_scale(&self, array: &mut Array3<f64>, scalar: f64) {
        array.mapv_inplace(|x| x * scalar);
    }

    fn fallback_fma(&self, a: &Array3<f64>, b: &Array3<f64>, c: &mut Array3<f64>, multiplier: f64) {
        ndarray::Zip::from(c).and(a).and(b).for_each(|c, &a, &b| {
            *c += multiplier * a * b;
        });
    }
}

impl Default for SimdAuto {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_simd_auto_creation() {
        let simd = SimdAuto::new();
        // Should create successfully with any detected capability
        assert!(matches!(
            simd.capability,
            SimdCapability::Avx512 | 
            SimdCapability::Avx2 | 
            SimdCapability::Sse42 | 
            SimdCapability::Neon | 
            SimdCapability::Swar
        ));
    }

    #[test]
    fn test_add_operation() {
        let simd = SimdAuto::new();
        let a = Array3::<f64>::ones((2, 2, 2));
        let b = Array3::<f64>::ones((2, 2, 2)) * 2.0;
        let mut out = Array3::<f64>::zeros((2, 2, 2));

        simd.add_inplace(&a, &b, &mut out);

        // Should result in array of 3.0s
        assert!(out.iter().all(|&x| (x - 3.0).abs() < 1e-10));
    }

    #[test]
    fn test_scale_operation() {
        let simd = SimdAuto::new();
        let mut array = Array3::<f64>::ones((2, 2, 2)) * 2.0;

        simd.scale_inplace(&mut array, 3.0);

        // Should result in array of 6.0s
        assert!(array.iter().all(|&x| (x - 6.0).abs() < 1e-10));
    }
}
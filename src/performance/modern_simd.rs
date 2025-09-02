//! Modern SIMD implementation using safe patterns and portable_simd when stable
//!
//! References:
//! - "SIMD Programming" by Intel (2023)
//! - Rust portable SIMD RFC: https://github.com/rust-lang/rfcs/pull/2977

use ndarray::{Array3, ArrayView3, ArrayViewMut3, Zip};
use rayon::prelude::*;

/// SIMD lane width for f64 on common architectures
#[cfg(target_arch = "x86_64")]
const SIMD_LANES: usize = 4; // AVX2: 256 bits / 64 bits = 4

#[cfg(not(target_arch = "x86_64"))]
const SIMD_LANES: usize = 2; // Fallback for other architectures

/// Vectorized field operations using safe patterns
pub struct SimdOps;

impl SimdOps {
    /// Add two fields using SIMD-friendly patterns
    pub fn add_fields(a: ArrayView3<f64>, b: ArrayView3<f64>, mut out: ArrayViewMut3<f64>) {
        // Use ndarray's parallel zip for automatic vectorization
        Zip::from(&mut out)
            .and(&a)
            .and(&b)
            .par_for_each(|o, &a_val, &b_val| {
                *o = a_val + b_val;
            });
    }

    /// Scale field by scalar using SIMD-friendly patterns
    pub fn scale_field(field: ArrayView3<f64>, scalar: f64, mut out: ArrayViewMut3<f64>) {
        Zip::from(&mut out).and(&field).par_for_each(|o, &f_val| {
            *o = f_val * scalar;
        });
    }

    /// Compute field norm using SIMD-friendly reduction
    pub fn field_norm(field: ArrayView3<f64>) -> f64 {
        field
            .as_slice()
            .unwrap()
            .par_chunks(SIMD_LANES * 16) // Process multiple SIMD vectors at once
            .map(|chunk| chunk.iter().map(|&x| x * x).sum::<f64>())
            .sum::<f64>()
            .sqrt()
    }

    /// Compute dot product using SIMD-friendly patterns
    pub fn dot_product(a: ArrayView3<f64>, b: ArrayView3<f64>) -> f64 {
        a.as_slice()
            .unwrap()
            .par_chunks(SIMD_LANES * 16)
            .zip(b.as_slice().unwrap().par_chunks(SIMD_LANES * 16))
            .map(|(a_chunk, b_chunk)| {
                a_chunk
                    .iter()
                    .zip(b_chunk.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>()
            })
            .sum()
    }

    /// Apply stencil operation with SIMD-friendly access patterns
    pub fn apply_3d_stencil<const S: usize>(
        input: ArrayView3<f64>,
        mut output: ArrayViewMut3<f64>,
        stencil: &[f64; S],
    ) where
        [(); S]: Sized,
    {
        let (nx, ny, nz) = input.dim();
        let half_s = S / 2;

        // Process interior points - using standard iteration for simplicity
        for i in half_s..nx - half_s {
            for j in half_s..ny - half_s {
                // Process z-dimension in chunks for better cache usage
                for k in half_s..nz - half_s {
                    let mut sum = 0.0;

                    // Apply stencil
                    for (di, &coeff) in stencil.iter().enumerate() {
                        let idx_i = (i + di).saturating_sub(half_s);
                        sum += input[[idx_i, j, k]] * coeff;
                    }

                    output[[i, j, k]] = sum;
                }
            }
        }
    }

    /// Fused multiply-add operation
    pub fn fma_fields(
        a: ArrayView3<f64>,
        b: ArrayView3<f64>,
        c: ArrayView3<f64>,
        mut out: ArrayViewMut3<f64>,
    ) {
        Zip::from(&mut out)
            .and(&a)
            .and(&b)
            .and(&c)
            .par_for_each(|o, &a_val, &b_val, &c_val| {
                *o = a_val.mul_add(b_val, c_val); // Uses FMA instruction when available
            });
    }
}

/// SWAR (SIMD Within A Register) operations for portability
pub mod swar {
    use super::*;

    /// Compute sum of 4 f64 values using integer operations
    pub fn sum4_swar(values: [f64; 4]) -> f64 {
        // Convert to bits for manipulation
        let bits: [u64; 4] = [
            values[0].to_bits(),
            values[1].to_bits(),
            values[2].to_bits(),
            values[3].to_bits(),
        ];

        // This is a simplified example - real SWAR would do more
        // For now, just sum normally
        values.iter().sum()
    }

    /// Parallel maximum using SWAR techniques
    pub fn max4_swar(values: [f64; 4]) -> f64 {
        values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    }
}

/// Architecture-specific optimizations
#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
    use super::*;

    /// Check if AVX2 is available
    pub fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    /// Check if AVX-512 is available
    pub fn has_avx512() -> bool {
        is_x86_feature_detected!("avx512f")
    }

    /// Select best SIMD width based on CPU features
    pub fn optimal_simd_width() -> usize {
        if has_avx512() {
            8 // 512 bits / 64 bits
        } else if has_avx2() {
            4 // 256 bits / 64 bits
        } else {
            2 // SSE2: 128 bits / 64 bits
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_add() {
        let a = Array3::from_shape_fn((16, 16, 16), |(i, j, k)| (i + j + k) as f64);
        let b = Array3::ones((16, 16, 16));
        let mut out = Array3::zeros((16, 16, 16));

        SimdOps::add_fields(a.view(), b.view(), out.view_mut());

        Zip::from(&out).and(&a).and(&b).for_each(|&o, &a, &b| {
            assert_relative_eq!(o, a + b);
        });
    }

    #[test]
    fn test_field_norm() {
        let field = Array3::from_shape_fn((32, 32, 32), |(i, j, k)| {
            if i == 16 && j == 16 && k == 16 {
                1.0
            } else {
                0.0
            }
        });

        let norm = SimdOps::field_norm(field.view());
        assert_relative_eq!(norm, 1.0);
    }
}

/// When portable_simd becomes stable, we can use this module
#[cfg(feature = "portable_simd")]
pub mod portable {
    use std::simd::{f64x4, f64x8, SimdFloat, StdFloat};

    /// Add arrays using portable SIMD
    pub fn add_arrays_simd(a: &[f64], b: &[f64], out: &mut [f64]) {
        let chunks = a.len() / 4;

        for i in 0..chunks {
            let idx = i * 4;
            let a_vec = f64x4::from_slice(&a[idx..]);
            let b_vec = f64x4::from_slice(&b[idx..]);
            let result = a_vec + b_vec;
            result.copy_to_slice(&mut out[idx..]);
        }

        // Handle remainder
        for i in chunks * 4..a.len() {
            out[i] = a[i] + b[i];
        }
    }
}

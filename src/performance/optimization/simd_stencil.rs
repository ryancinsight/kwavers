//! SIMD-optimized stencil operations for FDTD
//!
//! Uses safe SIMD operations with architecture detection for portable performance.

#![allow(unsafe_code)] // SIMD intrinsics require unsafe for performance

use ndarray::{ArrayView3, ArrayViewMut3};
use std::arch::x86_64::*;

/// SIMD-optimized Laplacian computation
#[derive(Debug)]
pub struct SimdLaplacian {
    /// Grid spacing squared
    dx2_inv: f64,
    dy2_inv: f64,
    dz2_inv: f64,
}

impl SimdLaplacian {
    /// Create new SIMD Laplacian operator
    pub fn new(dx: f64, dy: f64, dz: f64) -> Self {
        Self {
            dx2_inv: 1.0 / (dx * dx),
            dy2_inv: 1.0 / (dy * dy),
            dz2_inv: 1.0 / (dz * dz),
        }
    }

    /// Apply Laplacian using SIMD operations
    pub fn apply(&self, input: ArrayView3<f64>, output: ArrayViewMut3<f64>) {
        let (nx, ny, nz) = input.dim();

        // Check CPU features at runtime
        if is_x86_feature_detected!("avx2") {
            unsafe { self.apply_avx2(input, output) }
        } else if is_x86_feature_detected!("sse2") {
            unsafe { self.apply_sse2(input, output) }
        } else {
            self.apply_scalar(input, output)
        }
    }

    /// AVX2 implementation (4 doubles at once)
    #[target_feature(enable = "avx2")]
    unsafe fn apply_avx2(&self, input: ArrayView3<f64>, mut output: ArrayViewMut3<f64>) {
        let (nx, ny, nz) = input.dim();
        let dx2_inv = _mm256_set1_pd(self.dx2_inv);
        let dy2_inv = _mm256_set1_pd(self.dy2_inv);
        let dz2_inv = _mm256_set1_pd(self.dz2_inv);
        let neg_two = _mm256_set1_pd(-2.0);

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                // Process 4 elements at once in z-direction
                let mut k = 1;
                while k + 3 < nz - 1 {
                    // Load center values
                    let center = _mm256_loadu_pd(&input[[i, j, k]]);

                    // X-direction stencil
                    let xp = _mm256_loadu_pd(&input[[i + 1, j, k]]);
                    let xm = _mm256_loadu_pd(&input[[i - 1, j, k]]);
                    let x_term = _mm256_mul_pd(
                        _mm256_add_pd(_mm256_add_pd(xp, xm), _mm256_mul_pd(center, neg_two)),
                        dx2_inv,
                    );

                    // Y-direction stencil
                    let yp = _mm256_loadu_pd(&input[[i, j + 1, k]]);
                    let ym = _mm256_loadu_pd(&input[[i, j - 1, k]]);
                    let y_term = _mm256_mul_pd(
                        _mm256_add_pd(_mm256_add_pd(yp, ym), _mm256_mul_pd(center, neg_two)),
                        dy2_inv,
                    );

                    // Z-direction stencil (scalar for now due to strided access)
                    let mut z_vals = [0.0; 4];
                    for dk in 0..4 {
                        z_vals[dk] = (input[[i, j, k + dk + 1]] - 2.0 * input[[i, j, k + dk]]
                            + input[[i, j, k + dk - 1]])
                            * self.dz2_inv;
                    }
                    let z_term = _mm256_loadu_pd(&z_vals[0]);

                    // Sum all terms
                    let result = _mm256_add_pd(_mm256_add_pd(x_term, y_term), z_term);

                    // Store result
                    _mm256_storeu_pd(&mut output[[i, j, k]], result);

                    k += 4;
                }

                // Handle remaining elements
                while k < nz - 1 {
                    output[[i, j, k]] = self.compute_laplacian_point(&input, i, j, k);
                    k += 1;
                }
            }
        }
    }

    /// SSE2 implementation (2 doubles at once)
    #[target_feature(enable = "sse2")]
    unsafe fn apply_sse2(&self, input: ArrayView3<f64>, mut output: ArrayViewMut3<f64>) {
        let (nx, ny, nz) = input.dim();
        let dx2_inv = _mm_set1_pd(self.dx2_inv);
        let dy2_inv = _mm_set1_pd(self.dy2_inv);
        let dz2_inv = _mm_set1_pd(self.dz2_inv);
        let neg_two = _mm_set1_pd(-2.0);

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let mut k = 1;
                while k + 1 < nz - 1 {
                    // Process 2 elements at once
                    let center = _mm_loadu_pd(&input[[i, j, k]]);

                    // X-direction
                    let xp = _mm_loadu_pd(&input[[i + 1, j, k]]);
                    let xm = _mm_loadu_pd(&input[[i - 1, j, k]]);
                    let x_term = _mm_mul_pd(
                        _mm_add_pd(_mm_add_pd(xp, xm), _mm_mul_pd(center, neg_two)),
                        dx2_inv,
                    );

                    // Y-direction
                    let yp = _mm_loadu_pd(&input[[i, j + 1, k]]);
                    let ym = _mm_loadu_pd(&input[[i, j - 1, k]]);
                    let y_term = _mm_mul_pd(
                        _mm_add_pd(_mm_add_pd(yp, ym), _mm_mul_pd(center, neg_two)),
                        dy2_inv,
                    );

                    // Z-direction (scalar)
                    let mut z_vals = [0.0; 2];
                    for dk in 0..2 {
                        z_vals[dk] = (input[[i, j, k + dk + 1]] - 2.0 * input[[i, j, k + dk]]
                            + input[[i, j, k + dk - 1]])
                            * self.dz2_inv;
                    }
                    let z_term = _mm_loadu_pd(&z_vals[0]);

                    // Sum and store
                    let result = _mm_add_pd(_mm_add_pd(x_term, y_term), z_term);
                    _mm_storeu_pd(&mut output[[i, j, k]], result);

                    k += 2;
                }

                // Handle remainder
                if k < nz - 1 {
                    output[[i, j, k]] = self.compute_laplacian_point(&input, i, j, k);
                }
            }
        }
    }

    /// Scalar fallback implementation
    fn apply_scalar(&self, input: ArrayView3<f64>, mut output: ArrayViewMut3<f64>) {
        let (nx, ny, nz) = input.dim();

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    output[[i, j, k]] = self.compute_laplacian_point(&input, i, j, k);
                }
            }
        }
    }

    /// Compute Laplacian at a single point
    #[inline(always)]
    fn compute_laplacian_point(
        &self,
        input: &ArrayView3<f64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> f64 {
        let center = input[[i, j, k]];

        (input[[i + 1, j, k]] - 2.0 * center + input[[i - 1, j, k]]) * self.dx2_inv
            + (input[[i, j + 1, k]] - 2.0 * center + input[[i, j - 1, k]]) * self.dy2_inv
            + (input[[i, j, k + 1]] - 2.0 * center + input[[i, j, k - 1]]) * self.dz2_inv
    }
}

/// SWAR (SIMD Within A Register) fallback for non-x86 architectures
#[derive(Debug)]
pub struct SwarLaplacian {
    dx2_inv: f64,
    dy2_inv: f64,
    dz2_inv: f64,
}

impl SwarLaplacian {
    pub fn new(dx: f64, dy: f64, dz: f64) -> Self {
        Self {
            dx2_inv: 1.0 / (dx * dx),
            dy2_inv: 1.0 / (dy * dy),
            dz2_inv: 1.0 / (dz * dz),
        }
    }

    /// Apply using SWAR techniques for portable optimization
    pub fn apply(&self, input: ArrayView3<f64>, mut output: ArrayViewMut3<f64>) {
        let (nx, ny, nz) = input.dim();

        // Process in cache-friendly blocks
        const BLOCK_SIZE: usize = 8;

        for i_block in (1..nx - 1).step_by(BLOCK_SIZE) {
            let i_end = (i_block + BLOCK_SIZE).min(nx - 1);

            for j_block in (1..ny - 1).step_by(BLOCK_SIZE) {
                let j_end = (j_block + BLOCK_SIZE).min(ny - 1);

                for k_block in (1..nz - 1).step_by(BLOCK_SIZE) {
                    let k_end = (k_block + BLOCK_SIZE).min(nz - 1);

                    // Process block
                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            for k in k_block..k_end {
                                let center = input[[i, j, k]];

                                // Compute all differences at once
                                let dx = input[[i + 1, j, k]] + input[[i - 1, j, k]] - 2.0 * center;
                                let dy = input[[i, j + 1, k]] + input[[i, j - 1, k]] - 2.0 * center;
                                let dz = input[[i, j, k + 1]] + input[[i, j, k - 1]] - 2.0 * center;

                                output[[i, j, k]] =
                                    dx * self.dx2_inv + dy * self.dy2_inv + dz * self.dz2_inv;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_laplacian_correctness() {
        let input = Array3::from_shape_fn((32, 32, 32), |(i, j, k)| {
            (i as f64).sin() * (j as f64).cos() * (k as f64)
        });
        let mut output_simd = Array3::zeros((32, 32, 32));
        let mut output_scalar = Array3::zeros((32, 32, 32));

        let op = SimdLaplacian::new(1.0, 1.0, 1.0);
        op.apply(input.view(), output_simd.view_mut());
        op.apply_scalar(input.view(), output_scalar.view_mut());

        // Compare SIMD and scalar results
        for i in 1..31 {
            for j in 1..31 {
                for k in 1..31 {
                    assert_relative_eq!(
                        output_simd[[i, j, k]],
                        output_scalar[[i, j, k]],
                        epsilon = 1e-10
                    );
                }
            }
        }
    }
}

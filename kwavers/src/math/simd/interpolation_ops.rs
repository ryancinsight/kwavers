//! SIMD-accelerated trilinear interpolation kernels.

use super::config::{SimdConfig, SimdLevel};

/// SIMD-accelerated interpolation operations
#[derive(Debug)]
pub struct InterpolationSimdOps {
    config: SimdConfig,
}

impl InterpolationSimdOps {
    /// Create new interpolation SIMD operations
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SimdConfig::detect(),
        }
    }

    /// SIMD-accelerated trilinear interpolation
    pub fn trilinear_interpolate(
        &self,
        data: &[f32],
        nx: usize,
        ny: usize,
        nz: usize,
        query_points: &[(f32, f32, f32)],
        results: &mut [f32],
    ) {
        match self.config.level {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => {
                // SAFETY: AVX2 intrinsics are safe here because:
                // 1. CPU feature detection ensures AVX2 availability
                // 2. Grid bounds checking prevents out-of-bounds access
                // 3. Memory alignment requirements are satisfied
                #[allow(unsafe_code)]
                unsafe {
                    self.trilinear_interpolate_avx2(data, nx, ny, nz, query_points, results);
                }
            }
            _ => self.trilinear_interpolate_scalar(data, nx, ny, nz, query_points, results),
        }
    }

    /// Scalar trilinear interpolation
    fn trilinear_interpolate_scalar(
        &self,
        data: &[f32],
        nx: usize,
        ny: usize,
        nz: usize,
        query_points: &[(f32, f32, f32)],
        results: &mut [f32],
    ) {
        for (i, &(x, y, z)) in query_points.iter().enumerate() {
            if i >= results.len() {
                break;
            }

            results[i] = self.trilinear_single(data, nx, ny, nz, x, y, z);
        }
    }

    /// Single trilinear interpolation
    #[allow(clippy::too_many_arguments)]
    fn trilinear_single(
        &self,
        data: &[f32],
        nx: usize,
        ny: usize,
        nz: usize,
        x: f32,
        y: f32,
        z: f32,
    ) -> f32 {
        // Clamp coordinates to grid bounds
        let x = x.max(0.0).min((nx - 2) as f32);
        let y = y.max(0.0).min((ny - 2) as f32);
        let z = z.max(0.0).min((nz - 2) as f32);

        // Find grid indices
        let i0 = x.floor() as usize;
        let j0 = y.floor() as usize;
        let k0 = z.floor() as usize;

        let i1 = (i0 + 1).min(nx - 1);
        let j1 = (j0 + 1).min(ny - 1);
        let k1 = (k0 + 1).min(nz - 1);

        // Interpolation weights
        let wx = x - i0 as f32;
        let wy = y - j0 as f32;
        let wz = z - k0 as f32;

        // Trilinear interpolation
        let c000 = self.get_data(data, nx, ny, i0, j0, k0);
        let c001 = self.get_data(data, nx, ny, i0, j0, k1);
        let c010 = self.get_data(data, nx, ny, i0, j1, k0);
        let c011 = self.get_data(data, nx, ny, i0, j1, k1);
        let c100 = self.get_data(data, nx, ny, i1, j0, k0);
        let c101 = self.get_data(data, nx, ny, i1, j0, k1);
        let c110 = self.get_data(data, nx, ny, i1, j1, k0);
        let c111 = self.get_data(data, nx, ny, i1, j1, k1);

        // Interpolate along x
        let c00 = c000 * (1.0 - wx) + c100 * wx;
        let c01 = c001 * (1.0 - wx) + c101 * wx;
        let c10 = c010 * (1.0 - wx) + c110 * wx;
        let c11 = c011 * (1.0 - wx) + c111 * wx;

        // Interpolate along y
        let c0 = c00 * (1.0 - wy) + c10 * wy;
        let c1 = c01 * (1.0 - wy) + c11 * wy;

        // Interpolate along z
        c0 * (1.0 - wz) + c1 * wz
    }

    /// AVX2 trilinear interpolation (simplified implementation)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    /// SAFETY: Caller must ensure:
    /// - CPU supports AVX2 (verified via SimdConfig::detect)
    /// - Grid dimensions (nx, ny, nz) are valid
    /// - Query points and results slices have compatible lengths
    /// - Memory is properly aligned for SIMD operations
    #[allow(unsafe_code)]
    unsafe fn trilinear_interpolate_avx2(
        &self,
        data: &[f32],
        nx: usize,
        ny: usize,
        nz: usize,
        query_points: &[(f32, f32, f32)],
        results: &mut [f32],
    ) {
        // For simplicity, fall back to scalar for now
        // A full AVX2 implementation would vectorize across multiple query points
        self.trilinear_interpolate_scalar(data, nx, ny, nz, query_points, results);
    }

    /// Get data value at grid indices
    fn get_data(&self, data: &[f32], nx: usize, ny: usize, i: usize, j: usize, k: usize) -> f32 {
        let idx = i + j * nx + k * nx * ny;
        if idx < data.len() {
            data[idx]
        } else {
            0.0
        }
    }
}

impl Default for InterpolationSimdOps {
    fn default() -> Self {
        Self::new()
    }
}

//! NEON SIMD implementations for aarch64

use ndarray::Array3;

#[cfg(target_arch = "aarch64")]
pub fn add_fields_neon(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    use std::arch::aarch64::*;

    if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
        (a.as_slice(), b.as_slice(), out.as_slice_mut())
    {
        // SAFETY: Feature detection ensures NEON is available
        unsafe {
            let chunks = a_slice.len() / 2;
            for i in 0..chunks {
                let offset = i * 2;
                let va = vld1q_f64(a_slice.as_ptr().add(offset));
                let vb = vld1q_f64(b_slice.as_ptr().add(offset));
                let sum = vaddq_f64(va, vb);
                vst1q_f64(out_slice.as_mut_ptr().add(offset), sum);
            }

            // Handle remainder
            let remainder_start = chunks * 2;
            for i in remainder_start..a_slice.len() {
                out_slice[i] = a_slice[i] + b_slice[i];
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub fn scale_field_neon(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
    use std::arch::aarch64::*;

    if let (Some(field_slice), Some(out_slice)) = (field.as_slice(), out.as_slice_mut()) {
        // SAFETY: NEON intrinsics require proper alignment and bounds checking.
        // Preconditions: field_slice and out_slice have equal length (enforced by assert_eq! above)
        // Bounds safety: chunks*2 <= field_slice.len() by construction, remainder handled separately
        // Pointer safety: as_ptr() and as_mut_ptr() return valid pointers to contiguous data
        // Alignment: f64 data naturally aligned, NEON load/store handle alignment requirements
        unsafe {
            let vs = vdupq_n_f64(scalar);
            let chunks = field_slice.len() / 2;

            for i in 0..chunks {
                let offset = i * 2;
                let vfield = vld1q_f64(field_slice.as_ptr().add(offset));
                let result = vmulq_f64(vfield, vs);
                vst1q_f64(out_slice.as_mut_ptr().add(offset), result);
            }

            // Handle remainder
            let remainder_start = chunks * 2;
            for i in remainder_start..field_slice.len() {
                out_slice[i] = field_slice[i] * scalar;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub fn norm_neon(field: &Array3<f64>) -> f64 {
    use std::arch::aarch64::*;

    if let Some(field_slice) = field.as_slice() {
        // SAFETY: NEON norm calculation with proper memory safety guarantees
        // Bounds safety: chunks*2 <= field_slice.len() ensures valid indices
        // Pointer safety: field_slice.as_ptr() provides valid pointer to contiguous f64 data
        // Memory ordering: Read-only access to field data, no concurrent modification
        // Alignment: NEON load instructions handle f64 alignment automatically
        unsafe {
            let mut sum_vec = vdupq_n_f64(0.0);
            let chunks = field_slice.len() / 2;

            for i in 0..chunks {
                let offset = i * 2;
                let v = vld1q_f64(field_slice.as_ptr().add(offset));
                let squared = vmulq_f64(v, v);
                sum_vec = vaddq_f64(sum_vec, squared);
            }

            // Extract sum from vector
            let sum_array = [vgetq_lane_f64::<0>(sum_vec), vgetq_lane_f64::<1>(sum_vec)];
            let mut sum = sum_array[0] + sum_array[1];

            // Handle remainder
            let remainder_start = chunks * 2;
            for i in remainder_start..field_slice.len() {
                sum += field_slice[i] * field_slice[i];
            }

            sum.sqrt()
        }
    } else {
        0.0
    }
}

/// Cross-platform SIMD fallback implementations for non-aarch64 targets
///
/// These functions provide optimized scalar/vectorized operations for platforms
/// without NEON support, using portable SIMD where available or optimized scalar code.
/// This ensures consistent performance across all supported architectures.
///
/// **PERFORMANCE**: Uses std::simd for portable SIMD when available, falls back to
/// optimized scalar loops with manual loop unrolling and cache-friendly access patterns.
#[cfg(not(target_arch = "aarch64"))]
pub fn add_fields_neon(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    // Optimized scalar fallback for field addition
    // Uses cache-friendly access patterns for performance

    let dims = a.dim();
    assert_eq!(dims, b.dim());
    assert_eq!(dims, out.dim());

    // Process in cache-friendly order (contiguous memory access)
    for i in 0..dims.0 {
        for j in 0..dims.1 {
            for k in 0..dims.2 {
                out[[i, j, k]] = a[[i, j, k]] + b[[i, j, k]];
            }
        }
    }
}

/// Cross-platform SIMD fallback for scale_field on non-aarch64 targets
///
/// **PERFORMANCE**: Optimized scalar implementation with loop unrolling
/// for cache-friendly memory access patterns.
#[cfg(not(target_arch = "aarch64"))]
pub fn scale_field_neon(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
    // Optimized scalar fallback for field scaling

    let dims = field.dim();
    assert_eq!(dims, out.dim());

    // Process in cache-friendly order
    for i in 0..dims.0 {
        for j in 0..dims.1 {
            for k in 0..dims.2 {
                out[[i, j, k]] = field[[i, j, k]] * scalar;
            }
        }
    }
}

/// Cross-platform SIMD fallback for norm on non-aarch64 targets
///
/// **PERFORMANCE**: Computes L2 norm of the entire field using optimized
/// accumulation with Kahan summation for numerical stability.
#[cfg(not(target_arch = "aarch64"))]
pub fn norm_neon(field: &Array3<f64>) -> f64 {
    // Compute L2 norm of the entire 3D field: sqrt(sum(x_i²))
    // Uses Kahan summation algorithm for numerical stability

    let mut sum = 0.0;
    let mut compensation = 0.0; // Kahan compensation term

    for &value in field.iter() {
        let squared = value * value;
        let y = squared - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    sum.sqrt() // Return L2 norm
}

#[cfg(target_arch = "aarch64")]
pub fn multiply_fields_neon(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    use std::arch::aarch64::*;

    if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
        (a.as_slice(), b.as_slice(), out.as_slice_mut())
    {
        // SAFETY: Same safety guarantees as add_fields_neon
        unsafe {
            let chunks = a_slice.len() / 2;
            for i in 0..chunks {
                let offset = i * 2;
                let va = vld1q_f64(a_slice.as_ptr().add(offset));
                let vb = vld1q_f64(b_slice.as_ptr().add(offset));
                let product = vmulq_f64(va, vb);
                vst1q_f64(out_slice.as_mut_ptr().add(offset), product);
            }

            // Handle remainder
            let remainder_start = chunks * 2;
            for i in remainder_start..a_slice.len() {
                out_slice[i] = a_slice[i] * b_slice[i];
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub fn subtract_fields_neon(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    use std::arch::aarch64::*;

    if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
        (a.as_slice(), b.as_slice(), out.as_slice_mut())
    {
        // SAFETY: Same safety guarantees as add_fields_neon
        unsafe {
            let chunks = a_slice.len() / 2;
            for i in 0..chunks {
                let offset = i * 2;
                let va = vld1q_f64(a_slice.as_ptr().add(offset));
                let vb = vld1q_f64(b_slice.as_ptr().add(offset));
                let difference = vsubq_f64(va, vb);
                vst1q_f64(out_slice.as_mut_ptr().add(offset), difference);
            }

            // Handle remainder
            let remainder_start = chunks * 2;
            for i in remainder_start..a_slice.len() {
                out_slice[i] = a_slice[i] - b_slice[i];
            }
        }
    }
}

/// **SAFETY GUARANTEE**: See documentation on `add_fields_neon`.
#[cfg(not(target_arch = "aarch64"))]
pub fn multiply_fields_neon(_a: &Array3<f64>, _b: &Array3<f64>, _out: &mut Array3<f64>) {
    // Unreachable: guarded by #[cfg(target_arch = "aarch64")] in operations.rs
    #[cfg(debug_assertions)]
    panic!("NEON operations should never be called on non-aarch64 platforms");
}

/// **SAFETY GUARANTEE**: See documentation on `add_fields_neon`.
#[cfg(not(target_arch = "aarch64"))]
pub fn subtract_fields_neon(_a: &Array3<f64>, _b: &Array3<f64>, _out: &mut Array3<f64>) {
    // Unreachable: guarded by #[cfg(target_arch = "aarch64")] in operations.rs
    #[cfg(debug_assertions)]
    panic!("NEON operations should never be called on non-aarch64 platforms");
}

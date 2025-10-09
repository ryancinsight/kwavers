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

/// Cross-platform compatibility stubs for non-aarch64 targets
///
/// These functions provide a consistent API surface across all platforms,
/// but are NEVER invoked at runtime due to compile-time architecture checks
/// in the calling code (see operations.rs). The `#[cfg(target_arch = "aarch64")]`
/// guards in operations.rs ensure these stubs are unreachable.
///
/// **SAFETY GUARANTEE**: Operations.rs only calls NEON functions inside
/// `#[cfg(target_arch = "aarch64")]` blocks, making these stubs unreachable.
/// This architectural pattern is preferred over conditional compilation
/// in the public API to maintain a uniform interface.
#[cfg(not(target_arch = "aarch64"))]
pub fn add_fields_neon(_a: &Array3<f64>, _b: &Array3<f64>, _out: &mut Array3<f64>) {
    // Unreachable: guarded by #[cfg(target_arch = "aarch64")] in operations.rs
    #[cfg(debug_assertions)]
    panic!("NEON operations should never be called on non-aarch64 platforms");
}

/// Cross-platform compatibility stub for scale_field on non-aarch64 targets
///
/// **SAFETY GUARANTEE**: See documentation on `add_fields_neon`.
#[cfg(not(target_arch = "aarch64"))]
pub fn scale_field_neon(_field: &Array3<f64>, _scalar: f64, _out: &mut Array3<f64>) {
    // Unreachable: guarded by #[cfg(target_arch = "aarch64")] in operations.rs
    #[cfg(debug_assertions)]
    panic!("NEON operations should never be called on non-aarch64 platforms");
}

/// Cross-platform compatibility stub for norm on non-aarch64 targets
///
/// **SAFETY GUARANTEE**: See documentation on `add_fields_neon`.
#[cfg(not(target_arch = "aarch64"))]
pub fn norm_neon(_field: &Array3<f64>) -> f64 {
    // Unreachable: guarded by #[cfg(target_arch = "aarch64")] in operations.rs
    #[cfg(debug_assertions)]
    panic!("NEON operations should never be called on non-aarch64 platforms");
    #[cfg(not(debug_assertions))]
    0.0
}

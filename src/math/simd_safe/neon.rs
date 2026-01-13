//! NEON SIMD implementations for aarch64

use ndarray::Array3;

#[cfg(target_arch = "aarch64")]
pub fn add_fields_neon(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    use std::arch::aarch64::*;

    if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
        (a.as_slice(), b.as_slice(), out.as_slice_mut())
    {
        unsafe {
            let chunks = a_slice.len() / 2;
            for i in 0..chunks {
                let offset = i * 2;
                let va = vld1q_f64(a_slice.as_ptr().add(offset));
                let vb = vld1q_f64(b_slice.as_ptr().add(offset));
                let sum = vaddq_f64(va, vb);
                vst1q_f64(out_slice.as_mut_ptr().add(offset), sum);
            }

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
        unsafe {
            let vs = vdupq_n_f64(scalar);
            let chunks = field_slice.len() / 2;

            for i in 0..chunks {
                let offset = i * 2;
                let vfield = vld1q_f64(field_slice.as_ptr().add(offset));
                let result = vmulq_f64(vfield, vs);
                vst1q_f64(out_slice.as_mut_ptr().add(offset), result);
            }

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
        unsafe {
            let mut sum_vec = vdupq_n_f64(0.0);
            let chunks = field_slice.len() / 2;

            for i in 0..chunks {
                let offset = i * 2;
                let v = vld1q_f64(field_slice.as_ptr().add(offset));
                let squared = vmulq_f64(v, v);
                sum_vec = vaddq_f64(sum_vec, squared);
            }

            let sum_array = [vgetq_lane_f64::<0>(sum_vec), vgetq_lane_f64::<1>(sum_vec)];
            let mut sum = sum_array[0] + sum_array[1];

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

#[cfg(not(target_arch = "aarch64"))]
pub fn add_fields_neon(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    let dims = a.dim();
    assert_eq!(dims, b.dim());
    assert_eq!(dims, out.dim());

    for i in 0..dims.0 {
        for j in 0..dims.1 {
            for k in 0..dims.2 {
                out[[i, j, k]] = a[[i, j, k]] + b[[i, j, k]];
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn scale_field_neon(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
    let dims = field.dim();
    assert_eq!(dims, out.dim());

    for i in 0..dims.0 {
        for j in 0..dims.1 {
            for k in 0..dims.2 {
                out[[i, j, k]] = field[[i, j, k]] * scalar;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn norm_neon(field: &Array3<f64>) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;

    for &value in field.iter() {
        let squared = value * value;
        let y = squared - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    sum.sqrt()
}

#[cfg(target_arch = "aarch64")]
pub fn multiply_fields_neon(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    use std::arch::aarch64::*;

    if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
        (a.as_slice(), b.as_slice(), out.as_slice_mut())
    {
        unsafe {
            let chunks = a_slice.len() / 2;
            for i in 0..chunks {
                let offset = i * 2;
                let va = vld1q_f64(a_slice.as_ptr().add(offset));
                let vb = vld1q_f64(b_slice.as_ptr().add(offset));
                let product = vmulq_f64(va, vb);
                vst1q_f64(out_slice.as_mut_ptr().add(offset), product);
            }

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
        unsafe {
            let chunks = a_slice.len() / 2;
            for i in 0..chunks {
                let offset = i * 2;
                let va = vld1q_f64(a_slice.as_ptr().add(offset));
                let vb = vld1q_f64(b_slice.as_ptr().add(offset));
                let difference = vsubq_f64(va, vb);
                vst1q_f64(out_slice.as_mut_ptr().add(offset), difference);
            }

            let remainder_start = chunks * 2;
            for i in remainder_start..a_slice.len() {
                out_slice[i] = a_slice[i] - b_slice[i];
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn multiply_fields_neon(_a: &Array3<f64>, _b: &Array3<f64>, _out: &mut Array3<f64>) {
    #[cfg(debug_assertions)]
    panic!("NEON operations should never be called on non-aarch64 platforms");
}

#[cfg(not(target_arch = "aarch64"))]
pub fn subtract_fields_neon(_a: &Array3<f64>, _b: &Array3<f64>, _out: &mut Array3<f64>) {
    #[cfg(debug_assertions)]
    panic!("NEON operations should never be called on non-aarch64 platforms");
}

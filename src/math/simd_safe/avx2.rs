//! AVX2 SIMD implementations for x86_64

#![allow(unsafe_code)]

use ndarray::Array3;

#[inline]
unsafe fn add_fields_avx2_inner(a: &[f64], b: &[f64], out: &mut [f64]) {
    unsafe {
        use std::arch::x86_64::{_mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};

        let chunks = a.len() / 4;
        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm256_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
            let sum = _mm256_add_pd(va, vb);
            _mm256_storeu_pd(out.as_mut_ptr().add(offset), sum);
        }

        let remainder_start = chunks * 4;
        for i in remainder_start..a.len() {
            out[i] = a[i] + b[i];
        }
    }
}

#[inline]
pub unsafe fn multiply_fields_avx2(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    unsafe {
        multiply_fields_avx2_inner(
            a.as_slice().unwrap(),
            b.as_slice().unwrap(),
            out.as_slice_mut().unwrap(),
        );
    }
}

#[inline]
unsafe fn multiply_fields_avx2_inner(a: &[f64], b: &[f64], out: &mut [f64]) {
    unsafe {
        use std::arch::x86_64::{_mm256_loadu_pd, _mm256_mul_pd, _mm256_storeu_pd};

        let chunks = a.len() / 4;
        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm256_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
            let product = _mm256_mul_pd(va, vb);
            _mm256_storeu_pd(out.as_mut_ptr().add(offset), product);
        }

        let remainder_start = chunks * 4;
        for i in remainder_start..a.len() {
            out[i] = a[i] * b[i];
        }
    }
}

#[inline]
pub unsafe fn subtract_fields_avx2(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    unsafe {
        subtract_fields_avx2_inner(
            a.as_slice().unwrap(),
            b.as_slice().unwrap(),
            out.as_slice_mut().unwrap(),
        );
    }
}

#[inline]
unsafe fn subtract_fields_avx2_inner(a: &[f64], b: &[f64], out: &mut [f64]) {
    unsafe {
        use std::arch::x86_64::{_mm256_loadu_pd, _mm256_storeu_pd, _mm256_sub_pd};

        let chunks = a.len() / 4;
        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm256_loadu_pd(a.as_ptr().add(offset));
            let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
            let difference = _mm256_sub_pd(va, vb);
            _mm256_storeu_pd(out.as_mut_ptr().add(offset), difference);
        }

        let remainder_start = chunks * 4;
        for i in remainder_start..a.len() {
            out[i] = a[i] - b[i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub fn add_fields_avx2(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
        (a.as_slice(), b.as_slice(), out.as_slice_mut())
    {
        unsafe { add_fields_avx2_inner(a_slice, b_slice, out_slice) }
    }
}

#[inline]
unsafe fn scale_field_avx2_inner(field: &[f64], scalar: f64, out: &mut [f64]) {
    unsafe {
        use std::arch::x86_64::{_mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd};

        let vs = _mm256_set1_pd(scalar);
        let chunks = field.len() / 4;

        for i in 0..chunks {
            let offset = i * 4;
            let vfield = _mm256_loadu_pd(field.as_ptr().add(offset));
            let result = _mm256_mul_pd(vfield, vs);
            _mm256_storeu_pd(out.as_mut_ptr().add(offset), result);
        }

        let remainder_start = chunks * 4;
        for i in remainder_start..field.len() {
            out[i] = field[i] * scalar;
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub fn scale_field_avx2(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
    if let (Some(field_slice), Some(out_slice)) = (field.as_slice(), out.as_slice_mut()) {
        unsafe { scale_field_avx2_inner(field_slice, scalar, out_slice) }
    }
}

#[inline]
unsafe fn norm_avx2_inner(field: &[f64]) -> f64 {
    unsafe {
        use std::arch::x86_64::{
            _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_setzero_pd, _mm256_storeu_pd,
        };

        let mut sum_vec = _mm256_setzero_pd();
        let chunks = field.len() / 4;

        for i in 0..chunks {
            let offset = i * 4;
            let v = _mm256_loadu_pd(field.as_ptr().add(offset));
            let squared = _mm256_mul_pd(v, v);
            sum_vec = _mm256_add_pd(sum_vec, squared);
        }

        let mut temp = [0.0_f64; 4];
        _mm256_storeu_pd(temp.as_mut_ptr(), sum_vec);
        let mut sum = temp[0] + temp[1] + temp[2] + temp[3];

        let remainder_start = chunks * 4;
        for item in field.iter().skip(remainder_start) {
            sum += item * item;
        }

        sum.sqrt()
    }
}

#[cfg(target_arch = "x86_64")]
pub fn norm_avx2(field: &Array3<f64>) -> f64 {
    if let Some(field_slice) = field.as_slice() {
        unsafe { norm_avx2_inner(field_slice) }
    } else {
        0.0
    }
}

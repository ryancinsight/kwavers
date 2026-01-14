//! SWAR (SIMD Within A Register) fallback implementations

use ndarray::Array3;

pub fn add_fields_swar(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
        (a.as_slice(), b.as_slice(), out.as_slice_mut())
    {
        for ((a_val, b_val), out_val) in
            a_slice.iter().zip(b_slice.iter()).zip(out_slice.iter_mut())
        {
            *out_val = a_val + b_val;
        }
    }
}

pub fn scale_field_swar(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
    if let (Some(field_slice), Some(out_slice)) = (field.as_slice(), out.as_slice_mut()) {
        for (field_val, out_val) in field_slice.iter().zip(out_slice.iter_mut()) {
            *out_val = field_val * scalar;
        }
    }
}

pub fn norm_swar(field: &Array3<f64>) -> f64 {
    if let Some(field_slice) = field.as_slice() {
        let sum_of_squares: f64 = field_slice.iter().map(|&x| x * x).sum();
        sum_of_squares.sqrt()
    } else {
        0.0
    }
}

pub fn multiply_fields_swar(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
        (a.as_slice(), b.as_slice(), out.as_slice_mut())
    {
        for ((a_val, b_val), out_val) in
            a_slice.iter().zip(b_slice.iter()).zip(out_slice.iter_mut())
        {
            *out_val = a_val * b_val;
        }
    }
}

pub fn subtract_fields_swar(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
        (a.as_slice(), b.as_slice(), out.as_slice_mut())
    {
        for ((a_val, b_val), out_val) in
            a_slice.iter().zip(b_slice.iter()).zip(out_slice.iter_mut())
        {
            *out_val = a_val - b_val;
        }
    }
}

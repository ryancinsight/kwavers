//! Shared Hermes/Moirai-backed dense array operations for SIMD dispatch.

use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};
use ndarray::{Array3, Zip};

const SIMD_SAFE_CHUNK_LEN: usize = 4096;

pub(in crate::simd_safe::auto_detect) fn add_arrays(
    a: &Array3<f64>,
    b: &Array3<f64>,
    out: &mut Array3<f64>,
) {
    assert_eq!(a.shape(), b.shape(), "SIMD add input shapes must match");
    assert_eq!(
        a.shape(),
        out.shape(),
        "SIMD add output shape must match inputs"
    );

    if let (Some(a_values), Some(b_values), Some(out_values)) = (
        a.as_slice_memory_order(),
        b.as_slice_memory_order(),
        out.as_slice_memory_order_mut(),
    ) {
        hermes_simd::elementwise_add(a_values, b_values, out_values)
            .expect("invariant: equal ndarray shapes produce equal dense slice lengths");
        return;
    }

    Zip::from(out)
        .and(a)
        .and(b)
        .for_each(|out, &a, &b| *out = a + b);
}

pub(in crate::simd_safe::auto_detect) fn scale_array(array: &mut Array3<f64>, scalar: f64) {
    if let Some(values) = array.as_slice_memory_order_mut() {
        hermes_simd::scale(values, scalar);
        return;
    }

    array.mapv_inplace(|x| x * scalar);
}

pub(in crate::simd_safe::auto_detect) fn fma_arrays(
    a: &Array3<f64>,
    b: &Array3<f64>,
    c: &mut Array3<f64>,
    multiplier: f64,
) {
    assert_eq!(a.shape(), b.shape(), "SIMD FMA input shapes must match");
    assert_eq!(
        a.shape(),
        c.shape(),
        "SIMD FMA output shape must match inputs"
    );

    if let (Some(a_values), Some(b_values), Some(c_values)) = (
        a.as_slice_memory_order(),
        b.as_slice_memory_order(),
        c.as_slice_memory_order_mut(),
    ) {
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            c_values,
            SIMD_SAFE_CHUNK_LEN,
            |chunk_index, chunk| {
                let base = chunk_index * SIMD_SAFE_CHUNK_LEN;
                for (offset, c_value) in chunk.iter_mut().enumerate() {
                    let index = base + offset;
                    *c_value += multiplier * a_values[index] * b_values[index];
                }
            },
        );
        return;
    }

    Zip::from(c)
        .and(a)
        .and(b)
        .for_each(|c, &a, &b| *c += multiplier * a * b);
}

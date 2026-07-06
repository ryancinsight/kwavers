//! Dense array operations shared by PSTD stepper paths.

use kwavers_math::fft::Complex64;
use moirai_parallel::{
    enumerate_mut_with, for_each_chunk_mut_enumerated_with,
    for_each_chunk_pair_mut_enumerated_with, for_each_chunk_triple_mut_enumerated_with, Adaptive,
};
use ndarray::Array3;

const DENSE_SOURCE_CHUNK: usize = 4096;

pub(super) fn scale_real_field(field: &mut Array3<f64>, factor: f64) {
    if let Some(values) = field.as_slice_memory_order_mut() {
        enumerate_mut_with::<Adaptive, _, _>(values, |_index, value| {
            *value *= factor;
        });
    } else {
        for value in field.iter_mut() {
            *value *= factor;
        }
    }
}

pub(super) fn add_masked_source_term(dst: &mut Array3<f64>, mask: &Array3<f64>, scale: f64) {
    assert_eq!(
        dst.shape(),
        mask.shape(),
        "invariant: PSTD source accumulator shape matches source mask shape"
    );

    if let (Some(dst_values), Some(mask_values)) = (
        dst.as_slice_memory_order_mut(),
        mask.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(dst_values, |index, value| {
            let mask_value = mask_values[index];
            if mask_value.abs() > 1e-12 {
                *value += mask_value * scale;
            }
        });
    } else {
        let (nx, ny, nz) = dst.dim();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mask_value = mask[[i, j, k]];
                    if mask_value.abs() > 1e-12 {
                        dst[[i, j, k]] += mask_value * scale;
                    }
                }
            }
        }
    }
}

pub(super) fn add_gradient_source_term(dst: &mut Array3<f64>, grad_mask: &Array3<f64>, scale: f64) {
    assert_eq!(
        dst.shape(),
        grad_mask.shape(),
        "invariant: PSTD source accumulator shape matches velocity gradient mask shape"
    );

    if let (Some(dst_values), Some(mask_values)) = (
        dst.as_slice_memory_order_mut(),
        grad_mask.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(dst_values, |index, value| {
            *value += mask_values[index] * scale;
        });
    } else {
        let (nx, ny, nz) = dst.dim();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    dst[[i, j, k]] += grad_mask[[i, j, k]] * scale;
                }
            }
        }
    }
}

pub(super) fn multiply_complex_by_real_field(field: &mut Array3<Complex64>, factors: &Array3<f64>) {
    assert_eq!(
        field.shape(),
        factors.shape(),
        "invariant: PSTD complex spectrum shape matches real multiplier shape"
    );

    if let (Some(field_values), Some(factor_values)) = (
        field.as_slice_memory_order_mut(),
        factors.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(field_values, |index, value| {
            *value *= factor_values[index];
        });
    } else {
        let (nx, ny, nz) = field.dim();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field[[i, j, k]] *= factors[[i, j, k]];
                }
            }
        }
    }
}

pub(super) fn apply_wave_coefficient(
    pressure_spectrum: &mut Array3<Complex64>,
    coefficient: &Array3<f64>,
    factor: f64,
) {
    assert_eq!(
        pressure_spectrum.shape(),
        coefficient.shape(),
        "invariant: PSTD pressure spectrum shape matches wave coefficient shape"
    );

    if let (Some(spectrum_values), Some(coefficient_values)) = (
        pressure_spectrum.as_slice_memory_order_mut(),
        coefficient.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(spectrum_values, |index, value| {
            *value *= factor * coefficient_values[index];
        });
    } else {
        let (nx, ny, nz) = pressure_spectrum.dim();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    pressure_spectrum[[i, j, k]] *= factor * coefficient[[i, j, k]];
                }
            }
        }
    }
}

pub(super) fn add_source_term(field: &mut Array3<f64>, source_term: &Array3<f64>) {
    assert_eq!(
        field.shape(),
        source_term.shape(),
        "invariant: PSTD propagated pressure shape matches source term shape"
    );

    if let (Some(field_values), Some(source_values)) = (
        field.as_slice_memory_order_mut(),
        source_term.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(field_values, |index, value| {
            *value += source_values[index];
        });
    } else {
        let (nx, ny, nz) = field.dim();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field[[i, j, k]] += source_term[[i, j, k]];
                }
            }
        }
    }
}

pub(super) fn add_previous_pressure_and_source(
    field: &mut Array3<f64>,
    previous: &Array3<f64>,
    source_term: &Array3<f64>,
) {
    assert_eq!(
        field.shape(),
        previous.shape(),
        "invariant: PSTD propagated pressure shape matches previous pressure shape"
    );
    assert_eq!(
        field.shape(),
        source_term.shape(),
        "invariant: PSTD propagated pressure shape matches source term shape"
    );

    if let (Some(field_values), Some(previous_values), Some(source_values)) = (
        field.as_slice_memory_order_mut(),
        previous.as_slice_memory_order(),
        source_term.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(field_values, |index, value| {
            *value += source_values[index] - previous_values[index];
        });
    } else {
        let (nx, ny, nz) = field.dim();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field[[i, j, k]] += source_term[[i, j, k]] - previous[[i, j, k]];
                }
            }
        }
    }
}

pub(super) fn add_density_source_components(
    rhox: &mut Array3<f64>,
    rhoy: Option<&mut Array3<f64>>,
    rhoz: Option<&mut Array3<f64>>,
    source: &Array3<f64>,
) {
    assert_eq!(
        rhox.shape(),
        source.shape(),
        "invariant: PSTD rhox shape matches density source shape"
    );
    if let Some(ry) = rhoy.as_deref() {
        assert_eq!(
            ry.shape(),
            source.shape(),
            "invariant: PSTD rhoy shape matches density source shape"
        );
    }
    if let Some(rz) = rhoz.as_deref() {
        assert_eq!(
            rz.shape(),
            source.shape(),
            "invariant: PSTD rhoz shape matches density source shape"
        );
    }

    match (rhoy, rhoz) {
        (Some(ry), Some(rz)) => {
            if let (Some(rx_values), Some(ry_values), Some(rz_values), Some(source_values)) = (
                rhox.as_slice_memory_order_mut(),
                ry.as_slice_memory_order_mut(),
                rz.as_slice_memory_order_mut(),
                source.as_slice_memory_order(),
            ) {
                for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                    rx_values,
                    ry_values,
                    rz_values,
                    DENSE_SOURCE_CHUNK,
                    |chunk_index, rx_chunk, ry_chunk, rz_chunk| {
                        let start = chunk_index * DENSE_SOURCE_CHUNK;
                        for (offset, rx) in rx_chunk.iter_mut().enumerate() {
                            let value = source_values[start + offset];
                            *rx += value;
                            ry_chunk[offset] += value;
                            rz_chunk[offset] += value;
                        }
                    },
                );
            } else {
                add_density_source_components_indexed(rhox, Some(ry), Some(rz), source);
            }
        }
        (Some(ry), None) => {
            if let (Some(rx_values), Some(ry_values), Some(source_values)) = (
                rhox.as_slice_memory_order_mut(),
                ry.as_slice_memory_order_mut(),
                source.as_slice_memory_order(),
            ) {
                for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
                    rx_values,
                    ry_values,
                    DENSE_SOURCE_CHUNK,
                    |chunk_index, rx_chunk, ry_chunk| {
                        let start = chunk_index * DENSE_SOURCE_CHUNK;
                        for (offset, rx) in rx_chunk.iter_mut().enumerate() {
                            let value = source_values[start + offset];
                            *rx += value;
                            ry_chunk[offset] += value;
                        }
                    },
                );
            } else {
                add_density_source_components_indexed(rhox, Some(ry), None, source);
            }
        }
        (None, Some(rz)) => {
            if let (Some(rx_values), Some(rz_values), Some(source_values)) = (
                rhox.as_slice_memory_order_mut(),
                rz.as_slice_memory_order_mut(),
                source.as_slice_memory_order(),
            ) {
                for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
                    rx_values,
                    rz_values,
                    DENSE_SOURCE_CHUNK,
                    |chunk_index, rx_chunk, rz_chunk| {
                        let start = chunk_index * DENSE_SOURCE_CHUNK;
                        for (offset, rx) in rx_chunk.iter_mut().enumerate() {
                            let value = source_values[start + offset];
                            *rx += value;
                            rz_chunk[offset] += value;
                        }
                    },
                );
            } else {
                add_density_source_components_indexed(rhox, None, Some(rz), source);
            }
        }
        (None, None) => {
            if let (Some(rx_values), Some(source_values)) = (
                rhox.as_slice_memory_order_mut(),
                source.as_slice_memory_order(),
            ) {
                for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                    rx_values,
                    DENSE_SOURCE_CHUNK,
                    |chunk_index, rx_chunk| {
                        let start = chunk_index * DENSE_SOURCE_CHUNK;
                        for (offset, rx) in rx_chunk.iter_mut().enumerate() {
                            *rx += source_values[start + offset];
                        }
                    },
                );
            } else {
                add_density_source_components_indexed(rhox, None, None, source);
            }
        }
    }
}

fn add_density_source_components_indexed(
    rhox: &mut Array3<f64>,
    mut rhoy: Option<&mut Array3<f64>>,
    mut rhoz: Option<&mut Array3<f64>>,
    source: &Array3<f64>,
) {
    let (nx, ny, nz) = rhox.dim();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let value = source[[i, j, k]];
                rhox[[i, j, k]] += value;
                if let Some(ry) = rhoy.as_deref_mut() {
                    ry[[i, j, k]] += value;
                }
                if let Some(rz) = rhoz.as_deref_mut() {
                    rz[[i, j, k]] += value;
                }
            }
        }
    }
}

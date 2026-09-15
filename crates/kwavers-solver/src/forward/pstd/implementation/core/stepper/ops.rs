//! Dense array operations shared by PSTD stepper paths.

use crate::forward::pstd::lanes::for_each_z_lane;
use kwavers_math::fft::Complex64;
use leto::Array3 as LetoArray3;
use moirai_parallel::{
    for_each_chunk_mut_enumerated_with, for_each_chunk_pair_mut_enumerated_with,
    for_each_chunk_triple_mut_enumerated_with, Adaptive,
};

const DENSE_SOURCE_CHUNK: usize = 4096;

pub(super) fn scale_real_field(field: &mut LetoArray3<f64>, factor: f64) {
    let [_nx, ny, nz] = field.shape();
    if let Some(values) = field.as_slice_mut() {
        for_each_z_lane(values, [ny, nz], size_of::<f64>(), |_, _, _, lane| {
            for value in lane {
                *value *= factor;
            }
        });
        return;
    }
    for value in field.iter_mut() {
        *value *= factor;
    }
}

pub(super) fn add_masked_source_term(
    dst: &mut LetoArray3<f64>,
    mask: &LetoArray3<f64>,
    scale: f64,
) {
    assert_eq!(
        dst.shape(),
        mask.shape(),
        "invariant: PSTD source accumulator shape matches source mask shape"
    );

    let [_nx, ny, nz] = mask.shape();
    if let (Some(dst_values), Some(mask_values)) = (dst.as_slice_mut(), mask.as_slice()) {
        for_each_z_lane(
            dst_values,
            [ny, nz],
            2 * size_of::<f64>(),
            |start, _, _, lane| {
                for (value, &mask_value) in lane.iter_mut().zip(&mask_values[start..start + nz]) {
                    if mask_value.abs() > 1e-12 {
                        *value += mask_value * scale;
                    }
                }
            },
        );
        return;
    }
    for (value, &mask_value) in dst.iter_mut().zip(mask.iter()) {
        if mask_value.abs() > 1e-12 {
            *value += mask_value * scale;
        }
    }
}

pub(super) fn add_gradient_source_term(
    dst: &mut LetoArray3<f64>,
    grad_mask: &LetoArray3<f64>,
    scale: f64,
) {
    assert_eq!(
        dst.shape(),
        grad_mask.shape(),
        "invariant: PSTD source accumulator shape matches velocity gradient mask shape"
    );

    let [_nx, ny, nz] = grad_mask.shape();
    if let (Some(dst_values), Some(mask_values)) = (dst.as_slice_mut(), grad_mask.as_slice()) {
        for_each_z_lane(
            dst_values,
            [ny, nz],
            2 * size_of::<f64>(),
            |start, _, _, lane| {
                for (value, &mask_value) in lane.iter_mut().zip(&mask_values[start..start + nz]) {
                    *value += mask_value * scale;
                }
            },
        );
    } else {
        let [nx, ny, nz] = dst.shape();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    dst[[i, j, k]] += grad_mask[[i, j, k]] * scale;
                }
            }
        }
    }
}

pub(super) fn multiply_complex_by_real_field(
    field: &mut LetoArray3<Complex64>,
    factors: &LetoArray3<f64>,
) {
    assert_eq!(
        field.shape(),
        factors.shape(),
        "invariant: PSTD complex spectrum shape matches real multiplier shape"
    );

    let [_nx, ny, nz] = factors.shape();
    if let (Some(field_values), Some(factor_values)) = (field.as_slice_mut(), factors.as_slice()) {
        let element_bytes = size_of::<Complex64>() + size_of::<f64>();
        for_each_z_lane(
            field_values,
            [ny, nz],
            element_bytes,
            |start, _, _, lane| {
                for (value, &factor) in lane.iter_mut().zip(&factor_values[start..start + nz]) {
                    *value *= factor;
                }
            },
        );
    } else {
        let [nx, ny, nz] = field.shape();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field[[i, j, k]] *= factors[[i, j, k]];
                }
            }
        }
    }
}

pub(super) fn add_density_source_components(
    rhox: &mut LetoArray3<f64>,
    rhoy: Option<&mut LetoArray3<f64>>,
    rhoz: Option<&mut LetoArray3<f64>>,
    source: &LetoArray3<f64>,
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
                rhox.as_slice_mut(),
                ry.as_slice_mut(),
                rz.as_slice_mut(),
                source.as_slice(),
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
            if let (Some(rx_values), Some(ry_values), Some(source_values)) =
                (rhox.as_slice_mut(), ry.as_slice_mut(), source.as_slice())
            {
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
            if let (Some(rx_values), Some(rz_values), Some(source_values)) =
                (rhox.as_slice_mut(), rz.as_slice_mut(), source.as_slice())
            {
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
            if let (Some(rx_values), Some(source_values)) = (rhox.as_slice_mut(), source.as_slice())
            {
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
    rhox: &mut LetoArray3<f64>,
    mut rhoy: Option<&mut LetoArray3<f64>>,
    mut rhoz: Option<&mut LetoArray3<f64>>,
    source: &LetoArray3<f64>,
) {
    let [nx, ny, nz] = rhox.shape();
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

#[cfg(test)]
mod tests;

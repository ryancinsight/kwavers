//! Dense array operations shared by PSTD stepper paths.

use kwavers_math::fft::Complex64;
use leto::Array3 as LetoArray3;
use moirai_parallel::{
    enumerate_mut_with, for_each_chunk_mut_enumerated_with,
    for_each_chunk_pair_mut_enumerated_with, for_each_chunk_triple_mut_enumerated_with, Adaptive,
};
use leto::Array3 as NdArray3;

const DENSE_SOURCE_CHUNK: usize = 4096;

pub(super) trait DenseFieldMut {
    fn shape3(&self) -> [usize; 3];
    fn with_slice_mut<R>(&mut self, f: impl FnOnce(Option<&mut [f64]>) -> R) -> R;
    fn iter_mut_values<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut f64> + 'a>;
}

impl DenseFieldMut for NdArray3<f64> {
    fn shape3(&self) -> [usize; 3] {
        let (nx, ny, nz) = self.dim();
        [nx, ny, nz]
    }

    fn with_slice_mut<R>(&mut self, f: impl FnOnce(Option<&mut [f64]>) -> R) -> R {
        f(self.as_slice_memory_order_mut())
    }

    fn iter_mut_values<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut f64> + 'a> {
        Box::new(self.iter_mut())
    }
}

impl DenseFieldMut for LetoArray3<f64> {
    fn shape3(&self) -> [usize; 3] {
        self.shape()
    }

    fn with_slice_mut<R>(&mut self, f: impl FnOnce(Option<&mut [f64]>) -> R) -> R {
        f(self.as_slice_mut())
    }

    fn iter_mut_values<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut f64> + 'a> {
        Box::new(self.iter_mut())
    }
}

pub(super) fn scale_real_field<T: DenseFieldMut>(field: &mut T, factor: f64) {
    let used_dense_path = field.with_slice_mut(|values| {
        if let Some(values) = values {
            enumerate_mut_with::<Adaptive, _, _>(values, |_index, value| {
                *value *= factor;
            });
            true
        } else {
            false
        }
    });
    if used_dense_path {
        return;
    }
    for value in field.iter_mut_values() {
        *value *= factor;
    }
}

pub(super) fn add_masked_source_term<T: DenseFieldMut>(
    dst: &mut T,
    mask: &NdArray3<f64>,
    scale: f64,
) {
    assert_eq!(
        dst.shape3(),
        [mask.shape()[0], mask.shape()[1], mask.shape()[2]],
        "invariant: PSTD source accumulator shape matches source mask shape"
    );

    let used_dense_path = dst.with_slice_mut(|dst_values| {
        if let (Some(dst_values), Some(mask_values)) = (dst_values, mask.as_slice_memory_order()) {
            enumerate_mut_with::<Adaptive, _, _>(dst_values, |index, value| {
                let mask_value = mask_values[index];
                if mask_value.abs() > 1e-12 {
                    *value += mask_value * scale;
                }
            });
            true
        } else {
            false
        }
    });
    if used_dense_path {
        return;
    }
    for (value, &mask_value) in dst.iter_mut_values().zip(mask.iter()) {
        if mask_value.abs() > 1e-12 {
            *value += mask_value * scale;
        }
    }
}

pub(super) fn add_gradient_source_term(
    dst: &mut LetoArray3<f64>,
    grad_mask: &NdArray3<f64>,
    scale: f64,
) {
    assert_eq!(
        dst.shape(),
        grad_mask.shape(),
        "invariant: PSTD source accumulator shape matches velocity gradient mask shape"
    );

    if let (Some(dst_values), Some(mask_values)) =
        (dst.as_slice_mut(), grad_mask.as_slice_memory_order())
    {
        enumerate_mut_with::<Adaptive, _, _>(dst_values, |index, value| {
            *value += mask_values[index] * scale;
        });
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

    if let (Some(field_values), Some(factor_values)) = (field.as_slice_mut(), factors.as_slice()) {
        enumerate_mut_with::<Adaptive, _, _>(field_values, |index, value| {
            *value *= factor_values[index];
        });
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

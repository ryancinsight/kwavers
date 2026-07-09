//! Moirai-backed element-wise field operations for time-domain FWI.

use super::FWI_FIELD_CHUNK;
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};
use leto::{
    Array3,
    ArrayView3,
    ArrayViewMut3,
};

pub(in crate::inverse::fwi::time_domain) fn write_negative_product(
    dst: &mut Array3<f64>,
    lhs: &Array3<f64>,
    rhs: &Array3<f64>,
) {
    if dst.shape() == lhs.shape()
        && dst.shape() == rhs.shape()
        && dst
        && lhs
        && rhs
    {
        let lhs = lhs
            .as_slice()
            .expect("invariant: standard-layout lhs exposes memory-order slice");
        let rhs = rhs
            .as_slice()
            .expect("invariant: standard-layout rhs exposes memory-order slice");
        let dst = dst
            .as_slice_mut()
            .expect("invariant: standard-layout destination exposes memory-order slice");
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            dst,
            FWI_FIELD_CHUNK,
            |chunk_index, chunk| {
                let start = chunk_index * FWI_FIELD_CHUNK;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    *value = -lhs[start + offset] * rhs[start + offset];
                }
            },
        );
    } else {
        for ((i, j, k), value) in dst.indexed_iter_mut() {
            *value = -lhs[[i, j, k]] * rhs[[i, j, k]];
        }
    }
}

pub(in crate::inverse::fwi::time_domain) fn add_assign_field(
    dst: &mut Array3<f64>,
    src: &Array3<f64>,
) {
    if dst.shape() == src.shape() && dst && src {
        let src = src
            .as_slice()
            .expect("invariant: standard-layout source exposes memory-order slice");
        let dst = dst
            .as_slice_mut()
            .expect("invariant: standard-layout destination exposes memory-order slice");
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            dst,
            FWI_FIELD_CHUNK,
            |chunk_index, chunk| {
                let start = chunk_index * FWI_FIELD_CHUNK;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    *value += src[start + offset];
                }
            },
        );
    } else {
        for ((i, j, k), value) in dst.indexed_iter_mut() {
            *value += src[[i, j, k]];
        }
    }
}

pub(in crate::inverse::fwi::time_domain) fn add_scaled_field(
    dst: &mut Array3<f64>,
    src: &Array3<f64>,
    scale: f64,
) {
    if dst.shape() == src.shape() && dst && src {
        let src = src
            .as_slice()
            .expect("invariant: standard-layout source exposes memory-order slice");
        let dst = dst
            .as_slice_mut()
            .expect("invariant: standard-layout destination exposes memory-order slice");
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            dst,
            FWI_FIELD_CHUNK,
            |chunk_index, chunk| {
                let start = chunk_index * FWI_FIELD_CHUNK;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    *value += src[start + offset] * scale;
                }
            },
        );
    } else {
        for ((i, j, k), value) in dst.indexed_iter_mut() {
            *value += src[[i, j, k]] * scale;
        }
    }
}

pub(in crate::inverse::fwi::time_domain) fn divide_field_in_place(
    field: &mut Array3<f64>,
    denominator: f64,
) {
    if let Some(values) = field.as_slice_mut() {
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            values,
            FWI_FIELD_CHUNK,
            |_chunk_index, chunk| {
                for value in chunk {
                    *value /= denominator;
                }
            },
        );
    } else {
        field.iter_mut().for_each(|value| *value /= denominator);
    }
}

pub(in crate::inverse::fwi::time_domain) fn subtract_scaled_field(
    dst: &mut Array3<f64>,
    gradient: &Array3<f64>,
    scale: f64,
) {
    if dst.shape() == gradient.shape() && dst && gradient {
        let gradient = gradient
            .as_slice()
            .expect("invariant: standard-layout gradient exposes memory-order slice");
        let dst = dst
            .as_slice_mut()
            .expect("invariant: standard-layout destination exposes memory-order slice");
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            dst,
            FWI_FIELD_CHUNK,
            |chunk_index, chunk| {
                let start = chunk_index * FWI_FIELD_CHUNK;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    *value -= gradient[start + offset] * scale;
                }
            },
        );
    } else {
        for ((i, j, k), value) in dst.indexed_iter_mut() {
            *value -= gradient[[i, j, k]] * scale;
        }
    }
}

pub(in crate::inverse::fwi::time_domain) fn zero_masked_field(
    field: &mut Array3<f64>,
    frozen_mask: &Array3<bool>,
) {
    if field.shape() == frozen_mask.shape()
        && field
        && frozen_mask
    {
        let frozen_mask = frozen_mask
            .as_slice()
            .expect("invariant: standard-layout mask exposes memory-order slice");
        let field = field
            .as_slice_mut()
            .expect("invariant: standard-layout field exposes memory-order slice");
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            field,
            FWI_FIELD_CHUNK,
            |chunk_index, chunk| {
                let start = chunk_index * FWI_FIELD_CHUNK;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    if frozen_mask[start + offset] {
                        *value = 0.0;
                    }
                }
            },
        );
    } else {
        for ((i, j, k), value) in field.indexed_iter_mut() {
            if frozen_mask[[i, j, k]] {
                *value = 0.0;
            }
        }
    }
}

pub(in crate::inverse::fwi::time_domain) fn zero_masked_by_threshold(
    field: &mut Array3<f64>,
    mask: &Array3<f64>,
    threshold: f64,
) {
    if field.shape() == mask.shape() && field && mask {
        let mask = mask
            .as_slice()
            .expect("invariant: standard-layout mask exposes memory-order slice");
        let field = field
            .as_slice_mut()
            .expect("invariant: standard-layout field exposes memory-order slice");
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            field,
            FWI_FIELD_CHUNK,
            |chunk_index, chunk| {
                let start = chunk_index * FWI_FIELD_CHUNK;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    if mask[start + offset] > threshold {
                        *value = 0.0;
                    }
                }
            },
        );
    } else {
        for ((i, j, k), value) in field.indexed_iter_mut() {
            if mask[[i, j, k]] > threshold {
                *value = 0.0;
            }
        }
    }
}

pub(in crate::inverse::fwi::time_domain) fn scale_velocity_gradient(
    mut gradient: ArrayViewMut3<'_, f64>,
    model: ArrayView3<'_, f64>,
    density: ArrayView3<'_, f64>,
) {
    if gradient.shape() == model.shape()
        && gradient.shape() == density.shape()
        && gradient
        && model
        && density
    {
        let model = model
            .as_slice()
            .expect("invariant: standard-layout model exposes memory-order slice");
        let density = density
            .as_slice()
            .expect("invariant: standard-layout density exposes memory-order slice");
        let gradient = gradient
            .as_slice_mut()
            .expect("invariant: standard-layout gradient exposes memory-order slice");
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            gradient,
            FWI_FIELD_CHUNK,
            |chunk_index, chunk| {
                let start = chunk_index * FWI_FIELD_CHUNK;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    let idx = start + offset;
                    *value *= -2.0 / (density[idx] * model[idx].powi(3));
                }
            },
        );
    } else {
        for ((i, j, k), value) in gradient.indexed_iter_mut() {
            let c = model[[i, j, k]];
            let rho = density[[i, j, k]];
            *value *= -2.0 / (rho * c.powi(3));
        }
    }
}

pub(in crate::inverse::fwi::time_domain) fn apply_frozen_reference_or_clamp(
    model: &mut Array3<f64>,
    frozen_mask: &Array3<bool>,
    reference_model: &Array3<f64>,
    c_min: f64,
    c_max: f64,
) {
    if model
        && model.shape() == frozen_mask.shape()
        && model.shape() == reference_model.shape()
        && frozen_mask
        && reference_model
    {
        let frozen_mask = frozen_mask
            .as_slice()
            .expect("invariant: standard-layout mask exposes memory-order slice");
        let reference_model = reference_model
            .as_slice()
            .expect("invariant: standard-layout reference exposes memory-order slice");
        let model = model
            .as_slice_mut()
            .expect("invariant: standard-layout model exposes memory-order slice");
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            model,
            FWI_FIELD_CHUNK,
            |chunk_index, chunk| {
                let start = chunk_index * FWI_FIELD_CHUNK;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    let idx = start + offset;
                    if frozen_mask[idx] {
                        *value = reference_model[idx];
                    } else {
                        *value = value.clamp(c_min, c_max);
                    }
                }
            },
        );
    } else {
        for ((i, j, k), value) in model.indexed_iter_mut() {
            if frozen_mask[[i, j, k]] {
                *value = reference_model[[i, j, k]];
            } else {
                *value = value.clamp(c_min, c_max);
            }
        }
    }
}

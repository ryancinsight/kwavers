//! Shared traversal helpers for regularization gradients.

use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};
use ndarray::{ArrayBase, Data, DataMut, Dimension, Zip};

const REGULARIZATION_CHUNK_LEN: usize = 4096;

pub(super) fn for_each_pair_mut<Sg, Sm, D, F>(
    gradient: &mut ArrayBase<Sg, D>,
    model: &ArrayBase<Sm, D>,
    f: F,
) where
    Sg: DataMut<Elem = f64>,
    Sm: Data<Elem = f64>,
    D: Dimension,
    F: Fn(&mut f64, f64) + Send + Sync + Copy,
{
    assert_eq!(
        gradient.dim(),
        model.dim(),
        "regularization gradient and model shapes must match"
    );

    if let Some(model_values) = model.as_slice_memory_order() {
        if let Some(gradient_values) = gradient.as_slice_memory_order_mut() {
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                gradient_values,
                REGULARIZATION_CHUNK_LEN,
                |chunk_index, chunk| {
                    let base = chunk_index * REGULARIZATION_CHUNK_LEN;
                    for (offset, gradient_value) in chunk.iter_mut().enumerate() {
                        f(gradient_value, model_values[base + offset]);
                    }
                },
            );
            return;
        }
    }

    Zip::from(gradient)
        .and(model)
        .for_each(|gradient_value, &model_value| f(gradient_value, model_value));
}

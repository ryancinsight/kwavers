//! Shared traversal helpers for regularization gradients.

use leto::{Array, Storage, StorageMut};
use leto_ops::zip_mut_with;
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};

const REGULARIZATION_CHUNK_LEN: usize = 4096;

pub(super) fn for_each_pair_mut<Sg, Sm, F, const N: usize>(
    gradient: &mut Array<f64, Sg, N>,
    model: &Array<f64, Sm, N>,
    f: F,
) where
    Sg: StorageMut<f64>,
    Sm: Storage<f64>,
    F: Fn(&mut f64, f64) + Send + Sync + Copy,
{
    assert_eq!(
        gradient.shape(),
        model.shape(),
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

    let mut gradient_view = gradient.view_mut();
    let model_view = model.view();
    let _ = zip_mut_with(&mut gradient_view, &model_view, |g, m| f(g, *m));
}

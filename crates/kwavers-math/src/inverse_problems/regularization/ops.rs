//! Shared traversal helpers for regularization gradients.

use crate::parallel::zip_mut_ref;
use leto::{ArrayView, ArrayViewMut};

pub(super) fn for_each_pair_mut<const N: usize, F>(
    gradient: ArrayViewMut<'_, f64, N>,
    model: ArrayView<'_, f64, N>,
    f: F,
) where
    F: Fn(&mut f64, f64) + Send + Sync + Copy,
{
    zip_mut_ref(gradient, model, |gradient_value, &model_value| {
        f(gradient_value, model_value);
    });
}

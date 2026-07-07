//! Shared traversal helpers for regularization gradients.

use crate::parallel::zip_mut_ref;
use ndarray::{ArrayBase, Data, DataMut, Dimension};

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

    zip_mut_ref(
        gradient.view_mut(),
        model.view(),
        |gradient_value, &model_value| {
            f(gradient_value, model_value);
        },
    );
}

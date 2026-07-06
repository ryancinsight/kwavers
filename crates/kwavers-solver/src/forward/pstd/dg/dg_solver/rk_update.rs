//! Dense Runge-Kutta coefficient update helpers for DG states.

use moirai_parallel::{enumerate_mut_with, Adaptive};
use ndarray::Array3;

pub(super) fn update_euler(
    stage: &mut Array3<f64>,
    original: &Array3<f64>,
    rhs: &Array3<f64>,
    dt: f64,
) {
    assert_eq!(
        stage.shape(),
        original.shape(),
        "invariant: DG RK stage shape matches original state"
    );
    assert_eq!(
        stage.shape(),
        rhs.shape(),
        "invariant: DG RK stage shape matches RHS"
    );

    if let (Some(stage_values), Some(original_values), Some(rhs_values)) = (
        stage.as_slice_memory_order_mut(),
        original.as_slice_memory_order(),
        rhs.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(stage_values, |index, value| {
            *value = original_values[index] + dt * rhs_values[index];
        });
        return;
    }

    for ((value, &original), &rhs) in stage.iter_mut().zip(original.iter()).zip(rhs.iter()) {
        *value = original + dt * rhs;
    }
}

pub(super) fn update_ssp_second(
    stage: &mut Array3<f64>,
    original: &Array3<f64>,
    rhs: &Array3<f64>,
    dt: f64,
) {
    assert_eq!(
        stage.shape(),
        original.shape(),
        "invariant: DG RK stage shape matches original state"
    );
    assert_eq!(
        stage.shape(),
        rhs.shape(),
        "invariant: DG RK stage shape matches RHS"
    );

    if let (Some(stage_values), Some(original_values), Some(rhs_values)) = (
        stage.as_slice_memory_order_mut(),
        original.as_slice_memory_order(),
        rhs.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(stage_values, |index, value| {
            let stage_first = *value;
            *value = 0.75 * original_values[index] + 0.25 * (stage_first + dt * rhs_values[index]);
        });
        return;
    }

    for ((value, &original), &rhs) in stage.iter_mut().zip(original.iter()).zip(rhs.iter()) {
        let stage_first = *value;
        *value = 0.75 * original + 0.25 * (stage_first + dt * rhs);
    }
}

pub(super) fn update_ssp_final(
    target: &mut Array3<f64>,
    original: &Array3<f64>,
    stage: &Array3<f64>,
    rhs: &Array3<f64>,
    dt: f64,
) {
    assert_eq!(
        target.shape(),
        original.shape(),
        "invariant: DG RK target shape matches original state"
    );
    assert_eq!(
        target.shape(),
        stage.shape(),
        "invariant: DG RK target shape matches RK stage"
    );
    assert_eq!(
        target.shape(),
        rhs.shape(),
        "invariant: DG RK target shape matches RHS"
    );

    if let (Some(target_values), Some(original_values), Some(stage_values), Some(rhs_values)) = (
        target.as_slice_memory_order_mut(),
        original.as_slice_memory_order(),
        stage.as_slice_memory_order(),
        rhs.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(target_values, |index, value| {
            *value = (1.0 / 3.0) * original_values[index]
                + (2.0 / 3.0) * (stage_values[index] + dt * rhs_values[index]);
        });
        return;
    }

    for (((value, &original), &stage), &rhs) in target
        .iter_mut()
        .zip(original.iter())
        .zip(stage.iter())
        .zip(rhs.iter())
    {
        *value = (1.0 / 3.0) * original + (2.0 / 3.0) * (stage + dt * rhs);
    }
}

pub(super) fn update_forward_euler(target: &mut Array3<f64>, rhs: &Array3<f64>, dt: f64) {
    assert_eq!(
        target.shape(),
        rhs.shape(),
        "invariant: DG forward-Euler target shape matches RHS"
    );

    if let (Some(target_values), Some(rhs_values)) = (
        target.as_slice_memory_order_mut(),
        rhs.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(target_values, |index, value| {
            *value += dt * rhs_values[index];
        });
        return;
    }

    for (value, &rhs) in target.iter_mut().zip(rhs.iter()) {
        *value += dt * rhs;
    }
}

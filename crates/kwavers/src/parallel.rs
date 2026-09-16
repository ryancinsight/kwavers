//! Provider-owned traversal adapters for application arrays.

use leto::{ArrayView, ArrayViewMut};
use moirai_parallel::{for_each_unit_task_mut_with, for_each_unit_task_pair_mut_with, Adaptive};

/// Row-major odometer increment of a logical multi-index; returns `false` once
/// the index wraps past the final element. Drives the strided fallback walks.
#[inline]
fn next_index<const N: usize>(index: &mut [usize; N], shape: &[usize; N]) -> bool {
    for d in (0..N).rev() {
        index[d] += 1;
        if index[d] < shape[d] {
            return true;
        }
        index[d] = 0;
    }
    false
}

pub(crate) fn zip_mut_ref<T, U, const N: usize, F>(
    mut out: ArrayViewMut<'_, T, N>,
    input: ArrayView<'_, U, N>,
    f: F,
) where
    T: Send,
    U: Sync,
    F: Fn(&mut T, &U) + Send + Sync,
{
    assert_eq!(
        out.shape(),
        input.shape(),
        "invariant: paired traversal output shape must match input shape"
    );

    match (out.as_mut_slice(), input.as_slice()) {
        (Some(out), Some(input)) => {
            let f_ref = &f;
            // One unit is read and written in `out` and read once in the
            // input, which is what sizes the task (moirai ADR 0059).
            for_each_unit_task_mut_with::<Adaptive, _, _, _, _>(
                out,
                1,
                2 * core::mem::size_of::<T>() + core::mem::size_of::<U>(),
                || (),
                |(), first, run| {
                    for (lane, value) in run.iter_mut().enumerate() {
                        f_ref(value, &input[first + lane]);
                    }
                },
            );
        }
        _ => {
            let shape = out.shape();
            let mut index = [0usize; N];
            for _ in 0..out.size() {
                let value = out.get_mut(index).expect("invariant: index in bounds");
                f(value, input.get(index).expect("invariant: index in bounds"));
                next_index(&mut index, &shape);
            }
        }
    }
}

pub(crate) fn zip_two_mut_two_refs<T, U, V, W, const N: usize, F>(
    mut first_out: ArrayViewMut<'_, T, N>,
    mut second_out: ArrayViewMut<'_, U, N>,
    first: ArrayView<'_, V, N>,
    second: ArrayView<'_, W, N>,
    f: F,
) where
    T: Send,
    U: Send,
    V: Sync,
    W: Sync,
    F: Fn(&mut T, &mut U, &V, &W) + Send + Sync,
{
    assert_eq!(
        first_out.shape(),
        second_out.shape(),
        "invariant: paired traversal output shapes must match"
    );
    assert_eq!(
        first_out.shape(),
        first.shape(),
        "invariant: paired traversal first input shape must match output shape"
    );
    assert_eq!(
        first_out.shape(),
        second.shape(),
        "invariant: paired traversal second input shape must match output shape"
    );

    match (
        first_out.as_mut_slice(),
        second_out.as_mut_slice(),
        first.as_slice(),
        second.as_slice(),
    ) {
        (Some(first_out), Some(second_out), Some(first), Some(second)) => {
            let f_ref = &f;
            // One unit is read and written in both outputs and read once in
            // each input, which is what sizes the task (moirai ADR 0059).
            for_each_unit_task_pair_mut_with::<Adaptive, _, _, _, _, _>(
                first_out,
                second_out,
                1,
                2 * (core::mem::size_of::<T>() + core::mem::size_of::<U>())
                    + core::mem::size_of::<V>()
                    + core::mem::size_of::<W>(),
                || (),
                |(), first_unit, first_run, second_run| {
                    for (lane, (first_value, second_value)) in
                        first_run.iter_mut().zip(second_run.iter_mut()).enumerate()
                    {
                        let index = first_unit + lane;
                        f_ref(first_value, second_value, &first[index], &second[index]);
                    }
                },
            );
        }
        _ => {
            let shape = first_out.shape();
            let mut index = [0usize; N];
            for _ in 0..first_out.size() {
                let first_value = first_out
                    .get_mut(index)
                    .expect("invariant: index in bounds");
                let second_value = second_out
                    .get_mut(index)
                    .expect("invariant: index in bounds");
                f(
                    first_value,
                    second_value,
                    first.get(index).expect("invariant: index in bounds"),
                    second.get(index).expect("invariant: index in bounds"),
                );
                next_index(&mut index, &shape);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{zip_mut_ref, zip_two_mut_two_refs};
    use leto::Array3;

    /// A shape past the policy's parallel floor and several unit tasks wide.
    const SHAPE: [usize; 3] = [32, 32, 32];
    const LEN: usize = SHAPE[0] * SHAPE[1] * SHAPE[2];

    /// `index / 4` as an exactly representable value, distinct per element.
    fn ramp() -> Array3<f64> {
        let mut value = 0.0;
        Array3::from_shape_vec(
            SHAPE,
            (0..LEN)
                .map(|_| {
                    value += 0.25;
                    value
                })
                .collect(),
        )
        .expect("shape matches the data")
    }

    #[test]
    fn every_element_pairs_with_its_own_input() {
        let input = ramp();
        let mut out = Array3::zeros(SHAPE);

        zip_mut_ref(out.view_mut(), input.view(), |slot, &value| {
            *slot = value.mul_add(3.0, -1.0);
        });

        for (index, (got, value)) in out
            .as_slice()
            .unwrap()
            .iter()
            .zip(input.as_slice().unwrap())
            .enumerate()
        {
            assert_eq!(
                got.to_bits(),
                value.mul_add(3.0, -1.0).to_bits(),
                "out[{index}]"
            );
        }
    }

    #[test]
    fn both_outputs_pair_with_their_own_inputs() {
        let first_in = ramp();
        let second_in = ramp();
        let mut first_out = Array3::zeros(SHAPE);
        let mut second_out = Array3::zeros(SHAPE);

        zip_two_mut_two_refs(
            first_out.view_mut(),
            second_out.view_mut(),
            first_in.view(),
            second_in.view(),
            |first_slot, second_slot, &a, &b| {
                *first_slot = a.mul_add(2.0, b);
                *second_slot = b.mul_add(-3.0, a);
            },
        );

        let values = first_in.as_slice().unwrap();
        for (index, value) in values.iter().enumerate() {
            assert_eq!(
                first_out.as_slice().unwrap()[index].to_bits(),
                value.mul_add(2.0, *value).to_bits(),
                "first out[{index}]"
            );
            assert_eq!(
                second_out.as_slice().unwrap()[index].to_bits(),
                value.mul_add(-3.0, *value).to_bits(),
                "second out[{index}]"
            );
        }
    }
}

//! Provider-owned traversal adapters for math kernels.

use leto::{ArrayView, ArrayViewMut};
use leto_ops::zip_mut_with;
use moirai_parallel::{for_each_unit_task_mut_with, Adaptive};

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
        "invariant: math traversal output shape must match input shape"
    );

    match (out.as_mut_slice(), input.as_slice()) {
        (Some(out), Some(input)) => {
            let f_ref = &f;
            // One unit is one element: read and written in `out`, read once in
            // `input`. The width comes from those bytes (moirai ADR 0059); the
            // policy still decides whether to spread at all, as before.
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
        _ => zip_mut_with(&mut out, &input, f).unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::zip_mut_ref;
    use leto::Array1;

    #[test]
    fn every_element_pairs_with_its_own_input() {
        // Past the policy's parallel floor and several unit tasks wide, so a
        // task boundary that dropped, doubled or shifted a run shows up.
        const LEN: usize = 1 << 16;

        let mut value = 0.0_f64;
        let input = Array1::from_shape_vec(
            [LEN],
            (0..LEN)
                .map(|_| {
                    value += 0.25;
                    value
                })
                .collect(),
        )
        .unwrap();
        let mut out = Array1::from_shape_vec([LEN], vec![0.0_f64; LEN]).unwrap();

        zip_mut_ref(out.view_mut(), input.view(), |slot, &source| {
            *slot = source.mul_add(3.0, -1.0);
        });

        for (index, (got, source)) in out
            .as_slice()
            .unwrap()
            .iter()
            .zip(input.as_slice().unwrap())
            .enumerate()
        {
            assert_eq!(
                got.to_bits(),
                source.mul_add(3.0, -1.0).to_bits(),
                "out[{index}]"
            );
        }
    }
}

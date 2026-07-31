//! Provider-owned traversal adapters for math kernels.

use leto::ArrayViewMut;
use leto_ops::{zip_mut_with as provider_zip_mut_with, ZipSources};

pub(crate) fn zip_mut_with<T, S, F, const N: usize>(
    mut out: ArrayViewMut<'_, T, N>,
    sources: S,
    f: F,
) where
    S: ZipSources<N>,
    F: Fn(&mut T, S::Values),
{
    provider_zip_mut_with(&mut out, sources, f)
        .expect("invariant: math zip source shapes and storage must match output");
}

#[cfg(test)]
mod tests {
    use leto::Array3;

    use super::zip_mut_with;

    #[test]
    fn zip_mut_with_updates_dense_arrays() {
        let first = Array3::from_shape_fn((2, 2, 2), |[i, j, k]| (i + j + k) as i32);
        let second = Array3::from_shape_fn((2, 2, 2), |[i, j, k]| (2 * i + j + k) as i32);
        let mut out = Array3::zeros([2, 2, 2]);

        zip_mut_with(
            out.view_mut(),
            (&first.view(), &second.view()),
            |out, (first, second)| {
                *out = first + second;
            },
        );

        assert_eq!(
            out,
            Array3::from_shape_fn((2, 2, 2), |[i, j, k]| {
                (i + j + k) as i32 + (2 * i + j + k) as i32
            })
        );
    }
}

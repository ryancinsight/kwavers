//! The NumPy ndarray ownership boundary for Kwavers Python bindings.

use leto::{Array as LetoArray, VecStorage};
use numpy::ndarray::{Array, Dim, Dimension, IntoDimension};

/// Transfer owned storage to Leto in logical C order.
pub(crate) fn into_leto<T: Clone, const R: usize>(
    source: Array<T, Dim<[usize; R]>>,
) -> LetoArray<T, VecStorage<T>, R>
where
    Dim<[usize; R]>: Dimension,
{
    let shape: [usize; R] = source
        .shape()
        .try_into()
        .expect("invariant: fixed ndarray rank");
    let count = source.len();
    let values = if count == 0 {
        Vec::new()
    } else if source.is_standard_layout() {
        let (buffer, offset) = source.into_raw_vec_and_offset();
        let origin = offset.expect("invariant: nonempty ndarray has an origin");
        if origin == 0 && count == buffer.len() {
            buffer
        } else {
            // Restrict the backing allocation to the logical window of a
            // sliced owner. Logical C order is already physical order here.
            buffer.into_iter().skip(origin).take(count).collect()
        }
    } else {
        source.iter().cloned().collect()
    };
    LetoArray::from_vec(shape, values).expect("invariant: ndarray shape and logical storage agree")
}

/// Transfer a native array to NumPy's ndarray representation in logical C order.
pub(crate) fn into_ndarray<T: Clone, const R: usize>(
    source: LetoArray<T, VecStorage<T>, R>,
) -> Array<T, Dim<[usize; R]>>
where
    [usize; R]: IntoDimension<Dim = Dim<[usize; R]>>,
    Dim<[usize; R]>: Dimension,
{
    let shape = source.shape();
    Array::from_shape_vec(shape, source.into_vec())
        .expect("invariant: Leto shape and logical storage agree")
}

#[cfg(test)]
mod tests;

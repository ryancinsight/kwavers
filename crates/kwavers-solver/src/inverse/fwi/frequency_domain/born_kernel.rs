//! Native-precision weighted Born field integration and real adjoint projection.

use eunomia::{Complex, FloatElement};
use std::ops::Range;

pub(super) mod arithmetic;
pub(super) mod fields;
use arithmetic::Arithmetic;
use fields::FieldProducts;

/// Cubic voxel volume with the explicit two-product rounding contract.
/// General `powi` has unspecified precision and can vary between evaluations;
/// a model and its finite-difference probes require the same spatial measure.
#[inline]
pub(super) fn cell_volume<T: FloatElement>(spacing: T) -> T {
    (spacing * spacing) * spacing
}

/// Integrates `(incident * green) * weight` in the original voxel order.
/// Scratch retains only the immutable spatial products: the pressure consumer
/// never reads weighted fields. The returned slice exposes the prepared prefix;
/// spare lanes remain untouched. All weights and fold additions remain fresh
/// and execute in native precision without reassociation or intermediate stores.
/// Equal input lengths and output capacity are checked before computation.
#[inline]
pub(super) fn integrate_products<'out, T: Arithmetic, P: FieldProducts<T>, const K: usize>(
    fields: &P,
    range: Range<usize>,
    weights: &[T],
    output: &'out mut [Complex<T>; K],
    sum: &mut Complex<T>,
) -> &'out [Complex<T>] {
    let len = range
        .end
        .checked_sub(range.start)
        .expect("Born block range is reversed");
    assert_eq!(len, weights.len(), "Born field/weight lengths differ");
    fields.write_products(range, output);
    T::integrate_block(output, weights, sum);
    &output[..len]
}

/// Bound a block's active working set to 32 KiB: incident, Green and derivative
/// complex values plus weights, gradient and accumulated gradient require
/// nine native FD scalars per voxel. The largest fitting
/// power of two is 256 voxels. This storage budget changes traversal only;
/// correctness is independent of cache size and all voxels remain included.
pub(super) const VOXELS_PER_BLOCK: usize = {
    let capacity = 32 * 1024 / (9 * std::mem::size_of::<f64>());
    1 << (usize::BITS - 1 - capacity.leading_zeros())
};

/// The same left fold as `Complex::sum`, over checked blocks and their tail.
/// Borrowed blocks and reused output storage amortize slice projections;
/// all arithmetic stays in `T`.
/// Each counter visits exactly `[0, bound)`: its prefix contains precisely the
/// contributions already folded. The final increment reaches `bound`, never
/// exceeding it; each scalar expression and its reduction order remain unchanged.
#[inline]
pub(super) fn integrate<T: Arithmetic, P: FieldProducts<T>, const K: usize>(
    fields: &P,
    weights: &[T],
) -> Complex<T> {
    const { assert!(K > 0, "Born block capacity must be positive") };
    assert_eq!(
        fields.len(),
        weights.len(),
        "Born field/weight lengths differ"
    );
    let mut sum = Complex {
        re: T::ZERO,
        im: T::ZERO,
    };
    let mut values = [Complex {
        re: T::ZERO,
        im: T::ZERO,
    }; K];
    for (block, weights) in weights.chunks(K).enumerate() {
        let start = block * K;
        integrate_products(
            fields,
            start..start + weights.len(),
            weights,
            &mut values,
            &mut sum,
        );
    }
    sum
}

/// `Re(conj(derivative) * residual)` needs no imaginary output. The subtraction
/// and negation match the original complex multiplication even for signed zero.
/// Const selection either assigns that value or adds it once to the caller's
/// existing accumulator. Both use one expression; the accumulation mode removes
/// an intermediate projection buffer and second traversal without reassociation.
#[inline]
pub(super) fn project_adjoint<
    'out,
    T: Arithmetic,
    P: FieldProducts<T>,
    const K: usize,
    const ACCUMULATE: bool,
>(
    fields: &P,
    range: Range<usize>,
    weights: &[T],
    residual: Complex<T>,
    products: &mut [Complex<T>; K],
    output: &'out mut [T; K],
) -> &'out [T] {
    let len = range
        .end
        .checked_sub(range.start)
        .expect("Born block range is reversed");
    assert_eq!(len, weights.len(), "Born field/weight lengths differ");
    fields.write_products(range, products);
    T::project_block::<K, ACCUMULATE>(products, weights, residual, output);
    &output[..len]
}

#[cfg(test)]
mod tests;

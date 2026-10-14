//! Native-precision weighted Born field integration and real adjoint projection.

use eunomia::{Complex, FloatElement};
use std::ops::Neg;

/// Cubic voxel volume with the explicit two-product rounding contract.
/// General `powi` has unspecified precision and can vary between evaluations;
/// a model and its finite-difference probes require the same spatial measure.
#[inline]
pub(super) fn cell_volume<T: FloatElement>(spacing: T) -> T {
    (spacing * spacing) * spacing
}

/// `(incident * green) * weight`, preserving the complex product's expression
/// tree. The sealed scalar contract is Copy; reading its fields directly
/// avoids Clone calls and temporary complex operations without widening,
/// reassociation or a change to the resulting real/imaginary components.
/// Equal input lengths and output capacity are checked before computation.
/// The returned slice exposes precisely the overwritten prefix; spare lanes
/// are never read or written. This gives full and partial blocks one kernel.
#[inline]
pub(super) fn weighted_products<'out, T: FloatElement, const K: usize>(
    incident: &[Complex<T>],
    green: &[Complex<T>],
    weights: &[T],
    output: &'out mut [Complex<T>; K],
) -> &'out [Complex<T>] {
    assert_eq!(
        incident.len(),
        green.len(),
        "Born field/Green lengths differ"
    );
    assert_eq!(
        incident.len(),
        weights.len(),
        "Born field/weight lengths differ"
    );
    let len = incident.len();
    assert!(len <= K, "Born block exceeds output capacity");
    let mut lane = 0;
    while lane < len {
        let (incident, green, weight) = (incident[lane], green[lane], weights[lane]);
        output[lane] = Complex {
            re: (incident.re * green.re - incident.im * green.im) * weight,
            im: (incident.re * green.im + incident.im * green.re) * weight,
        };
        lane += 1;
    }
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
pub(super) fn integrate<T: FloatElement, const K: usize>(
    incident: &[Complex<T>],
    green: &[Complex<T>],
    weights: &[T],
) -> Complex<T> {
    const { assert!(K > 0, "Born block capacity must be positive") };
    assert_eq!(
        incident.len(),
        green.len(),
        "Born field/Green lengths differ"
    );
    assert_eq!(
        incident.len(),
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
    for ((incident, green), weights) in incident
        .chunks(K)
        .zip(green.chunks(K))
        .zip(weights.chunks(K))
    {
        let values = weighted_products(incident, green, weights, &mut values);
        let len = values.len();
        let mut lane = 0;
        while lane < len {
            let value = values[lane];
            sum.re += value.re;
            sum.im += value.im;
            lane += 1;
        }
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
    T: FloatElement + Neg<Output = T>,
    const K: usize,
    const ACCUMULATE: bool,
>(
    derivatives: &[Complex<T>],
    residual: Complex<T>,
    output: &'out mut [T; K],
) -> &'out [T] {
    let len = derivatives.len();
    assert!(len <= K, "Born block exceeds output capacity");
    let mut lane = 0;
    while lane < len {
        let derivative = derivatives[lane];
        let projection = derivative.re * residual.re - (-derivative.im) * residual.im;
        if ACCUMULATE {
            output[lane] += projection;
        } else {
            output[lane] = projection;
        }
        lane += 1;
    }
    &output[..len]
}

#[cfg(test)]
mod tests;

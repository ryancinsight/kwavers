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
#[inline]
pub(super) fn weighted_product<T: FloatElement>(
    incident: Complex<T>,
    green: Complex<T>,
    weight: T,
) -> Complex<T> {
    Complex {
        re: (incident.re * green.re - incident.im * green.im) * weight,
        im: (incident.re * green.im + incident.im * green.re) * weight,
    }
}

/// Eight native FD scalars occupy one 64-byte row block. The const arity only
/// partitions traversal; every voxel and addition retains its original order.
pub(super) const VOXELS_PER_BLOCK: usize = 64 / std::mem::size_of::<f64>();

/// The same left fold as `Complex::sum`, over checked blocks and their tail.
/// Copy arrays amortize slice projections while all arithmetic stays in `T`.
#[inline]
pub(super) fn integrate<T: FloatElement, const K: usize>(
    incident: &[Complex<T>],
    green: &[Complex<T>],
    weights: &[T],
) -> Complex<T> {
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
    let (incident, incident_tail) = incident.as_chunks::<K>();
    let (green, green_tail) = green.as_chunks::<K>();
    let (weights, weights_tail) = weights.as_chunks::<K>();
    let mut sum = Complex {
        re: T::ZERO,
        im: T::ZERO,
    };
    for ((incident, green), weights) in incident.iter().zip(green).zip(weights) {
        let (incident, green, weights) = (*incident, *green, *weights);
        for lane in 0..K {
            let value = weighted_product(incident[lane], green[lane], weights[lane]);
            sum.re += value.re;
            sum.im += value.im;
        }
    }
    for lane in 0..incident_tail.len() {
        let value = weighted_product(incident_tail[lane], green_tail[lane], weights_tail[lane]);
        sum.re += value.re;
        sum.im += value.im;
    }
    sum
}

/// `Re(conj(derivative) * residual)` needs no imaginary output. The subtraction
/// and negation match the original complex multiplication even for signed zero.
#[inline]
pub(super) fn project_adjoint<T: FloatElement + Neg<Output = T>>(
    derivative: Complex<T>,
    residual: Complex<T>,
) -> T {
    derivative.re * residual.re - (-derivative.im) * residual.im
}

#[cfg(test)]
mod tests;

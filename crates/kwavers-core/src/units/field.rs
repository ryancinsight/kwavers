//! Dimension-tagged sample fields.
//!
//! A grid of `f64` samples carries no unit, so a fluence field and a deposition
//! field are the same type to the compiler even though substituting one for the
//! other is a physics error. [`DimensionedField`] attaches an Aequitas dimension
//! to the container as a zero-sized tag, making that substitution a compile
//! error while leaving the runtime representation untouched.
//!
//! The tag is on the field rather than on each sample deliberately: per-sample
//! `Quantity` values would change the memory layout and block the slice, SIMD,
//! and GPU paths the solvers rely on. Samples stay canonical-SI `f64`, exactly
//! as Aequitas stores a scalar quantity.

use aequitas::Quantity;
use core::fmt;
use core::marker::PhantomData;

/// Samples in the canonical SI base unit of dimension `D`, held in container
/// `S`.
///
/// `S` is whatever the producer already uses — an owned array, a borrowed view,
/// or a flat slice — so tagging costs no copy and no allocation.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DimensionedField<S, D> {
    samples: S,
    dimension: PhantomData<D>,
}

impl<S, D> DimensionedField<S, D> {
    /// Adopt samples that are already in the canonical SI base unit of `D`.
    ///
    /// This is the single point where an untagged container becomes a
    /// dimensioned one, so it is the place to check a producer's unit claim.
    #[must_use]
    pub const fn from_base(samples: S) -> Self {
        Self {
            samples,
            dimension: PhantomData,
        }
    }

    /// Borrow the underlying samples.
    #[must_use]
    pub const fn samples(&self) -> &S {
        &self.samples
    }

    /// Mutably borrow the underlying samples.
    #[must_use]
    pub const fn samples_mut(&mut self) -> &mut S {
        &mut self.samples
    }

    /// Move the samples out, discarding the dimension tag.
    #[must_use]
    pub fn into_samples(self) -> S {
        self.samples
    }

    /// Re-tag the same samples with a different dimension.
    ///
    /// Only correct where a derivation genuinely changes the dimension without
    /// changing the numbers. Prefer Aequitas arithmetic on the sampled
    /// quantities; reach for this at a boundary a producer has already
    /// converted.
    #[must_use]
    pub fn retag<E>(self) -> DimensionedField<S, E> {
        DimensionedField::from_base(self.samples)
    }
}

impl<S, D> DimensionedField<S, D> {
    /// Sample one element as a dimensional quantity.
    ///
    /// The index type is generic so one method serves flat slices (`usize`) and
    /// three-dimensional arrays (`[usize; 3]`) alike.
    #[must_use]
    pub fn quantity_at<I>(&self, index: I) -> Quantity<f64, D>
    where
        S: core::ops::Index<I, Output = f64>,
    {
        Quantity::from_base(self.samples[index])
    }
}

impl<S, D> fmt::Debug for DimensionedField<S, D> {
    /// Report the dimension rather than the samples: a voxel dump is not
    /// diagnostic, and `S` is frequently a view type without `Debug`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DimensionedField")
            .field("dimension", &core::any::type_name::<D>())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::DimensionedField;
    use aequitas::systems::si::{
        dimensions,
        units::{WattPerCubicMeter, WattPerSquareMeter},
    };

    type Deposition<S> = DimensionedField<S, dimensions::VolumetricPowerDensity>;
    type FluenceRate<S> = DimensionedField<S, dimensions::Intensity>;

    #[test]
    fn tagging_costs_no_storage_over_the_container() {
        assert_eq!(
            core::mem::size_of::<Deposition<Vec<f64>>>(),
            core::mem::size_of::<Vec<f64>>()
        );
        assert_eq!(
            core::mem::size_of::<Deposition<&[f64]>>(),
            core::mem::size_of::<&[f64]>()
        );
    }

    #[test]
    fn samples_read_back_in_the_declared_unit() {
        let deposition = Deposition::from_base(vec![1_500.0_f64, 0.0]);

        assert_eq!(
            deposition
                .quantity_at(0)
                .in_unit::<WattPerCubicMeter>()
                .to_bits(),
            1_500.0_f64.to_bits()
        );
    }

    /// The tag is the whole point: two fields with identical storage are
    /// distinct types, so a fluence rate cannot reach a deposition parameter.
    #[test]
    fn distinct_dimensions_are_distinct_types() {
        fn deposit(_field: &Deposition<Vec<f64>>) {}

        let rate = FluenceRate::from_base(vec![2.0_f64]);
        // `deposit(&rate)` does not compile; retagging is explicit.
        let deposition: Deposition<Vec<f64>> = rate.retag();
        deposit(&deposition);

        assert_eq!(
            deposition
                .quantity_at(0)
                .in_unit::<WattPerCubicMeter>()
                .to_bits(),
            2.0_f64.to_bits()
        );
        let rate = FluenceRate::from_base(vec![2.0_f64]);
        assert_eq!(
            rate.quantity_at(0)
                .in_unit::<WattPerSquareMeter>()
                .to_bits(),
            2.0_f64.to_bits()
        );
    }
}

//! Quantity-typed host storage for heterogeneous acoustic absorption.

use aequitas::Quantity;
use leto::Array3;
use std::marker::PhantomData;

/// A spatial field whose samples share one Aequitas dimension.
///
/// Storage stays contiguous in SI base units for zero-copy Leto views at
/// solver boundaries. Point reads and writes carry the field dimension.
///
/// ```compile_fail,E0308
/// use aequitas::systems::si::{dimensions, quantities::Frequency};
/// use kwavers_medium::heterogeneous::AbsorptionField;
/// let mut loss = AbsorptionField::<dimensions::ReciprocalLength>::from_base(
///     leto::Array3::zeros([1, 1, 1]),
/// );
/// loss.set([0, 0, 0], Frequency::from_base(2.0e6));
/// ```
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct AbsorptionField<D> {
    values: Array3<f64>,
    dimension: PhantomData<D>,
}

impl<D> AbsorptionField<D> {
    /// Adopt an existing array whose samples are in the SI base unit.
    #[must_use]
    pub fn from_base(values: Array3<f64>) -> Self {
        Self {
            values,
            dimension: PhantomData,
        }
    }

    /// Read the typed sample at a grid index.
    #[must_use]
    pub fn get(&self, index: [usize; 3]) -> Quantity<f64, D> {
        Quantity::from_base(self.values[index])
    }

    /// Replace the sample at a grid index.
    pub fn set(&mut self, index: [usize; 3], sample: Quantity<f64, D>) {
        self.values[index] = sample.into_base();
    }

    /// Assign the same typed sample to every grid cell.
    pub fn fill(&mut self, sample: Quantity<f64, D>) {
        self.values.fill(sample.into_base());
    }

    pub(crate) fn base(&self) -> &Array3<f64> {
        &self.values
    }

    pub(crate) fn base_mut(&mut self) -> &mut Array3<f64> {
        &mut self.values
    }
}

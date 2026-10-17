//! Statically dispatched representations of model-independent field products.

use super::arithmetic::Arithmetic;
use eunomia::{Complex, FloatElement};
use std::ops::Range;

mod sealed {
    pub trait Sealed {}
}

/// A representation writes the exact spatial product for a checked range.
/// Only this module implements the contract; all weighting and reductions
/// remain in the single generic consumer. No model-dependent result is stored.
pub(in super::super) trait FieldProducts<T: FloatElement>: sealed::Sealed {
    fn len(&self) -> usize;

    fn write_range(&self, range: Range<usize>, output: &mut [Complex<T>]);

    #[inline]
    fn write_products<const K: usize>(&self, range: Range<usize>, output: &mut [Complex<T>; K]) {
        let len = range
            .end
            .checked_sub(range.start)
            .expect("Born block range is reversed");
        assert!(range.end <= self.len(), "Born block exceeds field length");
        assert!(len <= K, "Born block exceeds output capacity");
        self.write_range(range, &mut output[..len]);
    }
}

/// Separate incident and Green fields retain the original matrix-free storage.
#[derive(Clone, Copy, Debug)]
pub(in super::super) struct FactoredProducts<'a, T: FloatElement> {
    incident: &'a [Complex<T>],
    green: &'a [Complex<T>],
}

impl<'a, T: FloatElement> FactoredProducts<'a, T> {
    #[inline]
    pub(in super::super) fn new(incident: &'a [Complex<T>], green: &'a [Complex<T>]) -> Self {
        assert_eq!(
            incident.len(),
            green.len(),
            "Born field/Green lengths differ"
        );
        Self { incident, green }
    }
}

impl<T: FloatElement> sealed::Sealed for FactoredProducts<'_, T> {}

impl<T: Arithmetic> FieldProducts<T> for FactoredProducts<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.incident.len()
    }

    #[inline]
    fn write_range(&self, range: Range<usize>, output: &mut [Complex<T>]) {
        let len = output.len();
        let mut lane = 0;
        while lane < len {
            let incident = self.incident[range.start + lane];
            let green = self.green[range.start + lane];
            output[lane] = <T as Arithmetic>::product(incident, green);
            lane += 1;
        }
    }
}

/// Products prepared by the same canonical expression, without model weights.
#[derive(Clone, Copy, Debug)]
pub(in super::super) struct StoredProducts<'a, T: FloatElement> {
    values: &'a [Complex<T>],
}

impl<'a, T: FloatElement> StoredProducts<'a, T> {
    #[inline]
    pub(in super::super) fn new(values: &'a [Complex<T>]) -> Self {
        Self { values }
    }
}

impl<T: FloatElement> sealed::Sealed for StoredProducts<'_, T> {}

impl<T: FloatElement> FieldProducts<T> for StoredProducts<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    fn write_range(&self, range: Range<usize>, output: &mut [Complex<T>]) {
        output.copy_from_slice(&self.values[range]);
    }
}

#[cfg(test)]
mod tests;

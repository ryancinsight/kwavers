//! Domain-separated counter designs.

use leto::Array2;
use tyche_core::{Counter, Design, SampleIndexError, Seed, SplitMix64, UserDomain};

use super::DesignSamplingExt;
use crate::geometry::{GeometricDomain, GeometryError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CounterDesign<const PARAMETERS: usize, const TAG: u64> {
    seed: Seed,
    sample_count: usize,
}

impl<const PARAMETERS: usize, const TAG: u64> Design<PARAMETERS>
    for CounterDesign<PARAMETERS, TAG>
{
    fn sample_count(&self) -> usize {
        self.sample_count
    }

    fn sample_unit_into(
        &self,
        index: usize,
        output: &mut [f64; PARAMETERS],
    ) -> Result<(), SampleIndexError> {
        if index >= self.sample_count {
            return Err(SampleIndexError::new(index, self.sample_count));
        }
        let index = u64::try_from(index)
            .expect("invariant: Tyche-supported targets have at most 64-bit usize");
        for (draw, coordinate) in output.iter_mut().enumerate() {
            let draw =
                u64::try_from(draw).expect("invariant: an array dimension is representable as u64");
            *coordinate = Counter::<UserDomain<TAG>, SplitMix64>::open_unit(self.seed, index, draw);
        }
        Ok(())
    }
}

pub(crate) fn sample_counter<
    const PARAMETERS: usize,
    const TAG: u64,
    G: GeometricDomain + ?Sized,
>(
    domain: &G,
    sample_count: usize,
    seed: Seed,
) -> Result<Array2<f64>, GeometryError> {
    domain.sample_design(&CounterDesign::<PARAMETERS, TAG> { seed, sample_count })
}

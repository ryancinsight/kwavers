//! Generic collocation sampler.

use core::num::NonZeroU32;

use kwavers_grid::geometry::sampling::DesignSamplingExt;
use kwavers_grid::geometry::{GeometricDomain, GeometryDimension, GeometryError};
use leto::Array2;
use tyche_core::{DigitalShift, LatinHypercube, Seed, Sobol, SobolRange, SplitMix64};

use super::CollocationSamplingStrategy;

/// Statically dispatched PINN collocation sampler.
///
/// The geometric domain is stored inline as `G`; there is no heap allocation
/// or vtable dispatch in the sampling path. Strategy selection occurs once per
/// requested design, then the complete fixed-dimensional Tyche kernel
/// monomorphizes through [`DesignSamplingExt`]. Boundary charts remain owned
/// by the domain and are independent of the interior experimental design.
#[derive(Debug, Clone)]
pub struct CollocationSampler<G> {
    domain: G,
    strategy: CollocationSamplingStrategy,
    seed: Seed,
}

impl<G: GeometricDomain> CollocationSampler<G> {
    /// Construct a reproducible collocation sampler.
    #[must_use]
    pub const fn new(domain: G, strategy: CollocationSamplingStrategy, seed: Seed) -> Self {
        Self {
            domain,
            strategy,
            seed,
        }
    }

    /// Borrow the physical domain.
    #[must_use]
    pub const fn domain(&self) -> &G {
        &self.domain
    }

    /// Configured interior design strategy.
    #[must_use]
    pub const fn strategy(&self) -> CollocationSamplingStrategy {
        self.strategy
    }

    /// Reproducibility seed.
    pub const fn seed(&self) -> Seed {
        self.seed
    }

    /// Generate exactly `sample_count` strict-interior collocation points.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError`] when a Latin-hypercube or Sobol request
    /// exceeds Tyche's validated `u32` range, or when the output shape is not
    /// addressable or reservable.
    pub fn sample_interior(&self, sample_count: usize) -> Result<Array2<f64>, GeometryError> {
        if sample_count == 0 {
            return self.domain.sample_interior(0, self.seed);
        }
        match self.strategy {
            CollocationSamplingStrategy::Uniform => {
                self.domain.sample_interior(sample_count, self.seed)
            }
            CollocationSamplingStrategy::LatinHypercube => {
                let count = design_count(sample_count)?;
                match self.domain.dimension() {
                    GeometryDimension::One => self.sample_latin_hypercube::<1>(count),
                    GeometryDimension::Two => self.sample_latin_hypercube::<2>(count),
                    GeometryDimension::Three => self.sample_latin_hypercube::<3>(count),
                }
            }
            CollocationSamplingStrategy::Sobol => {
                let count = design_count(sample_count)?;
                match self.domain.dimension() {
                    GeometryDimension::One => self.sample_sobol::<1>(count),
                    GeometryDimension::Two => self.sample_sobol::<2>(count),
                    GeometryDimension::Three => self.sample_sobol::<3>(count),
                }
            }
        }
    }

    /// Generate exactly `sample_count` domain-boundary points.
    ///
    /// Boundary parameterization is a domain concern, so the configured
    /// interior design does not replace the domain's measure-correct chart.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError`] when the output shape is not addressable or
    /// reservable.
    pub fn sample_boundary(&self, sample_count: usize) -> Result<Array2<f64>, GeometryError> {
        self.domain.sample_boundary(sample_count, self.seed)
    }

    fn sample_latin_hypercube<const PARAMETERS: usize>(
        &self,
        sample_count: NonZeroU32,
    ) -> Result<Array2<f64>, GeometryError> {
        let design = LatinHypercube::<PARAMETERS, SplitMix64>::new(self.seed, sample_count);
        self.domain.sample_design(&design)
    }

    fn sample_sobol<const PARAMETERS: usize>(
        &self,
        sample_count: NonZeroU32,
    ) -> Result<Array2<f64>, GeometryError> {
        let range = SobolRange::new(1, sample_count)
            .expect("invariant: a non-zero u32 count starting at one ends at most at u32::MAX");
        let scramble = DigitalShift::<SplitMix64>::new(self.seed);
        let design = Sobol::<PARAMETERS, _>::new(range, scramble)
            .expect("invariant: geometric domains have one, two, or three dimensions");
        self.domain.sample_design(&design)
    }
}

fn design_count(sample_count: usize) -> Result<NonZeroU32, GeometryError> {
    let count =
        u32::try_from(sample_count).map_err(|_| GeometryError::SampleCountExceedsLimit {
            sample_count,
            maximum: u32::MAX,
        })?;
    Ok(NonZeroU32::new(count).expect("invariant: zero counts return before design construction"))
}

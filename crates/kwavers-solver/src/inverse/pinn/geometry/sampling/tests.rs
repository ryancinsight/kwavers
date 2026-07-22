use core::num::NonZeroU32;

use kwavers_grid::geometry::sampling::DesignSamplingExt;
use kwavers_grid::geometry::{GeometricDomain, PointLocation, RectangularDomain, SphericalDomain};
use tyche_core::{DigitalShift, LatinHypercube, Seed, Sobol, SobolRange, SplitMix64};

use super::{CollocationSampler, CollocationSamplingStrategy};

#[test]
fn every_strategy_has_exact_cardinality_replay_and_membership() {
    let seed = Seed::new(42);
    for strategy in [
        CollocationSamplingStrategy::Uniform,
        CollocationSamplingStrategy::LatinHypercube,
        CollocationSamplingStrategy::Sobol,
    ] {
        verify_domain(
            RectangularDomain::new_1d(-1.0, 2.0).expect("valid interval"),
            strategy,
            seed,
        );
        verify_domain(
            RectangularDomain::new_2d(-1.0, 2.0, 3.0, 5.0).expect("valid rectangle"),
            strategy,
            seed,
        );
        verify_domain(
            RectangularDomain::new_3d(-1.0, 2.0, 3.0, 5.0, -4.0, -2.0).expect("valid cuboid"),
            strategy,
            seed,
        );
        verify_domain(
            SphericalDomain::new_2d(1.0, -2.0, 3.0).expect("valid disk"),
            strategy,
            seed,
        );
        verify_domain(
            SphericalDomain::new_3d(1.0, -2.0, 3.0, 4.0).expect("valid ball"),
            strategy,
            seed,
        );
    }
}

fn verify_domain<G: GeometricDomain>(domain: G, strategy: CollocationSamplingStrategy, seed: Seed) {
    let sampler = CollocationSampler::new(domain, strategy, seed);
    let points = sampler.sample_interior(257).expect("addressable design");
    let replay = sampler.sample_interior(257).expect("addressable replay");
    assert_eq!(points, replay);
    assert_eq!(
        points.shape(),
        [257, sampler.domain().dimension().as_usize()]
    );
    for row in 0..points.shape()[0] {
        let point = (0..points.shape()[1])
            .map(|column| points[[row, column]])
            .collect::<Vec<_>>();
        assert_eq!(
            sampler.domain().classify_point(&point, 0.0),
            PointLocation::Interior
        );
    }

    let boundary = sampler
        .sample_boundary(113)
        .expect("addressable boundary design");
    assert_eq!(
        boundary.shape(),
        [113, sampler.domain().dimension().as_usize()]
    );
}

#[test]
fn zero_count_preserves_matrix_dimension_for_every_strategy() {
    let domain = RectangularDomain::new_3d(0.0, 1.0, 0.0, 1.0, 0.0, 1.0).expect("valid cuboid");
    for strategy in [
        CollocationSamplingStrategy::Uniform,
        CollocationSamplingStrategy::LatinHypercube,
        CollocationSamplingStrategy::Sobol,
    ] {
        let points = CollocationSampler::new(domain, strategy, Seed::new(5))
            .sample_interior(0)
            .expect("empty output");
        assert_eq!(points.shape(), [0, 3]);
    }
}

#[test]
fn interior_strategy_does_not_change_the_domain_boundary_chart() {
    let domain = SphericalDomain::new_3d(1.0, -2.0, 3.0, 4.0).expect("valid ball");
    let seed = Seed::new(31);
    let expected = CollocationSampler::new(domain, CollocationSamplingStrategy::Uniform, seed)
        .sample_boundary(127)
        .expect("addressable boundary");

    for strategy in [
        CollocationSamplingStrategy::LatinHypercube,
        CollocationSamplingStrategy::Sobol,
    ] {
        let actual = CollocationSampler::new(domain, strategy, seed)
            .sample_boundary(127)
            .expect("addressable boundary");
        assert_eq!(actual, expected);
    }
}

#[test]
fn strategy_dispatch_matches_the_provider_design_exactly() {
    let seed = Seed::new(71);
    let count = NonZeroU32::new(64).expect("non-zero fixture count");
    let domain = RectangularDomain::new_2d(-2.0, 3.0, 5.0, 11.0).expect("valid rectangle");

    let latin = LatinHypercube::<2, SplitMix64>::new(seed, count);
    let expected_latin = domain.sample_design(&latin).expect("valid provider design");
    let actual_latin =
        CollocationSampler::new(domain, CollocationSamplingStrategy::LatinHypercube, seed)
            .sample_interior(64)
            .expect("valid collocation design");
    assert_eq!(actual_latin, expected_latin);

    let range = SobolRange::new(1, count).expect("valid Sobol range");
    let sobol = Sobol::<2, _>::new(range, DigitalShift::<SplitMix64>::new(seed))
        .expect("supported dimension");
    let expected_sobol = domain.sample_design(&sobol).expect("valid provider design");
    let actual_sobol = CollocationSampler::new(domain, CollocationSamplingStrategy::Sobol, seed)
        .sample_interior(64)
        .expect("valid collocation design");
    assert_eq!(actual_sobol, expected_sobol);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn finite_designs_reject_counts_above_tyche_range() {
    let domain = RectangularDomain::new_1d(0.0, 1.0).expect("valid interval");
    let sampler = CollocationSampler::new(domain, CollocationSamplingStrategy::Sobol, Seed::new(5));
    let count = usize::try_from(u64::from(u32::MAX) + 1).expect("64-bit usize");
    assert!(matches!(
        sampler.sample_interior(count),
        Err(kwavers_grid::geometry::GeometryError::SampleCountExceedsLimit {
            sample_count,
            maximum: u32::MAX,
        }) if sample_count == count
    ));
}

use ndarray::Array1;

use crate::domain::geometry::RectangularDomain;

use super::sampling::sobol_unit_hypercube_points;
use super::*;

#[test]
fn test_collocation_sampler() {
    let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0);
    let sampler = CollocationSampler::new(Box::new(domain), SamplingStrategy::Uniform, Some(42));

    let interior = sampler.sample_interior(100);
    let boundary = sampler.sample_boundary(50);

    assert_eq!(interior.shape(), &[100, 2]);
    assert_eq!(boundary.shape(), &[50, 2]);
}

#[test]
fn test_interface_condition_debug() {
    let ic = InterfaceCondition::ElasticContinuity;
    assert_eq!(format!("{:?}", ic), "ElasticContinuity");

    let ic2 = InterfaceCondition::AcousticElastic {
        fluid_density: 1000.0,
    };
    assert!(format!("{:?}", ic2).contains("1000"));
}

#[test]
fn test_multi_region_locate() {
    let region1: Box<dyn crate::domain::geometry::GeometricDomain> =
        Box::new(RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0));
    let region2: Box<dyn crate::domain::geometry::GeometricDomain> =
        Box::new(RectangularDomain::new_2d(1.0, 2.0, 0.0, 1.0));

    let multi = MultiRegionDomain::new(
        vec![region1, region2],
        vec![0, 1],
        vec![InterfaceCondition::ElasticContinuity],
    );

    let loc = multi.locate_point(&[0.5, 0.5], 1e-6);
    assert!(loc.is_some());
    assert_eq!(loc.unwrap().0, 0);
}

#[test]
fn test_adaptive_refinement() {
    let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0);
    let sampler = CollocationSampler::new(Box::new(domain), SamplingStrategy::Uniform, Some(42));

    let initial = sampler.sample_interior(10);
    let mut adaptive = AdaptiveRefinement::new(sampler, initial.clone(), 0.1);

    let mut residuals = Array1::zeros(10);
    residuals[0] = 0.5;
    residuals[5] = 0.3;

    adaptive.update_residuals(residuals);
    let refined = adaptive.refine(2.0);

    assert!(refined.nrows() > 10);
}

#[test]
fn test_sobol_unit_hypercube_points() {
    let pts = sobol_unit_hypercube_points(8, 2, Some(0));
    assert_eq!(pts.len(), 8);
    for p in pts {
        assert_eq!(p.len(), 2);
        assert!(p[0] >= 0.0 && p[0] < 1.0);
        assert!(p[1] >= 0.0 && p[1] < 1.0);
    }
}

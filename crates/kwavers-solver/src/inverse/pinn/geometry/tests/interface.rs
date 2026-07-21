use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_grid::geometry::{GeometricDomain, PointLocation, RectangularDomain, SphericalDomain};
use tyche_core::Seed;

use super::super::{MultiRegionDomain, MultiRegionError, PinnGeometryInterfaceCondition};

fn rectangle_2d() -> Box<dyn GeometricDomain> {
    Box::new(RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0).expect("valid rectangle"))
}

#[test]
fn interface_condition_debug_identifies_the_variant() {
    let condition = PinnGeometryInterfaceCondition::ElasticContinuity;
    assert_eq!(format!("{condition:?}"), "ElasticContinuity");

    let acoustic = PinnGeometryInterfaceCondition::AcousticElastic {
        fluid_density: DENSITY_WATER_NOMINAL,
    };
    assert!(format!("{acoustic:?}").contains("1000"));
}

#[test]
fn multi_region_location_returns_the_first_containing_region() {
    // Type erasure is intentional at this cold heterogeneous region boundary.
    let first: Box<dyn GeometricDomain> =
        Box::new(RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0).expect("valid first region"));
    let second: Box<dyn GeometricDomain> =
        Box::new(RectangularDomain::new_2d(1.0, 2.0, 0.0, 1.0).expect("valid second region"));
    let domain = MultiRegionDomain::new(
        vec![first, second],
        vec![0, 1],
        vec![PinnGeometryInterfaceCondition::ElasticContinuity],
    )
    .expect("valid adjacent regions");

    let location = domain
        .locate_point(&[0.5, 0.5], 16.0 * f64::EPSILON)
        .expect("point belongs to first region");
    assert_eq!(location.0, 0);
}

#[test]
fn multi_region_construction_rejects_each_invalid_topology() {
    assert!(matches!(
        MultiRegionDomain::new(Vec::new(), Vec::new(), Vec::new()),
        Err(MultiRegionError::Empty)
    ));
    assert!(matches!(
        MultiRegionDomain::new(vec![rectangle_2d()], Vec::new(), Vec::new()),
        Err(MultiRegionError::MaterialCount {
            regions: 1,
            materials: 0
        })
    ));
    assert!(matches!(
        MultiRegionDomain::new(vec![rectangle_2d(), rectangle_2d()], vec![0, 1], Vec::new()),
        Err(MultiRegionError::InterfaceCount {
            regions: 2,
            expected: 1,
            actual: 0
        })
    ));

    let interval: Box<dyn GeometricDomain> =
        Box::new(RectangularDomain::new_1d(0.0, 1.0).expect("valid interval"));
    assert!(matches!(
        MultiRegionDomain::new(
            vec![rectangle_2d(), interval],
            vec![0, 1],
            vec![PinnGeometryInterfaceCondition::ElasticContinuity]
        ),
        Err(MultiRegionError::DimensionMismatch {
            region: 1,
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn interface_sampling_applies_the_quota_to_each_pair() {
    let domain = MultiRegionDomain::new(
        vec![rectangle_2d(), rectangle_2d(), rectangle_2d()],
        vec![0, 1, 2],
        vec![
            PinnGeometryInterfaceCondition::ElasticContinuity,
            PinnGeometryInterfaceCondition::WeldedContact,
        ],
    )
    .expect("valid three-region domain");

    let points = domain
        .sample_interface_points(17, Seed::new(29))
        .expect("addressable interface matrix");
    let replay = domain
        .sample_interface_points(17, Seed::new(29))
        .expect("addressable replay");

    assert_eq!(points, replay);
    assert_eq!(points.shape(), [34, 2]);
    let reference =
        RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0).expect("valid reference rectangle");
    for row in 0..points.shape()[0] {
        let point = [points[[row, 0]], points[[row, 1]]];
        assert_eq!(
            reference.classify_point(&point, 64.0 * f64::EPSILON),
            PointLocation::Boundary
        );
    }
}

#[test]
fn interface_sampling_scales_roundoff_with_domain_extent() {
    let radius = 1.0e100;
    let ball = || -> Box<dyn GeometricDomain> {
        Box::new(SphericalDomain::new_3d(0.0, 0.0, 0.0, radius).expect("valid large ball"))
    };
    let domain = MultiRegionDomain::new(
        vec![ball(), ball()],
        vec![0, 1],
        vec![PinnGeometryInterfaceCondition::ElasticContinuity],
    )
    .expect("valid identical regions");

    let points = domain
        .sample_interface_points(1, Seed::new(5))
        .expect("addressable interface matrix");
    assert_eq!(points.shape(), [1, 3]);
}

#[test]
fn single_region_sampling_ignores_an_unused_extreme_quota() {
    let domain = MultiRegionDomain::new(vec![rectangle_2d()], vec![0], Vec::new())
        .expect("valid single-region domain");
    let points = domain
        .sample_interface_points(usize::MAX, Seed::new(29))
        .expect("a domain without interfaces needs no candidate storage");
    assert_eq!(points.shape(), [0, 2]);
}

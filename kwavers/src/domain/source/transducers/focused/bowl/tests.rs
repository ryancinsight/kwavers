use super::*;
use crate::core::error::KwaversError;

#[test]
fn bowl_layout_uses_canonical_equal_area_cap() {
    let element_size = 0.004_f64;
    let config = BowlConfig {
        radius_of_curvature: 0.08,
        diameter: 0.04,
        center: [0.01, -0.02, 0.003],
        focus: [0.01, -0.02, 0.083],
        frequency: 1.2e6,
        amplitude: 7.5e5,
        element_size: Some(element_size),
        ..Default::default()
    };

    let bowl = BowlTransducer::new(config.clone()).unwrap();
    let theta_max = (config.diameter * 0.5 / config.radius_of_curvature).asin();
    let expected_area = spherical_cap_area(config.radius_of_curvature, theta_max);
    let expected_count = (expected_area / element_size.powi(2)).ceil() as usize;

    assert_eq!(bowl.element_positions.len(), expected_count);
    assert_eq!(bowl.element_normals.len(), expected_count);
    assert_eq!(bowl.element_areas.len(), expected_count);

    let summed_area: f64 = bowl.element_areas.iter().sum();
    assert!((summed_area - expected_area).abs() < 1.0e-14);

    let expected_weight = expected_area / expected_count as f64;
    for ((position, normal), area) in bowl
        .element_positions
        .iter()
        .zip(&bowl.element_normals)
        .zip(&bowl.element_areas)
    {
        let distance_to_curvature_center = norm3(sub3(config.focus, *position));
        let normal_norm = norm3(*normal);
        let reconstructed_focus = add3(*position, scale3(*normal, config.radius_of_curvature));

        assert!((distance_to_curvature_center - config.radius_of_curvature).abs() < 1.0e-12);
        assert!((normal_norm - 1.0).abs() < 1.0e-12);
        assert!((reconstructed_focus[0] - config.focus[0]).abs() < 1.0e-12);
        assert!((reconstructed_focus[1] - config.focus[1]).abs() < 1.0e-12);
        assert!((reconstructed_focus[2] - config.focus[2]).abs() < 1.0e-12);
        assert!((*area - expected_weight).abs() < 1.0e-14);
    }
}

#[test]
fn bowl_element_size_controls_layout_count() {
    let coarse = BowlTransducer::new(BowlConfig {
        radius_of_curvature: 0.07,
        diameter: 0.035,
        element_size: Some(0.006),
        ..Default::default()
    })
    .unwrap();
    let fine = BowlTransducer::new(BowlConfig {
        radius_of_curvature: 0.07,
        diameter: 0.035,
        element_size: Some(0.003),
        ..Default::default()
    })
    .unwrap();

    let coarse_area: f64 = coarse.element_areas.iter().sum();
    let fine_area: f64 = fine.element_areas.iter().sum();

    assert!(fine.element_positions.len() > coarse.element_positions.len());
    assert!((fine_area - coarse_area).abs() < 1.0e-14);
}

#[test]
fn bowl_explicit_element_count_controls_layout_count() {
    let config = BowlConfig {
        radius_of_curvature: 0.16,
        diameter: 0.16,
        center: [0.0, 0.0, -0.16],
        focus: [0.0, 0.0, 0.0],
        frequency: 650.0e3,
        amplitude: 1.0e6,
        ..Default::default()
    };

    let bowl = BowlTransducer::with_element_count(config.clone(), 1024).unwrap();
    let theta_max = (config.diameter * 0.5 / config.radius_of_curvature).asin();
    let expected_area = spherical_cap_area(config.radius_of_curvature, theta_max);
    let summed_area: f64 = bowl.element_areas().iter().sum();

    assert_eq!(bowl.element_count(), 1024);
    assert_eq!(bowl.element_positions().len(), 1024);
    assert_eq!(bowl.element_normals().len(), 1024);
    assert_eq!(bowl.element_areas().len(), 1024);
    assert!((summed_area - expected_area).abs() < 1.0e-14);
}

#[test]
fn explicit_element_count_owns_discretization_option() {
    let mut config = BowlConfig::default();
    config.element_size = Some(0.0);

    let bowl = BowlTransducer::with_element_count(config, 32).unwrap();

    assert_eq!(bowl.element_count(), 32);
}

#[test]
fn bowl_rejects_nonfinite_or_degenerate_domains() {
    let mut zero_radius = BowlConfig::default();
    zero_radius.radius_of_curvature = 0.0;
    assert_validation_error(zero_radius);

    let mut zero_diameter = BowlConfig::default();
    zero_diameter.diameter = 0.0;
    assert_validation_error(zero_diameter);

    let mut excessive_diameter = BowlConfig::default();
    excessive_diameter.diameter = 2.0 * excessive_diameter.radius_of_curvature + 1.0e-6;
    assert_validation_error(excessive_diameter);

    let mut zero_frequency = BowlConfig::default();
    zero_frequency.frequency = 0.0;
    assert_validation_error(zero_frequency);

    let mut zero_element = BowlConfig::default();
    zero_element.element_size = Some(0.0);
    assert_validation_error(zero_element);

    let mut nonfinite_center = BowlConfig::default();
    nonfinite_center.center = [f64::NAN, 0.0, 0.0];
    assert_validation_error(nonfinite_center);

    let mut nonfinite_focus = BowlConfig::default();
    nonfinite_focus.focus = [0.0, f64::INFINITY, 0.064];
    assert_validation_error(nonfinite_focus);

    let mut degenerate_axis = BowlConfig::default();
    degenerate_axis.focus = degenerate_axis.center;
    assert_validation_error(degenerate_axis);

    let zero_count = BowlTransducer::with_element_count(BowlConfig::default(), 0);
    assert!(matches!(
        zero_count.unwrap_err(),
        KwaversError::Validation(_)
    ));
}

fn assert_validation_error(config: BowlConfig) {
    let error = BowlTransducer::new(config).unwrap_err();
    assert!(matches!(error, KwaversError::Validation(_)));
}

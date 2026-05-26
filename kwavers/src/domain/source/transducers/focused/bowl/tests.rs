use super::*;
use crate::core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use crate::core::error::KwaversError;
use crate::domain::grid::Grid;
use std::f64::consts::PI;

#[test]
fn bowl_layout_uses_canonical_equal_area_cap() {
    let element_size = 0.004_f64;
    let config = BowlConfig {
        radius_of_curvature: 0.08,
        diameter: 0.04,
        center: [0.01, -0.02, 0.003],
        focus: [0.01, -0.02, 0.083],
        frequency: 1.2 * MHZ_TO_HZ,
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
        amplitude: MPA_TO_PA,
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
fn bowl_source_uses_axis_specific_grid_spacing_and_origin() {
    let config = BowlConfig {
        radius_of_curvature: 0.08,
        diameter: 0.04,
        center: [0.0, 0.0, -0.08],
        focus: [0.0, 0.0, 0.0],
        frequency: 1.25 * MHZ_TO_HZ,
        amplitude: 4.0e5,
        phase: 0.31,
        apply_directivity: false,
        ..Default::default()
    };
    let bowl = BowlTransducer::with_element_count(config.clone(), 1).unwrap();
    let mut grid = Grid::new(3, 4, 5, 0.001, 0.002, 0.003).unwrap();
    grid.origin = [0.011, -0.017, 0.023];

    let time = 0.37e-6;
    let source = bowl.generate_source(&grid, time).unwrap();
    let omega = 2.0 * PI * config.frequency;
    let focus_delays = bowl.calculate_focus_delays();
    let element_position = bowl.element_positions()[0];
    let element_area = bowl.element_areas()[0];
    let phase = omega.mul_add(time - focus_delays[0], config.phase);

    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            for iz in 0..grid.nz {
                let point = [
                    (ix as f64).mul_add(grid.dx, grid.origin[0]),
                    (iy as f64).mul_add(grid.dy, grid.origin[1]),
                    (iz as f64).mul_add(grid.dz, grid.origin[2]),
                ];
                let distance = (point[2] - element_position[2])
                    .mul_add(
                        point[2] - element_position[2],
                        (point[1] - element_position[1]).mul_add(
                            point[1] - element_position[1],
                            (point[0] - element_position[0]).powi(2),
                        ),
                    )
                    .sqrt();
                let expected = if distance > 0.0 {
                    config.amplitude * element_area * phase.sin() / (4.0 * PI * distance)
                } else {
                    0.0
                };

                assert!(
                    (source[[ix, iy, iz]] - expected).abs() < expected.abs().max(1.0) * 1.0e-12,
                    "source[{ix},{iy},{iz}] = {}, expected {expected}",
                    source[[ix, iy, iz]]
                );
            }
        }
    }
}

#[test]
fn hemispherical_preset_generates_source_domain_fixed_count_layout() {
    let config = BowlConfig::hemispherical([0.0, 0.0, 0.16], [0.0, 0.0, 0.0], 650.0e3, MPA_TO_PA);

    assert!((config.radius_of_curvature - 0.16).abs() < 1.0e-12);
    assert!((config.diameter - 0.32).abs() < 1.0e-12);

    let bowl = BowlTransducer::with_element_count(config, 1024).unwrap();
    let summed_area: f64 = bowl.element_areas().iter().sum();
    let expected_hemisphere_area = 2.0 * std::f64::consts::PI * 0.16_f64.powi(2);

    assert_eq!(bowl.element_count(), 1024);
    assert!((summed_area - expected_hemisphere_area).abs() < 1.0e-14);
    for position in bowl.element_positions() {
        assert!(position[2] >= -1.0e-12);
    }
}

#[test]
fn axis_reference_preset_preserves_focus_axis_and_explicit_radius() {
    let axis_reference = [0.04, -0.02, 0.03];
    let focus = [-0.02, 0.01, 0.05];
    let radius = 0.135;
    let theta_min = 0.20_f64;
    let theta_max = 0.90_f64;
    let axis_unit = normalize3(sub3(focus, axis_reference)).unwrap();
    let expected_vertex = sub3(focus, scale3(axis_unit, radius));

    // Aperture chord at theta_max = 0.90 rad: 2 R sin(theta_max).
    let aperture_diameter = 2.0 * radius * theta_max.sin();
    let config = BowlConfig::from_axis_reference_focus(
        axis_reference,
        focus,
        radius,
        aperture_diameter,
        750.0e3,
        2.5e5,
    )
    .unwrap();

    assert!((config.radius_of_curvature - radius).abs() < 1.0e-14);
    assert_eq!(config.focus, focus);
    assert_eq!(config.frequency, 750.0e3);
    assert_eq!(config.amplitude, 2.5e5);
    for (actual, expected) in config.center.iter().zip(expected_vertex) {
        assert!((*actual - expected).abs() < 1.0e-14);
    }

    let bowl = BowlTransducer::with_polar_bounds(config, theta_min, theta_max, 96).unwrap();
    let summed_area: f64 = bowl.element_areas().iter().sum();
    let expected_area =
        2.0 * std::f64::consts::PI * radius.powi(2) * (theta_min.cos() - theta_max.cos());

    assert!((summed_area - expected_area).abs() < 1.0e-14);
    for position in bowl.element_positions() {
        let focus_to_element = sub3(focus, *position);
        let distance_to_focus = norm3(focus_to_element);
        let axis_projection = axis_unit[0].mul_add(
            focus_to_element[0],
            axis_unit[1].mul_add(focus_to_element[1], axis_unit[2] * focus_to_element[2]),
        ) / radius;

        assert!((distance_to_focus - radius).abs() < 1.0e-12);
        assert!(axis_projection <= theta_min.cos() + 1.0e-12);
        assert!(axis_projection >= theta_max.cos() - 1.0e-12);
    }
}

#[test]
fn focus_axis_preset_preserves_axis_radius_and_explicit_aperture() {
    let focus = [0.02, -0.01, 0.03];
    let axis = [0.0, 0.0, -2.0];
    let radius = 0.150;
    let aperture_diameter = 2.0 * radius;
    let theta_min = 0.22_f64;
    let theta_max = 1.18_f64;

    let config =
        BowlConfig::from_focus_axis(focus, axis, radius, aperture_diameter, 650.0e3, 1.5e5)
            .unwrap();

    assert_eq!(config.focus, focus);
    assert_eq!(config.center, [focus[0], focus[1], focus[2] + radius]);
    assert!((config.radius_of_curvature - radius).abs() < 1.0e-14);
    assert!((config.diameter - aperture_diameter).abs() < 1.0e-14);

    let bowl = BowlTransducer::with_polar_bounds(config, theta_min, theta_max, 128).unwrap();
    let expected_area =
        2.0 * std::f64::consts::PI * radius.powi(2) * (theta_min.cos() - theta_max.cos());
    let summed_area: f64 = bowl.element_areas().iter().sum();

    assert_eq!(bowl.element_count(), 128);
    assert!((summed_area - expected_area).abs() < 1.0e-14);
    for position in bowl.element_positions() {
        let focus_to_element = sub3(*position, focus);
        let distance_to_focus = norm3(focus_to_element);
        let normalized_z = (position[2] - focus[2]) / radius;

        assert!((distance_to_focus - radius).abs() < 1.0e-12);
        assert!(normalized_z <= theta_min.cos() + 1.0e-12);
        assert!(normalized_z >= theta_max.cos() - 1.0e-12);
    }
}

#[test]
fn axis_reference_preset_rejects_degenerate_axis() {
    let error = BowlConfig::from_axis_reference_focus(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.1,
        0.15,
        500.0e3,
        1.0e5,
    )
    .unwrap_err();

    assert!(matches!(error, KwaversError::Validation(_)));
}

#[test]
fn focus_axis_preset_rejects_degenerate_axis() {
    let error =
        BowlConfig::from_focus_axis([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.1, 0.15, 500.0e3, 1.0e5)
            .unwrap_err();

    assert!(matches!(error, KwaversError::Validation(_)));
}

#[test]
fn axis_reference_preset_rejects_excessive_aperture_chord() {
    let error = BowlConfig::from_axis_reference_focus(
        [0.0, 0.0, 0.04],
        [0.0, 0.0, 0.0],
        0.1,
        0.21,
        500.0e3,
        1.0e5,
    )
    .unwrap_err();

    assert!(matches!(error, KwaversError::Validation(_)));
}

#[test]
fn bowl_polar_span_supports_major_cap_beyond_hemisphere() {
    let config =
        BowlConfig::from_vertex_focus([0.0, 0.0, 0.16], [0.0, 0.0, 0.0], 0.32, 650.0e3, MPA_TO_PA);
    let theta_max = 0.58 * std::f64::consts::PI;

    let bowl = BowlTransducer::with_polar_span(config, theta_max, 128).unwrap();
    let summed_area: f64 = bowl.element_areas().iter().sum();
    let expected_area = 2.0 * std::f64::consts::PI * 0.16_f64.powi(2) * (1.0 - theta_max.cos());
    let min_z = bowl
        .element_positions()
        .iter()
        .map(|position| position[2])
        .fold(f64::INFINITY, f64::min);
    let max_z = bowl
        .element_positions()
        .iter()
        .map(|position| position[2])
        .fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(bowl.element_count(), 128);
    assert!((summed_area - expected_area).abs() < 1.0e-14);
    assert!(min_z < -0.030);
    assert!(max_z > 0.150);
}

#[test]
fn bowl_polar_span_rejects_invalid_angular_domain() {
    let config = BowlConfig::hemispherical([0.0, 0.0, 0.16], [0.0, 0.0, 0.0], 650.0e3, MPA_TO_PA);

    assert!(matches!(
        BowlTransducer::with_polar_span(config.clone(), 0.0, 128).unwrap_err(),
        KwaversError::Validation(_)
    ));
    assert!(matches!(
        BowlTransducer::with_polar_span(config, std::f64::consts::PI + 1.0e-12, 128).unwrap_err(),
        KwaversError::Validation(_)
    ));
}

#[test]
fn bowl_polar_bounds_support_annular_cutout_area() {
    let config =
        BowlConfig::from_vertex_focus([0.0, 0.0, 0.10], [0.0, 0.0, 0.0], 0.20, 500.0e3, 1.0e5);
    let theta_min = 0.20;
    let theta_max = 0.90;

    let bowl = BowlTransducer::with_polar_bounds(config, theta_min, theta_max, 96).unwrap();
    let summed_area: f64 = bowl.element_areas().iter().sum();
    let expected_area =
        2.0 * std::f64::consts::PI * 0.10_f64.powi(2) * (theta_min.cos() - theta_max.cos());
    let max_z = bowl
        .element_positions()
        .iter()
        .map(|position| position[2])
        .fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(bowl.element_count(), 96);
    assert!((summed_area - expected_area).abs() < 1.0e-14);
    assert!(max_z < 0.10 * theta_min.cos());
}

#[test]
fn bowl_axis_projection_bounds_support_major_cap_area() {
    let config =
        BowlConfig::from_vertex_focus([0.0, 0.0, 0.16], [0.0, 0.0, 0.0], 0.32, 650.0e3, MPA_TO_PA);
    let bowl = BowlTransducer::with_axis_projection_bounds(config, -0.28, 0.98, 128).unwrap();
    let summed_area: f64 = bowl.element_areas().iter().sum();
    let expected_area = 2.0 * std::f64::consts::PI * 0.16_f64.powi(2) * (0.98 - -0.28);
    let min_projection = bowl
        .element_positions()
        .iter()
        .map(|position| position[2] / 0.16)
        .fold(f64::INFINITY, f64::min);
    let max_projection = bowl
        .element_positions()
        .iter()
        .map(|position| position[2] / 0.16)
        .fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(bowl.element_count(), 128);
    assert!((summed_area - expected_area).abs() < 1.0e-14);
    assert!((min_projection + 0.28).abs() < 0.02);
    assert!((max_projection - 0.98).abs() < 0.02);
}

#[test]
fn bowl_axis_projection_bounds_reject_invalid_domain() {
    assert!(matches!(
        BowlAngularBounds::from_axis_projection_bounds(-1.1, 0.9).unwrap_err(),
        KwaversError::Validation(_)
    ));
    assert!(matches!(
        BowlAngularBounds::from_axis_projection_bounds(0.5, 0.5).unwrap_err(),
        KwaversError::Validation(_)
    ));
    assert!(matches!(
        BowlAngularBounds::from_axis_projection_bounds(-0.2, 1.1).unwrap_err(),
        KwaversError::Validation(_)
    ));
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

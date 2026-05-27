use super::SourceFactory;
use crate::domain::grid::Grid;
use crate::domain::source::{DomainSourceParameters, FocusedBowlAperture, SourceModel};

#[test]
fn focused_source_factory_honors_configured_element_count() {
    let mut grid = Grid::new(24, 24, 24, 0.004, 0.004, 0.004).unwrap();
    grid.origin = [-0.048, -0.048, -0.008];
    let element_count = 17;
    let config = DomainSourceParameters {
        model: SourceModel::Focused,
        position: [0.0, 0.0, 0.0],
        focus: Some([0.0, 0.0, 0.08]),
        radius: 0.02,
        frequency: 650.0e3,
        num_elements: Some(element_count),
        ..Default::default()
    };

    let source = SourceFactory::create_source(&config, &grid).unwrap();

    assert_eq!(source.positions().len(), element_count);
    assert_eq!(source.focal_point(), Some((0.0, 0.0, 0.08)));
}

#[test]
fn focused_source_factory_routes_base_geometry_through_bowl_constructor() {
    let mut grid = Grid::new(32, 32, 32, 0.004, 0.004, 0.004).unwrap();
    grid.origin = [-0.064, -0.064, -0.016];
    let position = [0.01, -0.015, 0.02];
    let focus = [0.04, 0.005, 0.09];
    let element_count = 13;
    let config = DomainSourceParameters {
        model: SourceModel::Focused,
        position,
        focus: Some(focus),
        radius: 0.018,
        frequency: 650.0e3,
        num_elements: Some(element_count),
        ..Default::default()
    };
    let curvature_radius = ((focus[0] - position[0]).powi(2)
        + (focus[1] - position[1]).powi(2)
        + (focus[2] - position[2]).powi(2))
    .sqrt();

    let source = SourceFactory::create_source(&config, &grid).unwrap();

    assert_eq!(source.positions().len(), element_count);
    assert_eq!(source.focal_point(), Some((focus[0], focus[1], focus[2])));
    for element in source.positions() {
        let distance_to_focus = ((element.0 - focus[0]).powi(2)
            + (element.1 - focus[1]).powi(2)
            + (element.2 - focus[2]).powi(2))
        .sqrt();
        assert!((distance_to_focus - curvature_radius).abs() < 1.0e-12);
    }
}

#[test]
fn focused_source_factory_accepts_axis_projection_aperture() {
    let mut grid = Grid::new(40, 40, 28, 0.01, 0.01, 0.01).unwrap();
    grid.origin = [-0.20, -0.20, -0.08];
    let element_count = 19;
    let config = DomainSourceParameters {
        model: SourceModel::Focused,
        position: [0.0, 0.0, 0.16],
        focus: Some([0.0, 0.0, 0.0]),
        radius: 0.16,
        frequency: 650.0e3,
        num_elements: Some(element_count),
        focused_bowl_aperture: FocusedBowlAperture::AxisProjectionBounds {
            axis_projection_min: -0.20,
            axis_projection_max: 0.95,
        },
        ..Default::default()
    };

    let source = SourceFactory::create_source(&config, &grid).unwrap();
    let positions = source.positions();
    let min_projection = positions
        .iter()
        .map(|position| position.2 / 0.16)
        .fold(f64::INFINITY, f64::min);
    let max_projection = positions
        .iter()
        .map(|position| position.2 / 0.16)
        .fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(positions.len(), element_count);
    assert!((min_projection + 0.20).abs() < 0.08);
    assert!((max_projection - 0.95).abs() < 0.08);
}

#[test]
fn focused_source_factory_accepts_hemisphere_aperture() {
    let mut grid = Grid::new(48, 48, 32, 0.01, 0.01, 0.01).unwrap();
    grid.origin = [-0.24, -0.24, -0.04];
    let element_count = 31;
    let radius = 0.16_f64;
    let config = DomainSourceParameters {
        model: SourceModel::Focused,
        position: [0.0, 0.0, radius],
        focus: Some([0.0, 0.0, 0.0]),
        radius: 0.01,
        frequency: 650.0e3,
        num_elements: Some(element_count),
        focused_bowl_aperture: FocusedBowlAperture::Hemisphere,
        ..Default::default()
    };

    let source = SourceFactory::create_source(&config, &grid).unwrap();
    let positions = source.positions();
    let min_projection = positions
        .iter()
        .map(|position| position.2 / radius)
        .fold(f64::INFINITY, f64::min);
    let max_projection = positions
        .iter()
        .map(|position| position.2 / radius)
        .fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(positions.len(), element_count);
    assert!(min_projection >= -1.0e-12);
    assert!(max_projection <= 1.0 + 1.0e-12);
}

#[test]
fn angular_focused_source_factory_requires_element_count() {
    let grid = Grid::new(8, 8, 8, 0.01, 0.01, 0.01).unwrap();
    let config = DomainSourceParameters {
        model: SourceModel::Focused,
        position: [0.0, 0.0, 0.08],
        focus: Some([0.0, 0.0, 0.0]),
        radius: 0.08,
        frequency: 650.0e3,
        focused_bowl_aperture: FocusedBowlAperture::PolarSpan { theta_max_rad: 1.0 },
        ..Default::default()
    };

    let error = SourceFactory::create_source(&config, &grid).unwrap_err();
    assert!(
        format!("{error:?}").contains("num_elements"),
        "expected num_elements validation, got {error:?}"
    );
}

#[test]
fn focused_source_factory_accepts_axis_reference_explicit_radius_aperture() {
    let mut grid = Grid::new(64, 64, 32, 0.008, 0.008, 0.008).unwrap();
    grid.origin = [-0.256, -0.256, -0.032];
    let axis_reference = [0.0, 0.0, 0.04];
    let focus = [0.0, 0.0, 0.0];
    let radius = 0.16_f64;
    let theta_min = 0.20_f64;
    let theta_max = 0.90_f64;
    let element_count = 23;
    let config = DomainSourceParameters {
        model: SourceModel::Focused,
        position: axis_reference,
        focus: Some(focus),
        radius: 0.01,
        frequency: 650.0e3,
        num_elements: Some(element_count),
        focused_bowl_aperture: FocusedBowlAperture::AxisReferencePolarBounds {
            radius_of_curvature_m: radius,
            theta_min_rad: theta_min,
            theta_max_rad: theta_max,
        },
        ..Default::default()
    };

    let source = SourceFactory::create_source(&config, &grid).unwrap();
    let axis_norm = ((focus[0] - axis_reference[0]).powi(2)
        + (focus[1] - axis_reference[1]).powi(2)
        + (focus[2] - axis_reference[2]).powi(2))
    .sqrt();
    let axis_unit = [
        (focus[0] - axis_reference[0]) / axis_norm,
        (focus[1] - axis_reference[1]) / axis_norm,
        (focus[2] - axis_reference[2]) / axis_norm,
    ];

    assert_eq!(source.positions().len(), element_count);
    assert_eq!(source.focal_point(), Some((focus[0], focus[1], focus[2])));
    for position in source.positions() {
        let vector_focus_to_element = [
            focus[0] - position.0,
            focus[1] - position.1,
            focus[2] - position.2,
        ];
        let distance = (vector_focus_to_element[0].powi(2)
            + vector_focus_to_element[1].powi(2)
            + vector_focus_to_element[2].powi(2))
        .sqrt();
        let axis_projection = axis_unit[0].mul_add(
            vector_focus_to_element[0],
            axis_unit[1].mul_add(
                vector_focus_to_element[1],
                axis_unit[2] * vector_focus_to_element[2],
            ),
        ) / radius;

        assert!((distance - radius).abs() < 1.0e-12);
        assert!(axis_projection <= theta_min.cos() + 1.0e-12);
        assert!(axis_projection >= theta_max.cos() - 1.0e-12);
    }
}

#[test]
fn focused_source_factory_accepts_axis_reference_hemisphere_aperture() {
    let mut grid = Grid::new(64, 64, 40, 0.008, 0.008, 0.008).unwrap();
    grid.origin = [-0.256, -0.256, -0.064];
    let axis_reference = [0.0, 0.0, 0.04];
    let focus = [0.0, 0.0, 0.0];
    let radius = 0.16_f64;
    let element_count = 29;
    let config = DomainSourceParameters {
        model: SourceModel::Focused,
        position: axis_reference,
        focus: Some(focus),
        radius: 0.01,
        frequency: 650.0e3,
        num_elements: Some(element_count),
        focused_bowl_aperture: FocusedBowlAperture::AxisReferenceHemisphere {
            radius_of_curvature_m: radius,
        },
        ..Default::default()
    };

    let source = SourceFactory::create_source(&config, &grid).unwrap();
    let axis_norm = ((focus[0] - axis_reference[0]).powi(2)
        + (focus[1] - axis_reference[1]).powi(2)
        + (focus[2] - axis_reference[2]).powi(2))
    .sqrt();
    let axis_unit = [
        (focus[0] - axis_reference[0]) / axis_norm,
        (focus[1] - axis_reference[1]) / axis_norm,
        (focus[2] - axis_reference[2]) / axis_norm,
    ];

    assert_eq!(source.positions().len(), element_count);
    assert_eq!(source.focal_point(), Some((focus[0], focus[1], focus[2])));
    for position in source.positions() {
        let focus_to_element = [
            focus[0] - position.0,
            focus[1] - position.1,
            focus[2] - position.2,
        ];
        let distance = (focus_to_element[0].powi(2)
            + focus_to_element[1].powi(2)
            + focus_to_element[2].powi(2))
        .sqrt();
        let axis_projection = axis_unit[0].mul_add(
            focus_to_element[0],
            axis_unit[1].mul_add(focus_to_element[1], axis_unit[2] * focus_to_element[2]),
        ) / radius;

        assert!((distance - radius).abs() < 1.0e-12);
        assert!(axis_projection >= -1.0e-12);
        assert!(axis_projection <= 1.0 + 1.0e-12);
    }
}

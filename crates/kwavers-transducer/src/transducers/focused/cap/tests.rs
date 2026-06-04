use super::{SphericalCapConfig, SphericalCapLayout};
use std::f64::consts::{FRAC_PI_2, PI};
use kwavers_core::constants::numerical::{TWO_PI};

#[test]
fn hemisphere_layout_places_elements_on_focused_sphere() {
    let radius = 0.11;
    let layout = SphericalCapLayout::new(SphericalCapConfig::hemisphere(
        64,
        radius,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ))
    .unwrap();

    assert_eq!(layout.elements().len(), 64);
    for element in layout.elements() {
        let p = element.position_m;
        let distance = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!((distance - radius).abs() < 1.0e-12);
        assert!(p[2] >= -1.0e-12 && p[2] <= radius + 1.0e-12);

        let n = element.normal_to_focus;
        let normal_norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((normal_norm - 1.0).abs() < 1.0e-12);
        assert!((p[0] + radius * n[0]).abs() < 1.0e-12);
        assert!((p[1] + radius * n[1]).abs() < 1.0e-12);
        assert!((p[2] + radius * n[2]).abs() < 1.0e-12);
    }
}

#[test]
fn cap_layout_preserves_equal_area_weights() {
    let radius = 0.075;
    let theta_min = 0.175;
    let theta_max = 0.960;
    let layout = SphericalCapLayout::new(SphericalCapConfig::focused_cap(
        128,
        radius,
        [0.01, -0.02, 0.03],
        [1.0, 2.0, 3.0],
        theta_min,
        theta_max,
    ))
    .unwrap();

    let expected_area = TWO_PI * radius * radius * (theta_min.cos() - theta_max.cos());
    let summed_area: f64 = layout
        .elements()
        .iter()
        .map(|element| element.area_weight_m2)
        .sum();
    assert!((summed_area - expected_area).abs() < 1.0e-14);
}

#[test]
fn vertex_focus_constructor_derives_axis() {
    let radius = 0.115;
    let vertex = [0.05, 0.0, 0.0];
    let focus = [-0.05, 0.0, 0.0];
    let layout = SphericalCapLayout::new(SphericalCapConfig::from_vertex_focus(
        32, radius, vertex, focus, 0.175, 0.960,
    ))
    .unwrap();

    for element in layout.elements() {
        let p = element.position_m;
        let distance =
            ((p[0] - focus[0]).powi(2) + (p[1] - focus[1]).powi(2) + (p[2] - focus[2]).powi(2))
                .sqrt();
        assert!((distance - radius).abs() < 1.0e-12);
        assert!(p[0] > focus[0], "cap must be on the vertex side of focus");
    }
}

#[test]
fn cap_layout_rejects_invalid_domains() {
    let valid = SphericalCapConfig::hemisphere(8, 0.1, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);

    assert!(SphericalCapLayout::new(SphericalCapConfig {
        element_count: 0,
        ..valid
    })
    .is_err());
    assert!(SphericalCapLayout::new(SphericalCapConfig {
        radius_m: 0.0,
        ..valid
    })
    .is_err());
    assert!(SphericalCapLayout::new(SphericalCapConfig {
        axis_vertex_to_focus: [0.0, 0.0, 0.0],
        ..valid
    })
    .is_err());
    assert!(SphericalCapLayout::new(SphericalCapConfig {
        theta_min_rad: FRAC_PI_2,
        theta_max_rad: FRAC_PI_2,
        ..valid
    })
    .is_err());
    assert!(SphericalCapLayout::new(SphericalCapConfig {
        theta_min_rad: 0.0,
        theta_max_rad: PI + 1.0e-12,
        ..valid
    })
    .is_err());
}

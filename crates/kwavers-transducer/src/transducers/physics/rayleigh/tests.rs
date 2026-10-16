use super::aperture::*;
use super::integral::*;
use super::path::*;
use super::quadrature::*;
use eunomia::Complex64;
use kwavers_core::error::{ConfigError, KwaversError};
use kwavers_math::j1;
use std::f64::consts::{PI, TAU};

fn piston(radius_m: f64) -> PlanarAperture {
    PlanarAperture::disk(
        [0.0; 3],
        [0.0, 0.0, 1.0],
        radius_m,
        Complex64::new(2.5e5, 0.0),
    )
    .unwrap()
}

#[test]
fn quadrature_integrates_disk_area_exactly() {
    let rule = gauss_legendre_unit(7).unwrap();
    let radius = 2.3e-3;
    let area = PI * radius * radius * rule.iter().map(|(_, weight)| weight).sum::<f64>();
    let expected = PI * radius * radius;
    assert!((area - expected).abs() <= 8.0 * f64::EPSILON * expected);
}

#[test]
fn annular_sector_area_matches_radial_angular_measure() {
    let shape = PlanarApertureShape::AnnularSector {
        inner_radius_m: 1.0e-3,
        outer_radius_m: 3.0e-3,
        start_angle_rad: 0.3,
        span_angle_rad: std::f64::consts::FRAC_PI_2,
    };
    let expected = std::f64::consts::FRAC_PI_2 * (9.0e-6 - 1.0e-6) / 2.0;
    assert!((shape.area_m2() - expected).abs() <= 4.0 * f64::EPSILON * expected);
}

#[test]
fn independently_driven_sectors_superpose_to_complete_annulus_on_axis() {
    let pressure = Complex64::from_polar(1.7e5, 0.4);
    let aperture = |start_angle_rad, span_angle_rad| {
        PlanarAperture::oriented(
            [0.0; 3],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            PlanarApertureShape::AnnularSector {
                inner_radius_m: 0.4e-3,
                outer_radius_m: 1.2e-3,
                start_angle_rad,
                span_angle_rad,
            },
            pressure,
        )
        .unwrap()
    };
    let complete = aperture(0.0, TAU);
    let sectors: Vec<_> = (0..4)
        .map(|sector| {
            aperture(
                sector as f64 * std::f64::consts::FRAC_PI_2,
                std::f64::consts::FRAC_PI_2,
            )
        })
        .collect();
    let spec = RayleighIntegralSpec::new(TAU / 0.7e-3, 0.0, 12, 32).unwrap();
    let point = [[0.0, 0.0, 20.0e-3]];
    let complete_pressure = rayleigh_pressure(&point, &[complete], &spec).unwrap()[0];
    let sector_pressure = rayleigh_pressure(&point, &sectors, &spec).unwrap()[0];
    assert!(
        (complete_pressure - sector_pressure).norm()
            <= 32.0 * f64::EPSILON * complete_pressure.norm(),
        "complete={complete_pressure:?}, sectors={sector_pressure:?}"
    );
}

#[test]
fn on_axis_pressure_matches_closed_form() {
    let piston = piston(1.2e-3);
    let k = TAU / 0.75e-3;
    let axial_range = 14.0e-3;
    let spec = RayleighIntegralSpec::new(k, 0.0, 24, 24).unwrap();
    let actual = rayleigh_pressure(&[[0.0, 0.0, axial_range]], &[piston], &spec).unwrap()[0];
    let rim_range = axial_range.hypot(piston.outer_radius_m());
    let expected = piston.surface_pressure_pa()
        * (Complex64::from_polar(1.0, k * axial_range) - Complex64::from_polar(1.0, k * rim_range));
    let relative = (actual - expected).norm() / expected.norm();
    assert!(relative <= 2.0e-12, "relative complex error {relative:e}");
}

#[test]
fn far_field_ratio_matches_circular_piston_directivity() {
    let piston = piston(0.8e-3);
    let k = TAU / 0.6e-3;
    let range = 0.4;
    let angle: f64 = 0.17;
    let points = [
        [0.0, 0.0, range],
        [range * angle.sin(), 0.0, range * angle.cos()],
    ];
    let pressure = rayleigh_pressure(
        &points,
        &[piston],
        &RayleighIntegralSpec::new(k, 0.0, 20, 96).unwrap(),
    )
    .unwrap();
    let argument = k * piston.outer_radius_m() * angle.sin();
    let expected_ratio = (2.0 * j1(argument) / argument).abs();
    let actual_ratio = pressure[1].norm() / pressure[0].norm();
    assert!(
        (actual_ratio - expected_ratio).abs() <= 2.0e-4,
        "ratio {actual_ratio:e}, far-field oracle {expected_ratio:e}"
    );
}

#[test]
fn rotation_preserves_complex_pressure() {
    let base = piston(0.7e-3);
    let rotated = PlanarAperture::disk(
        [0.0; 3],
        [1.0, 0.0, 0.0],
        base.outer_radius_m(),
        base.surface_pressure_pa(),
    )
    .unwrap();
    let spec = RayleighIntegralSpec::new(TAU / 0.8e-3, 3.0, 12, 48).unwrap();
    let along_z = rayleigh_pressure(&[[0.2e-3, 0.0, 30.0e-3]], &[base], &spec).unwrap()[0];
    let along_x = rayleigh_pressure(&[[30.0e-3, 0.0, 0.2e-3]], &[rotated], &spec).unwrap()[0];
    assert!((along_z - along_x).norm() <= 64.0 * f64::EPSILON * along_z.norm());
}

#[test]
fn rigid_baffle_suppresses_back_radiation() {
    let pressure = rayleigh_pressure(
        &[[0.0, 0.0, -0.01]],
        &[piston(0.5e-3)],
        &RayleighIntegralSpec::new(TAU / 1.0e-3, 0.0, 4, 12).unwrap(),
    )
    .unwrap();
    assert_eq!(pressure, vec![Complex64::new(0.0, 0.0)]);
}

#[test]
fn quadrature_work_is_bounded() {
    let error = RayleighIntegralSpec::new(1.0, 0.0, MAX_SURFACE_SAMPLES, 3)
        .expect_err("surface sample count exceeds the provider budget");
    assert!(error.to_string().contains("quadrature_surface_samples"));
}

#[test]
fn layered_path_integrates_each_segment_exactly() {
    let path = RayleighPropagationPath::layered(vec![
        RayleighLayer::new(2.0, 3.0, Some(0.25)).unwrap(),
        RayleighLayer::new(5.0, 7.0, None).unwrap(),
    ])
    .unwrap();
    let (short_phase, short_attenuation) = path.propagation_terms(0.1).unwrap();
    assert!((short_phase - 0.2).abs() <= f64::EPSILON * 0.2);
    assert!((short_attenuation - 0.3).abs() <= f64::EPSILON * 0.3);
    let (long_phase, long_attenuation) = path.propagation_terms(1.0).unwrap();
    assert!((long_phase - 4.25).abs() <= 2.0 * f64::EPSILON * 4.25);
    assert!((long_attenuation - 6.0).abs() <= 2.0 * f64::EPSILON * 6.0);
}

#[test]
fn layered_path_requires_one_final_half_space() {
    let finite = RayleighLayer::new(2.0, 0.0, Some(0.25)).unwrap();
    let half_space = RayleighLayer::new(3.0, 0.0, None).unwrap();
    let final_half_space = RayleighLayer::new(4.0, 0.0, None).unwrap();
    let finite_final = RayleighPropagationPath::layered(vec![finite])
        .expect_err("a finite final layer has no propagation half-space");
    let non_final_half_space = RayleighPropagationPath::layered(vec![half_space, final_half_space])
        .expect_err("a semi-infinite layer cannot precede another layer");
    for (error, value, constraint) in [
        (
            finite_final,
            "finite final layer",
            "final layer is semi-infinite",
        ),
        (
            non_final_half_space,
            "non-final semi-infinite layer",
            "only final layer is semi-infinite",
        ),
    ] {
        match error {
            KwaversError::Config(ConfigError::InvalidValue {
                parameter,
                value: actual_value,
                constraint: actual_constraint,
            }) => {
                assert_eq!(parameter, "layers");
                assert_eq!(actual_value, value);
                assert_eq!(actual_constraint, constraint);
            }
            other => panic!("expected layered-path configuration error, got {other:?}"),
        }
    }
}

#[test]
fn propagation_terms_reject_invalid_range() {
    let path = RayleighPropagationPath::homogeneous(1.0, 0.0).unwrap();
    for range_m in [-1.0, f64::NAN] {
        let error = path
            .propagation_terms(range_m)
            .expect_err("invalid propagation range must be rejected");
        match error {
            KwaversError::Config(ConfigError::InvalidValue {
                parameter,
                constraint,
                ..
            }) => {
                assert_eq!(parameter, "range_m");
                assert_eq!(constraint, "finite and >= 0");
            }
            other => panic!("expected invalid range configuration error, got {other:?}"),
        }
    }
}

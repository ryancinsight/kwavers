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
    let sectors: Vec<_> = (0..4_u8)
        .map(|sector| {
            aperture(
                f64::from(sector) * std::f64::consts::FRAC_PI_2,
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

fn rectangle(width_m: f64, height_m: f64) -> PlanarAperture {
    PlanarAperture::oriented(
        [0.0; 3],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        PlanarApertureShape::Rectangle { width_m, height_m },
        Complex64::new(1.0, 0.0),
    )
    .unwrap()
}

fn sinc(argument: f64) -> f64 {
    if argument == 0.0 {
        1.0
    } else {
        argument.sin() / argument
    }
}

#[test]
fn rectangle_area_and_enclosing_radius_are_exact() {
    let shape = PlanarApertureShape::Rectangle {
        width_m: 3.0e-3,
        height_m: 4.0e-3,
    };
    assert_eq!(shape.area_m2(), 3.0e-3 * 4.0e-3);
    assert_eq!(shape.outer_radius_m(), 2.5e-3);
}

#[test]
fn rectangle_surface_rule_integrates_area() {
    let (width, height) = (0.46e-3, 1.5e-3);
    let rule = surface_rule(
        PlanarApertureShape::Rectangle {
            width_m: width,
            height_m: height,
        },
        7,
        9,
    )
    .unwrap();
    assert_eq!(rule.len(), 63);
    let area: f64 = rule.iter().map(|(_, weight)| weight).sum();
    // Each unit-interval Gauss-Legendre weight set sums to 1 up to one rounding
    // per weight; 63 products and sums bound the error by 64 ε.
    assert!((area - width * height).abs() <= 64.0 * f64::EPSILON * width * height);
    assert!(rule
        .iter()
        .all(|([u, v], _)| u.abs() < 0.5 * width && v.abs() < 0.5 * height));
}

#[test]
fn rectangle_surface_rule_assigns_primary_order_to_width() {
    // One width node and three height nodes: the width coordinate is the
    // single centre node, the height coordinates are three distinct roots.
    let rule = surface_rule(
        PlanarApertureShape::Rectangle {
            width_m: 2.0e-3,
            height_m: 1.0e-3,
        },
        1,
        3,
    )
    .unwrap();
    assert_eq!(rule.len(), 3);
    assert!(rule.iter().all(|([u, _], _)| *u == 0.0));
    let mut heights: Vec<f64> = rule.iter().map(|([_, v], _)| *v).collect();
    heights.sort_by(f64::total_cmp);
    assert!(heights[0] < 0.0 && heights[1] == 0.0 && heights[2] > 0.0);
    assert_eq!(heights[0], -heights[2]);
}

#[test]
fn secondary_order_below_three_is_rejected_only_for_polar_shapes() {
    let k = TAU / 0.75e-3;
    let spec = RayleighIntegralSpec::new(k, 0.0, 8, 1).unwrap();
    let observation = [[0.0, 0.0, 0.05]];
    let rectangle_value =
        rayleigh_pressure(&observation, &[rectangle(1.0e-3, 1.0e-3)], &spec).unwrap()[0];
    assert!(rectangle_value.norm() > 0.0);
    let disk =
        PlanarAperture::disk([0.0; 3], [0.0, 0.0, 1.0], 1.0e-3, Complex64::new(1.0, 0.0)).unwrap();
    let error = rayleigh_pressure(&observation, &[disk], &spec).unwrap_err();
    assert!(matches!(
        error,
        KwaversError::Config(ConfigError::InvalidValue { ref parameter, .. }) if parameter == "secondary_order"
    ));
    assert!(RayleighIntegralSpec::new(k, 0.0, 8, 0).is_err_and(|error| matches!(
        error,
        KwaversError::Config(ConfigError::InvalidValue { ref parameter, .. }) if parameter == "secondary_order"
    )));
}

#[test]
fn rectangle_rejects_non_positive_dimensions() {
    let error = PlanarApertureGeometry::oriented(
        [0.0; 3],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        PlanarApertureShape::Rectangle {
            width_m: 1.0e-3,
            height_m: 0.0,
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        KwaversError::Config(ConfigError::InvalidValue { ref parameter, .. }) if parameter == "height_m"
    ));
}

#[test]
fn rectangle_far_field_matches_fourier_pattern() {
    // Fraunhofer limit of a uniform rectangle with half-sides a, b:
    // |p(θ, φ)| / |p(0)| = |sinc(k a sinθ cosφ) · sinc(k b sinθ sinφ)|.
    // At R = 10⁴ λ the neglected quadratic Fresnel phase k(x² + y²)/(2R) is
    // at most k(a² + b²)/(2R) on the aperture. It perturbs the off-axis value
    // and the on-axis normaliser, each by at most that phase (|e^{iφ} − 1| ≤ |φ|),
    // so the normalised pattern deviates by at most k(a² + b²)/R.
    let wavelength = 0.75e-3;
    let k = TAU / wavelength;
    let (half_width, half_height) = (5.0 * wavelength, 1.5 * wavelength);
    let aperture = rectangle(2.0 * half_width, 2.0 * half_height);
    let spec = RayleighIntegralSpec::new(k, 0.0, 40, 40).unwrap();
    let range = 1.0e4 * wavelength;
    let fresnel_bound = k * (half_width * half_width + half_height * half_height) / range;
    let on_axis = rayleigh_pressure(&[[0.0, 0.0, range]], &[aperture], &spec).unwrap()[0];
    for (sin_theta, phi) in [
        (0.05_f64, 0.0_f64),
        (0.15, 0.0),
        (0.35, 0.0),
        (0.15, PI / 2.0),
        (0.2, PI / 2.0),
    ] {
        let cos_theta = (1.0 - sin_theta * sin_theta).sqrt();
        let point = [
            range * sin_theta * phi.cos(),
            range * sin_theta * phi.sin(),
            range * cos_theta,
        ];
        let value = rayleigh_pressure(&[point], &[aperture], &spec).unwrap()[0];
        let expected = (sinc(k * half_width * sin_theta * phi.cos())
            * sinc(k * half_height * sin_theta * phi.sin()))
        .abs();
        let ratio = value.norm() / on_axis.norm();
        assert!(
            (ratio - expected).abs() <= fresnel_bound,
            "sinθ {sin_theta}, φ {phi}: ratio {ratio} against {expected}"
        );
    }
}

#[test]
fn rectangle_on_axis_far_field_is_area_weighted() {
    // Far on axis every surface element is in phase: |p| = k·A/(2πR), A = 4ab.
    let wavelength = 0.75e-3;
    let k = TAU / wavelength;
    let (half_width, half_height) = (4.0 * wavelength, 2.0 * wavelength);
    let aperture = rectangle(2.0 * half_width, 2.0 * half_height);
    let spec = RayleighIntegralSpec::new(k, 0.0, 40, 40).unwrap();
    let range = 1.0e4 * wavelength;
    let value = rayleigh_pressure(&[[0.0, 0.0, range]], &[aperture], &spec).unwrap()[0];
    let expected = 2.0 * k * half_width * half_height / (PI * range);
    // On axis the neglected Fresnel phase k(x² + y²)/(2R) is at most
    // k(a² + b²)/(2R), and |e^{iφ} − 1| ≤ |φ| bounds the relative deviation.
    let fresnel_bound = k * (half_width * half_width + half_height * half_height) / (2.0 * range);
    assert!(
        (value.norm() - expected).abs() <= fresnel_bound * expected,
        "|p| {} against {expected}",
        value.norm()
    );
}

#[test]
fn rectangle_halves_superpose_to_the_whole() {
    // Two abutting half-width rectangles carrying the same phasor are the
    // whole rectangle. The n-node Gauss-Legendre remainder on an edge of length
    // L scales as (kL)^2n (n!)^4 / ((2n)!)^3; the longer edge has kL = 12.2,
    // and with n = 24 the remainder is below 1e-30, so the sums differ only
    // by rounding over the 2 x 24 x 24
    // terms, each carrying a few ulps.
    let wavelength = 0.77e-3;
    let k = TAU / wavelength;
    let (width, height) = (0.9e-3, 1.5e-3);
    let half = |center_x: f64| {
        PlanarAperture::oriented(
            [center_x, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            PlanarApertureShape::Rectangle {
                width_m: 0.5 * width,
                height_m: height,
            },
            Complex64::new(0.3, -0.7),
        )
        .unwrap()
    };
    let whole = PlanarAperture::oriented(
        [0.0; 3],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        PlanarApertureShape::Rectangle {
            width_m: width,
            height_m: height,
        },
        Complex64::new(0.3, -0.7),
    )
    .unwrap();
    let spec = RayleighIntegralSpec::new(k, 16.1, 24, 24).unwrap();
    let points = [[0.4e-3, 0.2e-3, 10.0e-3], [3.0e-3, -1.0e-3, 40.0e-3]];
    let split =
        rayleigh_pressure(&points, &[half(-0.25 * width), half(0.25 * width)], &spec).unwrap();
    let joined = rayleigh_pressure(&points, &[whole], &spec).unwrap();
    for (a, b) in split.iter().zip(&joined) {
        let rounding_bound = 4.0 * (2.0 * 24.0 * 24.0) * f64::EPSILON;
        assert!(
            (a - b).norm() <= rounding_bound * b.norm(),
            "{a} against {b}"
        );
    }
}

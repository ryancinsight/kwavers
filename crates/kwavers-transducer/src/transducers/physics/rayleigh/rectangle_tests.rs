//! Rectangular apertures: surface rule, order validation and far-field oracles.

use super::aperture::*;
use super::integral::*;
use super::quadrature::*;
use eunomia::Complex64;
use kwavers_core::error::{ConfigError, KwaversError};
use std::f64::consts::{PI, TAU};

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
    // The rule's weights are w·wᵢ·h·hⱼ over the 1D Gauss-Legendre weights, so
    // the area is w·h·(Σwᵢ)(Σhⱼ) up to rounding: three products per weight, 62
    // additions, and for the reference three products plus the 6 and 8
    // additions of its 1D sums, 82 roundings of at most u = ε/2 each to first
    // order. That the 1D sums are 1 is the 1D rule's own test.
    let weight_sum = |order| -> f64 {
        gauss_legendre_unit(order)
            .unwrap()
            .iter()
            .map(|(_, weight)| weight)
            .sum()
    };
    let reference = width * height * weight_sum(7) * weight_sum(9);
    assert!((area - reference).abs() <= 41.0 * f64::EPSILON * reference);
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
    let (side, range) = (1.0e-3, 0.05);
    let spec = RayleighIntegralSpec::new(k, 0.0, 8, 1).unwrap();
    let observation = [[0.0, 0.0, range]];
    let one_node = rayleigh_pressure(&observation, &[rectangle(side, side)], &spec).unwrap()[0];
    // Eight Gauss-Legendre nodes across a phase excursion of at most
    // k(h/2)²/(2R) = 0.021 rad are exact far below the bound below.
    let reference_spec = RayleighIntegralSpec::new(k, 0.0, 8, 8).unwrap();
    let reference =
        rayleigh_pressure(&observation, &[rectangle(side, side)], &reference_spec).unwrap()[0];
    // One height node is the midpoint rule: per height line its error is
    // (h³/24)·g''(ξ), g(v) = e^{ikr}/r. With r′ ≤ h/(2R) and r″ ≤ 1/R,
    // |g''|·r₀ ≤ (k r′)² + k r″ + 2k r′²/R + 1/R² + 2r′²/R², relative to h·g(0)
    // times h²/24. Line phases agree to k(h/2)²/(2R), so the lines add with
    // cos(k h²/(4R)) to spare. The leading term (h²/24)·k/R is the curvature
    // phase, so the error is also at least that term less the others.
    let slope = side / (2.0 * range);
    let leading = side * side / 24.0 * (k / range);
    let rest = side * side / 24.0
        * ((k * slope).powi(2)
            + 2.0 * k * slope * slope / range
            + 1.0 / (range * range)
            + 2.0 * slope * slope / (range * range));
    let alignment = (k * side * side / (4.0 * range)).cos();
    let relative_error = (one_node - reference).norm() / reference.norm();
    assert!(
        relative_error <= (leading + rest) / alignment,
        "one height node: relative error {relative_error:.3e}"
    );
    assert!(
        relative_error >= leading - rest,
        "one height node must be the midpoint rule: relative error {relative_error:.3e}"
    );

    let disk =
        PlanarAperture::disk([0.0; 3], [0.0, 0.0, 1.0], 1.0e-3, Complex64::new(1.0, 0.0)).unwrap();
    let silent =
        PlanarAperture::disk([0.0; 3], [0.0, 0.0, 1.0], 1.0e-3, Complex64::new(0.0, 0.0)).unwrap();
    // A disk is rejected whether or not it radiates in this call.
    for aperture in [disk, silent] {
        let error = rayleigh_pressure(&observation, &[aperture], &spec).unwrap_err();
        assert!(matches!(
            error,
            KwaversError::Config(ConfigError::InvalidValue { ref parameter, .. }) if parameter == "secondary_order"
        ));
    }
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
    // and with n = 24 the remainder is below 1e-30, so the sums differ only by
    // rounding. Per term, with u = ε/2: the range carries 4u·R + 2u·L and the
    // phase k·R one more rounding, so the phase is off by at most 6·k·R_max·u
    // (L ≪ R); weight, attenuation, 1/R, the polar form and the prefactor add
    // 8u to the magnitude; and summing the 2 × 24 × 24 terms adds 1152u of the
    // absolute sum, which Σ weight / R ≤ w·h / z bounds. Both evaluations err,
    // so they may differ by twice that.
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
    let scale = k / TAU * Complex64::new(0.3, -0.7).norm() * width * height;
    for ((a, b), point) in split.iter().zip(&joined).zip(&points) {
        let farthest = (point[0].abs() + 0.5 * width)
            .hypot(point[1].abs() + 0.5 * height)
            .hypot(point[2]);
        let roundings = 6.0 * k * farthest + 8.0 + 2.0 * 24.0 * 24.0;
        let rounding_bound = 2.0 * roundings * 0.5 * f64::EPSILON * scale / point[2];
        assert!((a - b).norm() <= rounding_bound, "{a} against {b}");
    }
}

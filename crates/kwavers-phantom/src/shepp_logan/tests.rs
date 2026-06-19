//! Value-semantic tests for the Shepp–Logan phantom. Expected values are derived
//! by hand from which ellipses contain the test point.

use super::{SheppLogan, SheppLoganVariant};

#[test]
fn origin_value_is_outer_plus_inner_ellipse_original() {
    // At (0,0) only ellipses 1 (A=2.0, centred 0,0) and 2 (A=-0.98, centred
    // 0,-0.0184) contain the point — all others are offset/too small. Sum = 1.02.
    let p = SheppLogan::original();
    assert!((p.value_at(0.0, 0.0) - 1.02).abs() < 1e-12, "got {}", p.value_at(0.0, 0.0));
}

#[test]
fn origin_value_is_outer_plus_inner_ellipse_modified() {
    // Same geometry; modified intensities 1.0 + (−0.8) = 0.2 at the origin.
    let p = SheppLogan::modified();
    assert!((p.value_at(0.0, 0.0) - 0.2).abs() < 1e-12, "got {}", p.value_at(0.0, 0.0));
}

#[test]
fn outside_the_head_is_zero() {
    // (0.95, 0.95) is outside the outer ellipse ((0.95/0.69)² > 1) ⇒ no ellipse.
    let p = SheppLogan::modified();
    assert_eq!(p.value_at(0.95, 0.95), 0.0);
    assert_eq!(p.value_at(-0.99, -0.99), 0.0);
}

#[test]
fn point_inside_an_offset_inclusion_sums_three_ellipses_modified() {
    // The small ellipse 7 is centred at (0, −0.1), radius 0.046 — well inside both
    // big ellipses and clear of ellipse 5 (centred (0,0.35), b=0.25, reaching only
    // down to y=0.10). At its centre: ellipse1 (1.0) + ellipse2 (−0.8) + ellipse7
    // (0.1) = 0.3.
    let p = SheppLogan::modified();
    let v = p.value_at(0.0, -0.1);
    assert!((v - 0.3).abs() < 1e-12, "expected 0.3 at inclusion centre, got {v}");
}

#[test]
fn ellipse_membership_respects_semi_axes() {
    let p = SheppLogan::original();
    let outer = p.ellipses()[0]; // a=0.69, b=0.92, centred origin, φ=0
    assert!(outer.contains(0.68, 0.0));
    assert!(!outer.contains(0.70, 0.0)); // just past x semi-axis
    assert!(outer.contains(0.0, 0.91));
    assert!(!outer.contains(0.0, 0.93)); // just past y semi-axis
}

#[test]
fn rasterize_has_expected_shape_and_finite_values() {
    let p = SheppLogan::modified();
    let img = p.rasterize(64);
    assert_eq!(img.dim(), (64, 64));
    assert!(img.iter().all(|v| v.is_finite()));
    // Corners are outside the head ⇒ 0.
    assert_eq!(img[[0, 0]], 0.0);
    assert_eq!(img[[63, 63]], 0.0);
    // The image must contain the high-intensity skull (≈1.0) and the dark
    // interior region (≈0.2) for the modified variant.
    let max = img.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(max >= 0.99, "skull intensity ~1.0 expected, got max {max}");
}

#[test]
fn variant_selection_matches_named_constructors() {
    assert_eq!(SheppLogan::new(SheppLoganVariant::Original), SheppLogan::original());
    assert_eq!(SheppLogan::new(SheppLoganVariant::Modified), SheppLogan::modified());
    assert_eq!(SheppLogan::original().ellipses().len(), 10);
}

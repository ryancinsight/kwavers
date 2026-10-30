//! Rejection tests for [`KWaveArray`] element constructors: every
//! degenerate input is a typed `KwaversError::Config`, never a panic.

use super::super::KWaveArray;

/// Degenerate annulus diameters are typed `Config` errors, not panics: the
/// retired `assert!(0 ≤ inner < outer)` is now a `Result`.
#[test]
fn test_add_annular_element_rejects_degenerate_diameters() {
    use kwavers_core::error::{ConfigError, KwaversError};

    let centre = (0.0, 0.0, 0.0);
    let radius = 15.0e-3;
    // inner == outer, inner > outer, negative inner, and NaN inner all fail
    // the `outer > inner && inner >= 0` comparison.
    for (inner_d, outer_d) in [
        (4.0e-3, 4.0e-3),
        (6.0e-3, 4.0e-3),
        (-1.0e-3, 4.0e-3),
        (f64::NAN, 4.0e-3),
    ] {
        let mut arr = KWaveArray::new();
        let err = arr
            .add_annular_element(centre, radius, inner_d, outer_d)
            .expect_err("degenerate annulus must be rejected");
        assert!(
            matches!(err, KwaversError::Config(ConfigError::InvalidValue { .. })),
            "degenerate annulus must report Config::InvalidValue, got {err}"
        );
        assert_eq!(
            arr.num_elements(),
            0,
            "rejected annulus must add no element"
        );
    }

    // The boundary `inner == 0` is valid and adds exactly one element.
    let mut arr = KWaveArray::new();
    arr.add_annular_element(centre, radius, 0.0, 4.0e-3)
        .expect("inner == 0 is a full bowl section");
    assert_eq!(arr.num_elements(), 1);
}

/// A failing pair in `add_annular_array` aborts the batch with a typed
/// error; earlier valid pairs stay, the failing and later pairs are absent.
#[test]
fn test_add_annular_array_rejects_degenerate_pair() {
    use kwavers_core::error::{ConfigError, KwaversError};

    let centre = (0.0, 0.0, 0.0);
    let radius = 15.0e-3;
    let mut arr = KWaveArray::new();
    let err = arr
        .add_annular_array(
            centre,
            radius,
            &[(0.0, 4.0e-3), (6.0e-3, 4.0e-3), (0.0, 8.0e-3)],
        )
        .expect_err("degenerate pair must abort the batch");
    assert!(
        matches!(err, KwaversError::Config(ConfigError::InvalidValue { .. })),
        "degenerate pair must report Config::InvalidValue, got {err}"
    );
    assert_eq!(
        arr.num_elements(),
        1,
        "only the pairs before the failure are added"
    );
}

/// Degenerate rectangle dimensions are typed `Config` errors, not panics:
/// the retired `.expect()` on `oriented_with_convention` is now `?`.
#[test]
fn test_add_rect_element_rejects_degenerate_dimensions() {
    use kwavers_core::error::{ConfigError, KwaversError};

    let centre = (1.0e-3, 1.0e-3, 1.0e-3);
    // Zero/negative width or height, and NaN, all fail positive-finite
    // validation.
    for (width, height) in [
        (0.0, 2.0e-4),
        (8.0e-4, -1.0e-4),
        (f64::NAN, 2.0e-4),
        (8.0e-4, f64::INFINITY),
    ] {
        let mut arr = KWaveArray::new();
        let err = arr
            .add_rect_element(centre, width, height)
            .expect_err("degenerate rect must be rejected");
        assert!(
            matches!(err, KwaversError::Config(ConfigError::InvalidValue { .. })),
            "degenerate rect must report Config::InvalidValue, got {err}"
        );
        assert_eq!(arr.num_elements(), 0, "rejected rect must add no element");

        let mut arr = KWaveArray::new();
        let err = arr
            .add_rect_rot_element(centre, width, height, (0.0, 0.0, 30.0))
            .expect_err("degenerate rotated rect must be rejected");
        assert!(
            matches!(err, KwaversError::Config(ConfigError::InvalidValue { .. })),
            "degenerate rotated rect must report Config::InvalidValue, got {err}"
        );
        assert_eq!(arr.num_elements(), 0, "rejected rect must add no element");
    }

    // A non-finite centre or Euler angle is rejected before any frame is
    // built, so no NaN geometry can be constructed.
    let mut arr = KWaveArray::new();
    let err = arr
        .add_rect_element((f64::NAN, 1.0e-3, 1.0e-3), 8.0e-4, 2.0e-4)
        .expect_err("non-finite centre must be rejected");
    assert!(
        matches!(err, KwaversError::Config(ConfigError::InvalidValue { .. })),
        "non-finite centre must report Config::InvalidValue, got {err}"
    );

    let mut arr = KWaveArray::new();
    let err = arr
        .add_rect_rot_element(centre, 8.0e-4, 2.0e-4, (0.0, f64::INFINITY, 0.0))
        .expect_err("non-finite Euler angle must be rejected");
    assert!(
        matches!(err, KwaversError::Config(ConfigError::InvalidValue { .. })),
        "non-finite Euler angle must report Config::InvalidValue, got {err}"
    );
}

/// A degenerate disc — non-positive diameter or a focus coinciding with the
/// centre — is a typed `Config` error, not a panic: the retired
/// `disc_basis` assert and `.expect()` are now pre-validation plus `?`.
#[test]
fn test_add_disc_element_rejects_degenerate_inputs() {
    use kwavers_core::error::{ConfigError, KwaversError};

    let centre = (1.0e-3, 1.0e-3, 1.0e-3);
    for diameter in [0.0, -2.0e-3, f64::NAN, f64::INFINITY] {
        let mut arr = KWaveArray::new();
        let err = arr
            .add_disc_element(centre, diameter, None)
            .expect_err("degenerate disc must be rejected");
        assert!(
            matches!(err, KwaversError::Config(ConfigError::InvalidValue { .. })),
            "degenerate disc must report Config::InvalidValue, got {err}"
        );
        assert_eq!(arr.num_elements(), 0, "rejected disc must add no element");
    }

    // Focus at the centre (and within disc-axis epsilon of it) fails the
    // `norm > DISC_AXIS_EPSILON` comparison `disc_basis` asserts on.
    for focus in [centre, (centre.0 + 1.0e-15, centre.1, centre.2)] {
        let mut arr = KWaveArray::new();
        let err = arr
            .add_disc_element(centre, 4.0e-3, Some(focus))
            .expect_err("coincident focus must be rejected");
        assert!(
            matches!(err, KwaversError::Config(ConfigError::InvalidValue { .. })),
            "coincident focus must report Config::InvalidValue, got {err}"
        );
        assert_eq!(arr.num_elements(), 0, "rejected disc must add no element");
    }
}

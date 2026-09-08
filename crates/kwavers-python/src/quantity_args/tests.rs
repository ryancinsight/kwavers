//! The boundary accepts both forms and rejects a wrong dimension.
//!
//! These assert the contract every converted signature in this crate now
//! carries, once, rather than repeating it per call site.

use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict};

use super::{PyDimensionless, PyLength, PyMassDensity, PyPressure};

/// Build a duck-typed quantity in Python and bind it.
///
/// Deliberately not a `pyaequitas` object: the contract this crate depends on
/// is the attribute protocol, so the test exercises that and needs no wheel
/// installed to run.
fn quantity<'py>(
    py: Python<'py>,
    base: f64,
    exponents: &str,
    semantics: &str,
) -> Bound<'py, PyAny> {
    let source = format!(
        "class Foreign:\n    __aequitas_base__ = {base}\n    __aequitas_dimension__ = ({exponents}, '{semantics}')\n\nresult = Foreign()\n"
    );
    let globals = PyDict::new(py);
    py.run(
        &std::ffi::CString::new(source).expect("no interior nul"),
        Some(&globals),
        None,
    )
    .expect("the fixture defines");
    globals
        .get_item("result")
        .expect("lookup")
        .expect("`result` is assigned")
}

const LENGTH: &str = "(1, 0, 0, 0, 0, 0, 0)";
const TIME: &str = "(0, 0, 1, 0, 0, 0, 0)";
const PRESSURE: &str = "(-1, 1, -2, 0, 0, 0, 0)";
const SCALAR: &str = "(0, 0, 0, 0, 0, 0, 0)";

#[test]
fn a_bare_float_is_still_base_si_units() {
    // The published contract: `radius=0.0025` meant metres and still does.
    Python::attach(|py| {
        let value = 0.0025_f64.into_pyobject(py).expect("bind");
        let radius: PyLength = value.extract().expect("a float extracts");
        assert!((radius.base() - 0.0025).abs() < f64::EPSILON);
    });
}

#[test]
fn a_matching_quantity_is_accepted() {
    Python::attach(|py| {
        let value = quantity(py, 0.0025, LENGTH, "base");
        let radius: PyLength = value.extract().expect("a length extracts");
        assert!((radius.base() - 0.0025).abs() < f64::EPSILON);
    });
}

#[test]
fn a_wrong_dimension_raises_instead_of_being_scaled() {
    Python::attach(|py| {
        let value = quantity(py, 2.0, TIME, "base");
        let error = value
            .extract::<PyLength>()
            .expect_err("a time is not a radius");
        assert!(error.to_string().contains("expected a quantity"), "{error}");
    });
}

#[test]
fn a_dimensioned_quantity_cannot_pass_as_a_ratio() {
    Python::attach(|py| {
        let pressure = quantity(py, 1.0, PRESSURE, "base");
        assert!(
            pressure.extract::<PyDimensionless>().is_err(),
            "a pressure must not satisfy a dimensionless fraction"
        );

        let ratio = quantity(py, 0.3, SCALAR, "base");
        assert!(ratio.extract::<PyDimensionless>().is_ok());
    });
}

#[test]
fn a_stress_does_not_satisfy_a_pressure_parameter() {
    // Same exponents; the semantic marker is the whole difference, and it must
    // still separate them after crossing the boundary.
    Python::attach(|py| {
        let stress = quantity(py, 1.0, PRESSURE, "stress");
        assert!(stress.extract::<PyPressure>().is_err());

        let pressure = quantity(py, 1.0, PRESSURE, "base");
        assert!(pressure.extract::<PyPressure>().is_ok());
    });
}

#[test]
fn the_extracted_value_reaches_the_domain_as_a_typed_quantity() {
    use aequitas::systems::si::units::KilogramPerCubicMeter;

    Python::attach(|py| {
        let value = quantity(py, 1000.0, "(-3, 1, 0, 0, 0, 0, 0)", "base");
        let density: PyMassDensity = value.extract().expect("a mass density extracts");
        let typed = density.quantity();
        assert!((typed.in_unit::<KilogramPerCubicMeter>() - 1000.0).abs() < f64::EPSILON);
    });
}

#[test]
fn a_non_quantity_object_is_rejected() {
    Python::attach(|py| {
        let text = "not a quantity".into_pyobject(py).expect("bind");
        assert!(text.extract::<PyLength>().is_err());
    });
}

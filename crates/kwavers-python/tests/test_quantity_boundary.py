"""Dimensioned parameters accept a quantity, and reject the wrong dimension.

Every physical parameter on the converted surface used to be a bare float
whose dimension lived only in its name. These assert the two halves of the new
contract from the consumer's side: the published float form still means
canonical SI base units, and a quantity of the right dimension is accepted
while one of the wrong dimension raises.

The fixtures are duck-typed rather than `pyaequitas` objects. That is the
contract this package depends on -- two extension modules cannot share Rust
types -- so the tests need no second wheel installed to run, and they exercise
exactly what a foreign quantity would present.
"""

from __future__ import annotations

import math

import pytest

import pykwavers as kw


class Quantity:
    """An object carrying the Aequitas quantity protocol."""

    def __init__(self, base: float, exponents: tuple, semantics: str = "base") -> None:
        self.__aequitas_base__ = float(base)
        self.__aequitas_dimension__ = (exponents, semantics)


LENGTH = (1, 0, 0, 0, 0, 0, 0)
TIME = (0, 0, 1, 0, 0, 0, 0)
FREQUENCY = (0, 0, -1, 0, 0, 0, 0)
VELOCITY = (1, 0, -1, 0, 0, 0, 0)
MASS_DENSITY = (-3, 1, 0, 0, 0, 0, 0)
PRESSURE = (-1, 1, -2, 0, 0, 0, 0)
VOLTAGE = (2, 1, -3, -1, 0, 0, 0)
DIMENSIONLESS = (0, 0, 0, 0, 0, 0, 0)
ANGLE = (0, 0, 0, 0, 0, 0, 0)


def metres(value: float) -> Quantity:
    return Quantity(value, LENGTH)


# --- the float form is unchanged -------------------------------------------


def test_floats_still_mean_base_units() -> None:
    # 20 um radius, 1 um plate, 0.2 um gap, water. The published contract.
    from_floats = kw.cmut_collapse_voltage(20e-6, 1e-6, 0.2e-6)
    assert math.isfinite(from_floats)
    assert from_floats > 0.0


def test_quantities_and_floats_agree() -> None:
    from_floats = kw.cmut_collapse_voltage(20e-6, 1e-6, 0.2e-6)
    from_quantities = kw.cmut_collapse_voltage(
        metres(20e-6), metres(1e-6), metres(0.2e-6)
    )
    assert from_quantities == from_floats


def test_mixed_forms_are_accepted() -> None:
    # A caller migrating one argument at a time must not be broken.
    mixed = kw.cmut_collapse_voltage(metres(20e-6), 1e-6, metres(0.2e-6))
    assert mixed == kw.cmut_collapse_voltage(20e-6, 1e-6, 0.2e-6)


# --- the wrong dimension is refused ----------------------------------------


@pytest.mark.parametrize("wrong", [TIME, PRESSURE, VELOCITY])
def test_a_non_length_is_rejected_where_a_length_belongs(wrong: tuple) -> None:
    with pytest.raises(ValueError, match="expected a quantity"):
        kw.cmut_collapse_voltage(Quantity(1.0, wrong), 1e-6, 0.2e-6)


def test_a_length_is_rejected_where_a_density_belongs() -> None:
    with pytest.raises(ValueError):
        kw.cmut_resonance_immersion(20e-6, 1e-6, 0.2e-6, metres(1000.0))


def test_a_dimensioned_quantity_is_rejected_where_a_ratio_belongs() -> None:
    with pytest.raises(ValueError):
        kw.cmut_max_output_pressure(
            20e-6, 1e-6, 0.2e-6, 1000.0, 1500.0, Quantity(0.33, PRESSURE)
        )


def test_stress_does_not_satisfy_a_pressure_parameter() -> None:
    # Same exponents; only the semantic marker differs.
    stress = Quantity(1.7e11, PRESSURE, "stress")
    with pytest.raises(ValueError):
        kw.mems_clamped_plate_resonance(stress, 2e-6, 0.28, 3100.0, 4e-5)


# --- the converted surface, function by function ---------------------------


def test_plate_resonance_accepts_quantities() -> None:
    both = kw.mems_clamped_plate_resonance(
        Quantity(1.7e11, PRESSURE), metres(2e-6), Quantity(0.28, DIMENSIONLESS),
        Quantity(3100.0, MASS_DENSITY), metres(4e-5),
    )
    assert both == kw.mems_clamped_plate_resonance(1.7e11, 2e-6, 0.28, 3100.0, 4e-5)


def test_cmut_self_heating_accepts_quantities() -> None:
    value = kw.cmut_self_heating(
        20e-6, 1e-6, 0.2e-6, Quantity(10.0, VOLTAGE), Quantity(2e6, FREQUENCY)
    )
    assert value == kw.cmut_self_heating(20e-6, 1e-6, 0.2e-6, 10.0, 2e6)


def test_pmut_max_output_pressure_accepts_quantities() -> None:
    value = kw.pmut_max_output_pressure(
        "pzt", metres(50e-6), metres(2e-6), metres(5e-6),
        Quantity(20.0, VOLTAGE), Quantity(1000.0, MASS_DENSITY),
        Quantity(1500.0, VELOCITY),
    )
    assert value == kw.pmut_max_output_pressure(
        "pzt", 50e-6, 2e-6, 5e-6, 20.0, 1000.0, 1500.0
    )


def test_transducer_array_accepts_quantities() -> None:
    from_floats = kw.TransducerArray2D(64, 0.3e-3, 5e-3, 0.35e-3, 1540.0, 3e6)
    from_quantities = kw.TransducerArray2D(
        64, metres(0.3e-3), metres(5e-3), metres(0.35e-3),
        Quantity(1540.0, VELOCITY), Quantity(3e6, FREQUENCY),
    )
    assert from_quantities.element_width == from_floats.element_width


def test_transducer_setters_accept_quantities() -> None:
    array = kw.TransducerArray2D(64, 0.3e-3, 5e-3, 0.35e-3, 1540.0, 3e6)
    array.set_focus_distance(metres(0.05))
    array.set_position(metres(0.0), metres(0.0), metres(0.01))


def test_transducer_rejects_a_wrong_dimension() -> None:
    with pytest.raises(ValueError):
        kw.TransducerArray2D(
            64, Quantity(1.0, TIME), 5e-3, 0.35e-3, 1540.0, 3e6
        )


# --- the degrees contract --------------------------------------------------


def test_steering_angle_keeps_its_degree_contract() -> None:
    """A bare float is degrees, as it has always been documented.

    Typing this as a plain angle would have silently reinterpreted every
    existing caller's number as radians, so the float arm is unchanged and only
    the quantity arm is new.
    """
    from_degrees = kw.TransducerArray2D(64, 0.3e-3, 5e-3, 0.35e-3, 1540.0, 3e6)
    from_degrees.set_steering_angle(30.0)

    from_radians = kw.TransducerArray2D(64, 0.3e-3, 5e-3, 0.35e-3, 1540.0, 3e6)
    from_radians.set_steering_angle(Quantity(math.radians(30.0), ANGLE, "angle"))

    assert from_degrees.steering_angle == pytest.approx(from_radians.steering_angle)


def test_steering_angle_rejects_a_non_angle() -> None:
    array = kw.TransducerArray2D(64, 0.3e-3, 5e-3, 0.35e-3, 1540.0, 3e6)
    with pytest.raises(ValueError):
        array.set_steering_angle(metres(1.0))


def test_steering_angle_still_rejects_a_non_finite_float() -> None:
    array = kw.TransducerArray2D(64, 0.3e-3, 5e-3, 0.35e-3, 1540.0, 3e6)
    with pytest.raises(ValueError):
        array.set_steering_angle(float("nan"))

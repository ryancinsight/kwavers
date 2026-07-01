from __future__ import annotations

import sys
from pathlib import Path


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))


def test_chapter33_requires_pykwavers_without_optional_import_branch():
    source = (BOOK_DIR / "ch33_cmut_vs_pmut.py").read_text(encoding="utf-8")

    assert "import pykwavers as kw" in source
    assert "except ImportError" not in source
    assert "try:" not in source


def test_chapter33_routes_mems_physics_through_rust_bindings():
    source = (BOOK_DIR / "ch33_cmut_vs_pmut.py").read_text(encoding="utf-8")

    required_helpers = (
        "kw.mems_clamped_plate_resonance",
        "kw.mems_immersion_resonance",
        "kw.cmut_collapse_voltage",
        "kw.cmut_coupling_k2",
        "kw.cmut_self_heating",
        "kw.cmut_fractional_bandwidth",
        "kw.cmut_max_output_pressure",
        "kw.cmut_flex_gap_derating",
        "kw.pmut_coupling_k2",
        "kw.pmut_self_heating",
        "kw.pmut_fractional_bandwidth",
        "kw.pmut_max_output_pressure",
        "kw.ivus_figure_of_merit",
    )
    for helper in required_helpers:
        assert helper in source


def test_cmut_collapse_voltage_scales_with_gap():
    import pykwavers as kw

    radius_m = 14.0e-6
    thickness_m = 0.4e-6
    low_gap_v = kw.cmut_collapse_voltage(radius_m, thickness_m, 100.0e-9)
    high_gap_v = kw.cmut_collapse_voltage(radius_m, thickness_m, 400.0e-9)

    # Parallel-plate pull-in gives Vc proportional to g^(3/2) for fixed plate
    # stiffness; widening the gap by 4x therefore raises collapse voltage by 8x.
    assert abs(high_gap_v / low_gap_v - 8.0) <= 8.0e-12


def test_pmut_output_pressure_is_drive_scaled_and_film_ordered():
    import pykwavers as kw

    rho = 1000.0
    sound_speed = 1500.0
    low_drive = kw.pmut_max_output_pressure("pzt", 60.0e-6, 2.0e-6, 4.0e-6, 10.0, rho, sound_speed)
    high_drive = kw.pmut_max_output_pressure("pzt", 60.0e-6, 2.0e-6, 4.0e-6, 20.0, rho, sound_speed)
    aln_drive = kw.pmut_max_output_pressure("aln", 60.0e-6, 2.0e-6, 4.0e-6, 20.0, rho, sound_speed)

    # The Rust PMUT model is linear in drive voltage before material breakdown,
    # and PZT's larger piezoelectric coefficient gives larger pressure than AlN.
    assert abs(high_drive / low_drive - 2.0) <= 2.0e-12
    assert high_drive > aln_drive


def test_chapter33_default_ivus_verdict_prefers_cmut():
    import pykwavers as kw

    *_, recommendation = kw.ivus_figure_of_merit(
        14.0e-6,
        0.4e-6,
        0.25e-6,
        "pzt",
        20.0e-6,
        1.0e-6,
        2.0e-6,
        1060.0,
        5.0,
    )

    assert recommendation == 0.0

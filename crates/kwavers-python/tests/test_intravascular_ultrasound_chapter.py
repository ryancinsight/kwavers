from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))

import ch30_intravascular_ultrasound as ivus  # noqa: E402


def test_chapter30_uses_rust_kernels_without_extension_fallbacks():
    source = Path(ivus.__file__).read_text(encoding="utf-8")

    forbidden_tokens = (
        "_HAS_PYKWAVERS",
        "except ImportError",
        "kw = None",
        "np.convolve",
        "gaussian_kernel",
        "pressure**2 / (2.0 * rho * c)",
        "heat_source * tau_s / (rho * specific_heat)",
        "np.log10",
        "peak_p_pa / 1.0e6 / np.sqrt",
        "kw.mechanical_index",
        "kw.acoustic_intensity_from_amplitude",
        "kw.adiabatic_temperature_rise_kelvin",
        "DB_CM_MHZ_TO_NP_M_MHZ",
        "np.random.default_rng",
        "rng.rayleigh",
        "np.gradient",
        "sound_speed[lumen]",
        "angle_gain = np.exp(",
        "* angle_gain * np.exp",
        "def angle_difference",
        "radial_band = np.exp",
        "acoustic_radiation_force =",
        "deposition /= max",
        "delivered_fraction = 1.0 - np.exp",
        "def nearest_indices",
        "rr * np.cos(tt)",
        "rr * np.sin(tt)",
        "attenuation = np.exp(-2.0",
        "rf += 0.10 * np.exp",
        "def scan_convert",
        "np.rint((r - r_axis[0])",
        "polar[ri, ti]",
        "kw.bmode_db_fixed_reference",
        "kw.bmode_envelope",
        "kw.ivus_polar_bmode_rf",
        "kw.ivus_scan_convert",
        "for col in range(angle_samples)",
        "np.maximum(envelope",
        "np.clip((db",
        "np.count_nonzero(phantom.lumen_mask)",
        "np.count_nonzero(phantom.plaque_mask)",
        "np.mean(bmode[\"cartesian\"]",
        "def pixel_area_m2",
        "alpha_eff_np_m",
        "absorbed_power",
        "def target_to_offtarget_ratio",
        "kw.ivus_therapy_pressure_field",
        "kw.ivus_therapy_response",
        "response = kw.ivus_therapy_response",
        "pressure = pressure.reshape",
    )
    for token in forbidden_tokens:
        assert token not in source

    required_calls = (
        "kw.ivus_vessel_phantom",
        "kw.ivus_bmode_image",
        "kw.ivus_chapter_metrics",
        "kw.ivus_therapy_fields",
    )
    for call in required_calls:
        assert call in source


def test_chapter30_rust_backed_helpers_match_closed_forms():
    pressure = np.array([[1.0e6, 5.0e5]], dtype=float)
    intensity = np.asarray(
        ivus.kw.acoustic_intensity_from_amplitude(
            np.ascontiguousarray(pressure.ravel(), dtype=np.float64),
            ivus.RHO_TISSUE_KG_M3,
            ivus.C_TISSUE_M_S,
        )
    ).reshape(pressure.shape)
    expected_intensity = pressure**2 / (2.0 * ivus.RHO_TISSUE_KG_M3 * ivus.C_TISSUE_M_S)
    assert np.allclose(intensity, expected_intensity, rtol=1.0e-12, atol=0.0)

    heat_source = np.array([[1.0e6, 2.0e6]], dtype=float)
    tau_s = 0.25
    delta_t = np.asarray(
        ivus.kw.adiabatic_temperature_rise_kelvin(
            np.ascontiguousarray(heat_source.ravel(), dtype=np.float64),
            np.full(heat_source.size, tau_s, dtype=np.float64),
            ivus.RHO_TISSUE_KG_M3,
            ivus.CP_TISSUE_J_KG_K,
        )
    ).reshape(heat_source.shape)
    expected_delta_t = heat_source * tau_s / (ivus.RHO_TISSUE_KG_M3 * ivus.CP_TISSUE_J_KG_K)
    assert np.allclose(delta_t, expected_delta_t, rtol=1.0e-12, atol=0.0)

    db = np.asarray(
        ivus.kw.bmode_db_fixed_reference(
            np.array([1.0, 0.1, 1.0e-3, 0.0], dtype=np.float64), 1.0, -40.0
        )
    )
    assert np.allclose(db, np.array([0.0, -20.0, -40.0, -40.0]), rtol=0.0, atol=1.0e-12)

    catheter = ivus.TransducerDesign().catheter_radius_m
    peak = ivus.TransducerDesign().therapy_pressure_pa
    azimuth = ivus.TransducerDesign().therapy_azimuth_rad
    width = ivus.TransducerDesign().therapy_sector_width_rad
    decay = 3.2e-3
    pressure_field = np.asarray(
        ivus.kw.ivus_therapy_pressure_field(
            np.array([catheter, catheter + decay, catheter + decay]),
            np.array([azimuth, azimuth, azimuth + width]),
            catheter,
            peak,
            azimuth,
            width,
            decay,
        )
    )
    expected_pressure = np.array([0.0, peak / np.e, peak * np.exp(-1.5)])
    assert np.allclose(pressure_field, expected_pressure, rtol=1.0e-12, atol=1.0e-12)

    delivered = np.asarray(
        ivus.kw.ivus_microbubble_delivery_fraction(
            np.array([0.0, 1.75e-3, 1.75e-3]),
            np.array([10.0, 10.0, 10.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([False, True, True]),
            np.array([False, False, True]),
            ivus.C_TISSUE_M_S,
            1.75e-3,
            1.2e-3,
        )
    )
    expected_delivery = np.array([0.0, 1.0 - np.exp(-0.6), 1.0 - np.exp(-3.0)])
    assert np.allclose(delivered, expected_delivery, rtol=1.0e-12, atol=1.0e-12)

    response = ivus.kw.ivus_therapy_response(
        np.array([[1.0e6, 1.0e6]], dtype=np.float64),
        np.array([[2.30e-3, 2.30e-3]], dtype=np.float64),
        np.array([[1.0, 1.0]], dtype=np.float64),
        np.array([[True, True]], dtype=bool),
        np.array([[False, False]], dtype=bool),
        np.array([[False, True]], dtype=bool),
        np.array([[False, False]], dtype=bool),
        np.array([[False, True]], dtype=bool),
        0.55e-3,
        2.0e6,
        0.25,
        0.50,
        1000.0,
        1500.0,
        4000.0,
        1.75e-3,
        1.2e-3,
    )
    expected_intensity = 1.0e12 / (2.0 * 1000.0 * 1500.0)
    alpha_eff = 100.0 / 8.686 * 2.0
    expected_delta = 2.0 * alpha_eff * expected_intensity * 0.25 * 0.50 / (1000.0 * 4000.0)
    expected_ratio = (1.0 - np.exp(-3.0)) / (1.0 - np.exp(-0.6))
    assert np.allclose(response["intensity_w_m2"], [expected_intensity, expected_intensity], rtol=1.0e-12)
    assert np.allclose(response["temperature_rise_k"], [expected_delta, expected_delta], rtol=1.0e-12)
    assert np.allclose(response["deposition"], [1.0 - np.exp(-0.6), 1.0 - np.exp(-3.0)], rtol=1.0e-12)
    assert np.isclose(response["mechanical_index"], 1.0 / np.sqrt(2.0), rtol=1.0e-12)
    assert np.isclose(response["target_to_offtarget_ratio"], expected_ratio, rtol=1.0e-12)

    radius = np.array([[0.55e-3, 2.30e-3, 2.30e-3]], dtype=np.float64)
    theta = np.array([[-0.72, -0.72, -0.72]], dtype=np.float64)
    attenuation = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    eel = np.array([[False, True, True]], dtype=bool)
    lumen = np.array([[False, False, False]], dtype=bool)
    cap = np.array([[False, False, True]], dtype=bool)
    lipid = np.array([[False, False, False]], dtype=bool)
    plaque = np.array([[False, False, True]], dtype=bool)
    fields = ivus.kw.ivus_therapy_fields(
        radius,
        theta,
        attenuation,
        eel,
        lumen,
        cap,
        lipid,
        plaque,
        0.55e-3,
        1.0e6,
        -0.72,
        0.50,
        3.2e-3,
        2.0e6,
        0.25,
        0.50,
        1000.0,
        1500.0,
        4000.0,
        1.75e-3,
        1.2e-3,
    )
    expected_pressure = np.asarray(
        ivus.kw.ivus_therapy_pressure_field(
            radius.ravel(),
            theta.ravel(),
            0.55e-3,
            1.0e6,
            -0.72,
            0.50,
            3.2e-3,
        )
    )
    expected_response = ivus.kw.ivus_therapy_response(
        expected_pressure.reshape(radius.shape),
        radius,
        attenuation,
        eel,
        lumen,
        cap,
        lipid,
        plaque,
        0.55e-3,
        2.0e6,
        0.25,
        0.50,
        1000.0,
        1500.0,
        4000.0,
        1.75e-3,
        1.2e-3,
    )
    assert np.allclose(fields["pressure_pa"], expected_pressure, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        fields["intensity_w_m2"],
        expected_response["intensity_w_m2"],
        rtol=1.0e-12,
    )
    assert np.allclose(
        fields["temperature_rise_k"],
        expected_response["temperature_rise_k"],
        rtol=1.0e-12,
    )
    assert np.allclose(fields["deposition"], expected_response["deposition"], rtol=1.0e-12)
    assert np.isclose(fields["mechanical_index"], expected_response["mechanical_index"], rtol=1.0e-12)
    assert np.isclose(
        fields["target_to_offtarget_ratio"],
        expected_response["target_to_offtarget_ratio"],
        rtol=1.0e-12,
    )
    assert np.isclose(fields["peak_delta_t_k"], expected_response["peak_delta_t_k"], rtol=1.0e-12)

    rf = np.asarray(
        ivus.kw.ivus_polar_bmode_rf(
            np.array([[-1.0e-3, -1.0e-3], [1.0e-3, 1.0e-3]], dtype=np.float64),
            np.array([[-1.0e-3, 1.0e-3], [-1.0e-3, 1.0e-3]], dtype=np.float64),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
            np.array([1.0e-3, 1.55e-3], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            1.0e-3,
            20.0e6,
            0.10,
            0.22e-3,
        )
    )
    expected_attenuation = np.exp(-2.0 * (3.0 * 100.0 / 8.686) * 20.0 * 0.55e-3)
    expected_ring = 0.10 * np.exp(-((0.55e-3 / 0.22e-3) ** 2))
    assert np.allclose(rf, [4.10, 4.0 * expected_attenuation + expected_ring], rtol=1.0e-12, atol=1.0e-12)

    scan = np.asarray(
        ivus.kw.ivus_scan_convert(
            np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64),
            np.array([1.0e-3, 2.0e-3], dtype=np.float64),
            np.array([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi], dtype=np.float64),
            np.array([[0.5e-3, 1.0e-3], [2.0e-3, 2.5e-3]], dtype=np.float64),
            np.array([[-np.pi, -np.pi], [0.0, 0.0]], dtype=np.float64),
        )
    )
    assert np.array_equal(scan, np.array([0.0, 1.0, 7.0, 0.0]))

    image = ivus.kw.ivus_bmode_image(
        np.array([[-1.0e-3, -1.0e-3], [1.0e-3, 1.0e-3]], dtype=np.float64),
        np.array([[-1.0e-3, 1.0e-3], [-1.0e-3, 1.0e-3]], dtype=np.float64),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
        np.array([1.0e-3, 1.55e-3, 2.0e-3, 2.45e-3], dtype=np.float64),
        np.array([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi], dtype=np.float64),
        np.array([[0.5e-3, 1.0e-3], [2.0e-3, 2.5e-3]], dtype=np.float64),
        np.array([[-np.pi, -np.pi], [0.0, 0.0]], dtype=np.float64),
        1.0e-3,
        20.0e6,
        -60.0,
        0.10,
        0.22e-3,
    )
    assert len(image["rf"]) == 16
    assert len(image["envelope"]) == 16
    assert len(image["db"]) == 16
    assert len(image["polar"]) == 16
    assert len(image["cartesian"]) == 4
    assert np.all((np.asarray(image["polar"]) >= 0.0) & (np.asarray(image["polar"]) <= 1.0))
    assert np.array_equal(
        np.asarray(image["cartesian"]),
        ivus.kw.ivus_scan_convert(
            np.asarray(image["polar"]).reshape((4, 4)),
            np.array([1.0e-3, 1.55e-3, 2.0e-3, 2.45e-3], dtype=np.float64),
            np.array([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi], dtype=np.float64),
            np.array([[0.5e-3, 1.0e-3], [2.0e-3, 2.5e-3]], dtype=np.float64),
            np.array([[-np.pi, -np.pi], [0.0, 0.0]], dtype=np.float64),
        ),
    )

    metrics = ivus.kw.ivus_chapter_metrics(
        np.array([[-1.0e-3, -1.0e-3], [1.0e-3, 1.0e-3]], dtype=np.float64),
        np.array([[-1.0e-3, 1.0e-3], [-1.0e-3, 1.0e-3]], dtype=np.float64),
        np.array([[True, False], [False, False]], dtype=bool),
        np.array([[True, True], [True, False]], dtype=bool),
        np.array([[False, True], [False, False]], dtype=bool),
        np.array([[0.20, 0.70], [0.50, 0.10]], dtype=np.float64),
        1540.0,
        20.0e6,
        1.5e6,
        60.0,
        0.25,
        0.04,
        101.0,
    )
    assert np.isclose(metrics["imaging_wavelength_um"], 77.0)
    assert np.isclose(metrics["therapy_wavelength_mm"], 1540.0 / 1.5e6 * 1.0e3)
    assert np.isclose(metrics["lumen_area_mm2"], 4.0)
    assert np.isclose(metrics["plaque_area_mm2"], 4.0)
    assert np.isclose(metrics["bmode_mean_lumen_intensity"], 0.20)
    assert np.isclose(metrics["bmode_mean_wall_intensity"], 0.60)
    assert metrics["therapy_mechanical_index"] == 0.25
    assert metrics["therapy_peak_delta_t_c"] == 0.04
    assert metrics["therapy_target_to_offtarget_deposition_ratio"] == 101.0


def test_dataset_spec_records_public_ivus_boundary_contract():
    spec = ivus.default_dataset_spec()

    assert spec.frame_shape == (384, 384)
    assert spec.annotated_boundaries == ("lumen", "media-adventitia")
    assert spec.imaging_frequency_hz == 20.0e6
    assert any("ivus-segmentation-icsm2018" in url for url in spec.source_urls)


def test_transducer_design_separates_imaging_and_therapy_wavelengths():
    design = ivus.TransducerDesign()
    imaging_wavelength = ivus.C_TISSUE_M_S / design.imaging_frequency_hz
    therapy_wavelength = ivus.C_TISSUE_M_S / design.therapy_frequency_hz

    assert design.imaging_elements == 64
    assert design.therapy_sector_count == 4
    assert imaging_wavelength < 0.08e-3
    assert therapy_wavelength > 1.0e-3
    assert therapy_wavelength / imaging_wavelength > 10.0


def test_vessel_phantom_preserves_nested_ivus_anatomy_and_properties():
    design = ivus.TransducerDesign()
    phantom = ivus.vessel_phantom(n=128, design=design)

    assert phantom.labels.shape == (128, 128)
    assert np.count_nonzero(phantom.lumen_mask) > 500
    assert np.count_nonzero(phantom.plaque_mask) > np.count_nonzero(phantom.lipid_mask)
    assert np.all(phantom.lumen_mask <= phantom.eel_mask)
    assert np.all(phantom.fibrous_cap_mask <= phantom.plaque_mask)
    assert float(np.mean(phantom.sound_speed_m_s[phantom.calcium_mask])) > 2500.0
    assert float(np.mean(phantom.sound_speed_m_s[phantom.lipid_mask])) < 1500.0
    assert float(np.max(phantom.backscatter)) == 1.0
    assert np.all(phantom.backscatter[phantom.labels == 1] == 0.0)


def test_vessel_phantom_is_rust_seed_deterministic():
    design = ivus.TransducerDesign()
    a = ivus.vessel_phantom(n=64, design=design, seed=30)
    b = ivus.vessel_phantom(n=64, design=design, seed=30)
    c = ivus.vessel_phantom(n=64, design=design, seed=31)

    assert np.array_equal(a.labels, b.labels)
    assert np.array_equal(a.backscatter, b.backscatter)
    assert not np.array_equal(a.backscatter, c.backscatter)


def test_bmode_simulation_is_boundary_sensitive_and_finite():
    design = ivus.TransducerDesign()
    phantom = ivus.vessel_phantom(n=128, design=design)
    bmode = ivus.simulate_bmode(phantom, design, radial_samples=128, angle_samples=128)

    assert bmode["polar"].shape == (128, 128)
    assert bmode["cartesian"].shape == phantom.labels.shape
    assert np.all(np.isfinite(bmode["cartesian"]))
    assert 0.0 <= float(np.min(bmode["cartesian"])) <= float(np.max(bmode["cartesian"])) <= 1.0
    wall = phantom.eel_mask & ~phantom.lumen_mask
    assert float(np.mean(bmode["cartesian"][wall])) > float(np.mean(bmode["cartesian"][phantom.lumen_mask]))


def test_microbubble_therapy_targets_plaque_sector_without_excess_heat():
    design = ivus.TransducerDesign()
    phantom = ivus.vessel_phantom(n=128, design=design)
    therapy = ivus.simulate_therapy(phantom, design)
    target = phantom.fibrous_cap_mask | phantom.lipid_mask

    assert np.all(np.isfinite(therapy["pressure_pa"]))
    assert float(therapy["mechanical_index"]) < 0.30
    assert float(therapy["peak_delta_t_c"]) < 0.50
    assert float(therapy["target_to_offtarget_ratio"]) > 100.0
    assert float(np.max(therapy["deposition"][target])) > 0.90
    assert float(np.max(therapy["deposition"][~phantom.plaque_mask])) < 0.05

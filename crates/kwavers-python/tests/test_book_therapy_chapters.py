from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import numpy as np

BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
DOCS_DIR = Path(__file__).resolve().parents[3] / "docs" / "book"
sys.path.insert(0, str(BOOK_DIR))


def test_manifest_includes_bbb_and_hifu_chapters_with_scripts():
    manifest = tomllib.loads((BOOK_DIR / "chapters.toml").read_text(encoding="utf-8"))
    chapters = {int(entry["number"]): entry for entry in manifest["chapter"]}

    assert chapters[24]["script"] == "ch24_bbb_lifu_opening.py"
    assert "BBB" in chapters[24]["title"]
    assert (BOOK_DIR / chapters[24]["script"]).is_file()

    assert chapters[25]["script"] == "ch25_transcranial_brain_fus_planning.py"
    assert "HIFU" in chapters[25]["title"]
    assert (BOOK_DIR / chapters[25]["script"]).is_file()

    assert chapters[26]["script"] == "ch26_neuromodulation.py"
    assert "Neuromodulation" in chapters[26]["title"]
    assert (BOOK_DIR / chapters[26]["script"]).is_file()

    assert chapters[27]["script"] == "ch27_transcranial_ust_brain_imaging.py"
    assert "Transcranial" in chapters[27]["title"]
    assert (BOOK_DIR / chapters[27]["script"]).is_file()

    assert chapters[28]["script"] == "ch28_abdominal_histotripsy_fwi.py"
    assert "Abdominal" in chapters[28]["title"]
    assert (BOOK_DIR / chapters[28]["script"]).is_file()

    assert chapters[29]["script"] == "ch29_theranostic_fwi_platforms.py"
    assert "Same-Device" in chapters[29]["title"]
    assert (BOOK_DIR / chapters[29]["script"]).is_file()

    assert chapters[30]["script"] == "ch30_intravascular_ultrasound.py"
    assert "Intravascular" in chapters[30]["title"]
    assert (BOOK_DIR / chapters[30]["script"]).is_file()

    assert chapters[32]["script"] == "ch32_segmented_tissue_transducer_optimization.py"
    assert "Segmented Tissue" in chapters[32]["title"]
    assert (BOOK_DIR / chapters[32]["script"]).is_file()


def test_book_readme_links_bbb_and_hifu_markdown_chapters():
    readme = (DOCS_DIR / "README.md").read_text(encoding="utf-8")

    assert "(bbb_lifu_opening.md)" in readme
    assert "(hifu_transcranial_ablation.md)" in readme
    assert "(neuromodulation.md)" in readme
    assert "(transcranial_ust_brain_imaging.md)" in readme
    assert "(abdominal_histotripsy_fwi.md)" in readme
    assert "(theranostic_fwi_platforms.md)" in readme
    assert "(intravascular_ultrasound.md)" in readme
    assert "(segmented_tissue_transducer_planning.md)" in readme
    assert (DOCS_DIR / "bbb_lifu_opening.md").is_file()
    assert (DOCS_DIR / "hifu_transcranial_ablation.md").is_file()
    assert (DOCS_DIR / "neuromodulation.md").is_file()
    assert (DOCS_DIR / "transcranial_ust_brain_imaging.md").is_file()
    assert (DOCS_DIR / "abdominal_histotripsy_fwi.md").is_file()
    assert (DOCS_DIR / "theranostic_fwi_platforms.md").is_file()
    assert (DOCS_DIR / "intravascular_ultrasound.md").is_file()
    assert (DOCS_DIR / "segmented_tissue_transducer_planning.md").is_file()


def test_chapters24_and_26_require_pykwavers_without_import_fallbacks():
    chapter_requirements = {
        "ch24_bbb_lifu_opening.py": (
            "kw.solve_keller_miksis",
            "kw.mechanical_index_frequency_sweep",
            "kw.mechanical_index_field",
            "kw.bbb_permeability_hill",
            "kw.bbb_inertial_damage_probability",
            "kw.cavitation_therapeutic_window_indices",
            "kw.cavitation_inertial_fraction_onset_index",
            "kw.cem43_cumulative",
            "kw.ceus_backscatter_display",
        ),
        "ch26_neuromodulation.py": (
            "kw.mechanical_index_field",
            "kw.mechanical_index_cavitation_risk",
            "kw.acoustic_intensity_from_amplitude",
            "kw.compute_acoustic_membrane_tension_py",
            "kw.boltzmann_open_probability_py",
            "kw.lif_response_probability_py",
            "kw.cem43_cumulative",
        ),
    }
    forbidden_tokens = (
        "_HAS_KW",
        "except ImportError",
        "pykwavers not found",
        "0.45 / np.sqrt(f_mhz)",
        "0.18 / np.sqrt(f_mhz)",
        "(p_sweep / 1e6) / np.sqrt",
    )
    script_forbidden_tokens = {
        "ch24_bbb_lifu_opening.py": (
            "1.0 / (1.0 + np.exp",
            "dose[\"inertial\"] / (dose[\"harmonic\"]",
            "dose[\"stable\"] / (dose[\"harmonic\"]",
            "np.argmax(bh_ratio",
            "np.argmax(sh_ratio",
            "broad_frac =",
            "np.argmax(broad_frac",
            "kw.ceus_backscatter_signal",
            "signal / np.max(signal)",
            "np.argmax(signal",
            "kw.compute_cem43(T_arr[:",
            "cem43_sparse",
        ),
        "ch26_neuromodulation.py": (
            "1.0 / (1.0 + np.exp",
            "from scipy.ndimage import gaussian_filter1d",
            "gaussian_filter1d(",
            "spike_train = np.zeros",
            "kw.compute_cem43(T_arr[: i + 1]",
        ),
    }

    for script, required_calls in chapter_requirements.items():
        source = (BOOK_DIR / script).read_text(encoding="utf-8")

        assert "import pykwavers as kw" in source
        for token in forbidden_tokens + script_forbidden_tokens[script]:
            assert token not in source, (script, token)
        for call in required_calls:
            assert call in source, (script, call)


def test_mechanical_index_frequency_sweep_binding_matches_formula():
    import pykwavers as kw

    f_hz = np.array([0.25e6, 1.0e6, 2.25e6], dtype=float)
    pressure_pa = 0.45e6
    sweep = np.asarray(kw.mechanical_index_frequency_sweep(pressure_pa, f_hz))
    expected = (pressure_pa * 1.0e-6) / np.sqrt(f_hz * 1.0e-6)

    assert np.allclose(sweep, expected, rtol=1.0e-12, atol=0.0)
    assert float(sweep[0]) > float(sweep[1]) > float(sweep[2])


def test_eikonal_traveltime_2d_binding_matches_axis_formula():
    import pykwavers as kw

    spacing_m = 1.0e-3
    speed = np.full((7, 7), 1500.0, dtype=np.float64)
    travel = np.asarray(kw.eikonal_traveltime_2d(speed, spacing_m, 3, 1, 6))

    assert travel.shape == speed.shape
    assert travel[3, 1] == 0.0
    assert np.isclose(travel[3, 4], 3.0 * spacing_m / 1500.0, rtol=0.0, atol=1.0e-12)
    assert travel[3, 5] > travel[3, 4] > travel[3, 3] > travel[3, 2]


def test_kirchhoff_point_scatterer_binding_focuses_peak():
    import pykwavers as kw

    aperture_cols = np.ascontiguousarray([0, 4, 8, 12, 16, 20], dtype=np.int64)
    aperture_rows = np.zeros_like(aperture_cols)
    scatterer_rows = np.ascontiguousarray([13], dtype=np.int64)
    scatterer_cols = np.ascontiguousarray([10], dtype=np.int64)

    image = np.asarray(
        kw.kirchhoff_point_scatterer_image_2d(
            21,
            21,
            1.0e-3,
            1500.0,
            aperture_rows,
            aperture_cols,
            scatterer_rows,
            scatterer_cols,
            2.0e-7,
            360,
            5.0e5,
            6,
        )
    )

    peak = np.unravel_index(int(np.abs(image).argmax()), image.shape)
    assert abs(peak[0] - int(scatterer_rows[0])) <= 1
    assert abs(peak[1] - int(scatterer_cols[0])) <= 1
    assert abs(image[peak]) > 0.0


def test_bbb_inertial_damage_probability_binding_matches_logistic_formula():
    import pykwavers as kw

    dose = np.array([0.0, 1.0, 3.5, 5.0], dtype=float)
    threshold = 3.5
    slope = 4.0
    probability = np.asarray(
        kw.bbb_inertial_damage_probability(dose, threshold, slope)
    )
    expected = 1.0 / (1.0 + np.exp(-slope * (dose - threshold)))

    assert np.allclose(probability, expected, rtol=1.0e-12, atol=0.0)
    assert probability[2] == 0.5
    assert np.all((0.0 <= probability) & (probability <= 1.0))
    assert np.all(np.diff(probability) > 0.0)


def test_ceus_backscatter_display_binding_matches_signal_and_peak():
    import pykwavers as kw

    concentrations = np.array([0.0, 1.0, 100.0], dtype=float)
    display = kw.ceus_backscatter_display(concentrations, 2.5e-8, 10e-3, -80.0)
    signal = np.asarray(kw.ceus_backscatter_signal(concentrations, 2.5e-8, 10e-3))

    np.testing.assert_allclose(display["signal"], signal, rtol=0.0, atol=0.0)
    assert display["peak_concentration_ul_ml"] == 1.0
    assert display["peak_signal"] == signal[1]
    assert display["signal_db"][1] == 0.0
    assert display["signal_db"][0] == -80.0
    assert display["signal_db"][2] < 0.0


def test_mechanical_index_cavitation_risk_binding_matches_logistic_formula():
    import pykwavers as kw

    mi = np.array([0.2, 0.7, 1.0, 1.9], dtype=float)
    threshold = 0.7
    slope = 18.0
    risk = np.asarray(kw.mechanical_index_cavitation_risk(mi, threshold, slope))
    expected = 1.0 / (1.0 + np.exp(-slope * (mi - threshold)))

    assert np.allclose(risk, expected, rtol=1.0e-12, atol=0.0)
    assert risk[1] == 0.5
    assert np.all((0.0 <= risk) & (risk <= 1.0))
    assert np.all(np.diff(risk) > 0.0)


def test_cavitation_therapeutic_window_indices_binding_matches_ratio_contract():
    import pykwavers as kw

    harmonic = np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=float)
    stable = np.array([0.0, 0.05, 0.2, 0.5, 1.0], dtype=float)
    inertial = np.array([0.0, 0.1, 0.3, 0.5, 1.2], dtype=float)

    stable_idx, inertial_idx, cap_idx = kw.cavitation_therapeutic_window_indices(
        harmonic, stable, inertial, 0.01, 0.1, 0.04
    )

    assert stable_idx == 2
    assert inertial_idx == 4
    assert cap_idx == 3


def test_cavitation_inertial_fraction_onset_index_binding_matches_contract():
    import pykwavers as kw

    harmonic = np.array([10.0, 10.0, 10.0, 10.0], dtype=float)
    stable = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    inertial = np.array([0.0, 2.0, 8.0, 20.0], dtype=float)

    assert kw.cavitation_inertial_fraction_onset_index(
        harmonic, stable, inertial, 0.4
    ) == 2
    assert kw.cavitation_inertial_fraction_onset_index(
        harmonic, stable, inertial, 0.95
    ) == 3


def test_per_spot_cavitation_dose_grid_binding_matches_numpy_contract():
    import pykwavers as kw

    lateral = np.array([0.0, 2.0e-3], dtype=float)
    axial = np.array([0.0, 1.0e-3], dtype=float)
    pressures = np.array([0.0, 1.0e6, 2.0e6], dtype=float)
    power = np.array([0.0, 2.0, 8.0], dtype=float)

    dose_flat, eff_flat, p_flat, goal = kw.per_spot_cavitation_dose_grid(
        lateral,
        axial,
        1.0e6,
        1.0e6,
        1540.0,
        pressures,
        power,
        3,
        1.5e6,
        8.0,
        True,
    )

    eff_expected = np.array(
        [
            [
                kw.electronic_steering_efficiency(float(dx), float(dz), 1.0e6, 1540.0, True)
                for dx in lateral
            ]
            for dz in axial
        ],
        dtype=float,
    )
    p_expected = np.array(
        [
            [
                1.0e6
                * eff_expected[i, j]
                * np.exp(-8.0 * max(float(dz), 0.0))
                for j, _dx in enumerate(lateral)
            ]
            for i, dz in enumerate(axial)
        ],
        dtype=float,
    )
    dose_expected = 3.0 * np.interp(p_expected, pressures, power)

    assert np.allclose(
        np.asarray(eff_flat).reshape(axial.size, lateral.size),
        eff_expected,
        rtol=1.0e-12,
        atol=0.0,
    )
    assert np.allclose(
        np.asarray(p_flat).reshape(axial.size, lateral.size),
        p_expected,
        rtol=1.0e-12,
        atol=1.0e-9,
    )
    assert np.allclose(
        np.asarray(dose_flat).reshape(axial.size, lateral.size),
        dose_expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert goal == 15.0


def test_chapter24_per_spot_dose_helper_routes_grid_to_rust():
    source = (BOOK_DIR / "cavitation_dose_monitor.py").read_text(encoding="utf-8")
    start = source.index("def per_spot_dose(")
    end = source.index("\ndef plot_cavitation_monitor", start)
    body = source[start:end]

    assert "kw.per_spot_cavitation_dose_grid" in body
    forbidden_tokens = (
        "for i, dz in enumerate(ax):",
        "kw.electronic_steering_efficiency(float(dx)",
        "np.interp(p, pressures_pa, cavitation_power)",
    )
    for token in forbidden_tokens:
        assert token not in body, token


def test_raster_cavitation_pulsing_binding_matches_schedule_contract():
    import pykwavers as kw

    (
        time_s,
        coverage,
        cumulative_dose,
        per_spot_dose,
        per_spot_peak_temp,
        efficacy,
        dt_spot_s,
        treatment_s,
        p_spot_pa,
    ) = kw.raster_cavitation_pulsing(
        np.array([0.0, 0.0], dtype=float),
        np.array([0.0, 0.0], dtype=float),
        100.0,
        1.0e6,
        1540.0,
        np.array([0.0, 100.0], dtype=float),
        np.array([0.0, 10.0], dtype=float),
        2,
        10.0,
        "sequential",
        0,
        0.0,
        True,
        0.0,
        0.0,
        1.0,
        2.0,
        15.0,
        4,
    )

    np.testing.assert_allclose(np.asarray(time_s), np.array([0.0, 0.1, 0.2, 0.3]))
    np.testing.assert_allclose(np.asarray(coverage), np.array([0.0, 0.5, 0.5, 1.0]))
    np.testing.assert_allclose(
        np.asarray(cumulative_dose), np.array([10.0, 20.0, 30.0, 40.0])
    )
    np.testing.assert_allclose(np.asarray(per_spot_dose), np.array([20.0, 20.0]))
    np.testing.assert_allclose(
        np.asarray(per_spot_peak_temp),
        np.full(2, 2.0 + 2.0 * np.exp(-0.1)),
    )
    assert efficacy == 1.0
    assert dt_spot_s == 0.1
    np.testing.assert_allclose(treatment_s, 0.3)
    np.testing.assert_allclose(np.asarray(p_spot_pa), np.array([100.0, 100.0]))


def test_raster_pulsing_helper_routes_schedule_to_rust():
    source = (BOOK_DIR / "cavitation_dose_monitor.py").read_text(encoding="utf-8")
    start = source.index("def simulate_raster_pulsing(")
    end = source.index("\ndef per_spot_dose", start)
    body = source[start:end]

    assert "kw.raster_cavitation_pulsing" in body
    forbidden_tokens = (
        "kw.electronic_steering_efficiency(",
        "np.interp(p_spot",
        "kw.prf_efficacy_factor(",
        "per_spot_dose[s] +=",
        "coverage[k] =",
        "np.interp(tq, t_axis",
    )
    for token in forbidden_tokens:
        assert token not in body, token


def test_population_emission_binding_returns_finite_spectrum():
    import pykwavers as kw

    (
        freqs,
        psd,
        fundamental,
        subharmonic,
        ultraharmonic,
        broadband,
        stable,
        total,
        n_active,
        max_compression,
        max_mach,
    ) = kw.simulate_population_emission(
        80.0e3,
        1.0e6,
        2,
        7,
        2.0e-6,
        0.05,
        2.0,
        96,
        5.0e-2,
        0.12,
        0.0,
        False,
        True,
        0.5,
        0.5,
        3.0e-9,
        0.04,
        160,
        101_325.0,
        998.0,
        1481.0,
        1.0e-3,
        0.0725,
        2330.0,
        1.4,
    )

    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    assert n_active > 0
    assert freqs.shape == psd.shape
    assert freqs.size > 1
    assert np.all(np.diff(freqs) > 0.0)
    assert np.all(np.isfinite(psd))
    assert np.all(psd >= 0.0)
    assert stable == subharmonic + ultraharmonic
    assert total == fundamental + stable + broadband
    assert max_compression >= 1.0
    assert max_mach >= 0.0


def test_population_emission_helper_routes_to_rust():
    source = (BOOK_DIR / "cavitation_dose_monitor.py").read_text(encoding="utf-8")
    start = source.index("def simulate_population_emission(")
    end = source.index("\ndef population_dose_vs_pressure", start)
    body = source[start:end]

    assert "kw.simulate_population_emission" in body
    assert "seed: int" in body
    forbidden_tokens = (
        "rng",
        "rng.integers",
        "np.random.default_rng",
        "kw.simulate_coated_bubble_emission(",
        "kw.simulate_bubble_emission(",
        "np.fft.rfft",
        "kw.cavitation_emission_bands(",
        "traces.append",
        "np.hanning",
    )
    for token in forbidden_tokens:
        assert token not in body, token


def test_population_emission_book_callers_pass_seed_not_python_rng():
    for script in (
        "ch21e_real_liver_ct_histotripsy.py",
        "ch21e_treatment_pipeline.py",
        "ch24_bbb_lifu_opening.py",
    ):
        source = (BOOK_DIR / script).read_text(encoding="utf-8")
        assert "simulate_population_emission(" in source
        assert "rng=np.random.default_rng" not in source
        assert "rng=rng" not in source


def test_population_emission_sweep_binding_returns_band_vectors():
    import pykwavers as kw

    (
        harmonic,
        subharmonic,
        ultraharmonic,
        stable,
        inertial,
        signal,
        n_active,
        max_compression,
        max_mach,
    ) = kw.population_emission_sweep(
        np.array([70.0e3, 80.0e3], dtype=float),
        1.0e6,
        2,
        13,
        2.0e-6,
        0.05,
        2.0,
        96,
        5.0e-2,
        0.12,
        0.0,
        False,
        True,
        0.5,
        0.5,
        3.0e-9,
        0.04,
        160,
        101_325.0,
        998.0,
        1481.0,
        1.0e-3,
        0.0725,
        2330.0,
        1.4,
    )

    harmonic = np.asarray(harmonic)
    subharmonic = np.asarray(subharmonic)
    ultraharmonic = np.asarray(ultraharmonic)
    stable = np.asarray(stable)
    inertial = np.asarray(inertial)
    signal = np.asarray(signal)
    n_active = np.asarray(n_active)
    max_compression = np.asarray(max_compression)
    max_mach = np.asarray(max_mach)

    assert harmonic.shape == subharmonic.shape == ultraharmonic.shape == (2,)
    assert stable.shape == inertial.shape == signal.shape == (2,)
    assert n_active.shape == max_compression.shape == max_mach.shape == (2,)
    assert np.all(n_active > 0)
    assert np.all(np.isfinite(signal))
    assert np.all(signal >= 0.0)
    np.testing.assert_allclose(stable, subharmonic + ultraharmonic)
    np.testing.assert_allclose(signal, stable + inertial)


def test_population_dose_vs_pressure_helper_routes_sweep_to_rust():
    source = (BOOK_DIR / "cavitation_dose_monitor.py").read_text(encoding="utf-8")
    start = source.index("def population_dose_vs_pressure(")
    end = source.index("\ndef dose_vs_pressure", start)
    body = source[start:end]

    assert "kw.population_emission_sweep" in body
    forbidden_tokens = (
        "np.random.default_rng(seed)",
        "for i, pa in enumerate(pressures_pa):",
        "simulate_population_emission(",
        "out[\"harmonic\"][i]",
        "out[\"signal\"][i]",
    )
    for token in forbidden_tokens:
        assert token not in body, token


def test_volume_emission_spectrum_binding_returns_finite_spectrum():
    import pykwavers as kw

    freqs, psd, n_active = kw.volume_emission_spectrum(
        20.0e3,
        1.0e6,
        np.array([1.5e-6, 2.0e-6], dtype=float),
        2.0,
        64,
        5.0e-2,
        64,
        0.25,
        101_325.0,
        998.0,
        0.0725,
        1.4,
        1.0e-3,
        2330.0,
        1481.0,
        0.0,
    )

    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    assert n_active == 2
    assert freqs.shape == psd.shape
    assert freqs.size > 1
    assert np.all(np.diff(freqs) > 0.0)
    assert np.all(np.isfinite(psd))
    assert np.all(psd >= 0.0)
    assert np.any(psd > 0.0)


def test_volume_emission_sweep_binding_returns_band_vectors():
    import pykwavers as kw

    harmonic, subharmonic, ultraharmonic, stable, inertial, n_active = (
        kw.volume_emission_sweep(
            np.array([20.0e3, 30.0e3], dtype=float),
            1.0e6,
            np.array([1.5e-6, 2.0e-6], dtype=float),
            0.04,
            0.0,
            2.0,
            64,
            5.0e-2,
            64,
            0.25,
            101_325.0,
            998.0,
            0.0725,
            1.4,
            1.0e-3,
            2330.0,
            1481.0,
            0.0,
        )
    )

    harmonic = np.asarray(harmonic)
    subharmonic = np.asarray(subharmonic)
    ultraharmonic = np.asarray(ultraharmonic)
    stable = np.asarray(stable)
    inertial = np.asarray(inertial)
    n_active = np.asarray(n_active)
    assert harmonic.shape == subharmonic.shape == ultraharmonic.shape == (2,)
    assert stable.shape == inertial.shape == n_active.shape == (2,)
    assert np.all(n_active == 2)
    assert np.all(np.isfinite(stable))
    assert np.all(stable >= 0.0)
    np.testing.assert_allclose(stable, subharmonic + ultraharmonic)


def test_chapter24_vs_spectrum_helper_routes_to_rust():
    source = (BOOK_DIR / "cavitation_dose_monitor.py").read_text(encoding="utf-8")
    start = source.index("def vs_emission_spectrum(")
    end = source.index("\ndef simulate_population_emission", start)
    body = source[start:end]

    assert "kw.volume_emission_spectrum" in body
    forbidden_tokens = (
        "kw.solve_keller_miksis(",
        "kw.bubble_acoustic_emission_pressure(",
        "kw.hann_windowed_power_spectrum(",
        "kw.integrate_receiver_array_psd(",
        "for r0 in np.atleast_1d",
        "_decimate(",
    )
    for token in forbidden_tokens:
        assert token not in body, token


def test_chapter24_dose_vs_pressure_helper_routes_sweep_to_rust():
    source = (BOOK_DIR / "cavitation_dose_monitor.py").read_text(encoding="utf-8")
    start = source.index("def dose_vs_pressure(")
    end = source.index("\ndef closed_loop_sonication", start)
    body = source[start:end]

    assert "kw.volume_emission_sweep" in body
    forbidden_tokens = (
        "for i, pa in enumerate(pressures_pa):",
        "vs_emission_spectrum(",
        "kw.cavitation_emission_bands(",
        "out[\"harmonic\"][i]",
        "out[\"inertial\"][i]",
    )
    for token in forbidden_tokens:
        assert token not in body, token


def test_cavitation_monitor_timeseries_binding_matches_no_jitter_contract():
    import pykwavers as kw

    t, signal, power_pct, cumulative, goal = kw.cavitation_monitor_timeseries(
        np.array([0.0, 100.0, 200.0], dtype=float),
        np.array([0.0, 10.0, 40.0], dtype=float),
        3,
        1.0,
        100.0,
        0.0,
        1000.0,
        0.0,
        0.0,
        0.5,
        7,
    )

    assert np.array_equal(np.asarray(t), np.array([0.0, 1.0, 2.0]))
    assert np.array_equal(np.asarray(signal), np.array([10.0, 10.0, 10.0]))
    assert np.array_equal(np.asarray(power_pct), np.array([25.0, 25.0, 25.0]))
    assert np.array_equal(np.asarray(cumulative), np.array([0.0, 10.0, 20.0]))
    assert goal == 10.0


def test_closed_loop_cavitation_sonication_binding_matches_contract():
    import pykwavers as kw

    pressure, stable, inertial, stable_dose, inertial_dose = (
        kw.closed_loop_cavitation_sonication(
            np.array([0.0, 100.0, 200.0], dtype=float),
            np.array([0.0, 10.0, 40.0], dtype=float),
            np.array([0.0, 1.0, 4.0], dtype=float),
            3,
            0.5,
            100.0,
            20.0,
            100.0,
            0.5,
        )
    )

    assert np.array_equal(np.asarray(pressure), np.array([100.0, 150.0, 150.0]))
    assert np.array_equal(np.asarray(stable), np.array([10.0, 25.0, 25.0]))
    assert np.array_equal(np.asarray(inertial), np.array([1.0, 2.5, 2.5]))
    assert np.array_equal(np.asarray(stable_dose), np.array([0.0, 8.75, 21.25]))
    assert np.array_equal(np.asarray(inertial_dose), np.array([0.0, 0.875, 2.125]))


def test_chapter24_closed_loop_sonication_helper_routes_trace_to_rust():
    source = (BOOK_DIR / "cavitation_dose_monitor.py").read_text(encoding="utf-8")
    start = source.index("def closed_loop_sonication(")
    end = source.index("\ndef monitor_timeseries", start)
    body = source[start:end]

    assert "kw.closed_loop_cavitation_sonication" in body
    forbidden_tokens = (
        "np.interp(p, pressures_pa, stable_power)",
        "np.interp(p, pressures_pa, inertial_power)",
        "kw.cavitation_controller_pressure(",
        "kw.cumulative_cavitation_dose(se_hist",
        "kw.cumulative_cavitation_dose(ie_hist",
    )
    for token in forbidden_tokens:
        assert token not in body, token


def test_chapter24_monitor_timeseries_helper_routes_trace_to_rust():
    source = (BOOK_DIR / "cavitation_dose_monitor.py").read_text(encoding="utf-8")
    start = source.index("def monitor_timeseries(")
    end = source.index("\ndef simulated_monitor_timeseries", start)
    body = source[start:end]

    assert "kw.cavitation_monitor_timeseries" in body
    forbidden_tokens = (
        "np.random.default_rng(seed)",
        "np.interp(p, pressures_pa, cavitation_power)",
        "kw.cavitation_controller_pressure(",
        "kw.cumulative_cavitation_dose(sig, dt)",
    )
    for token in forbidden_tokens:
        assert token not in body, token


def test_simulated_population_monitor_binding_returns_finite_trace():
    import pykwavers as kw

    (
        t,
        signal,
        power_pct,
        cumulative,
        goal,
        stable,
        broadband,
    ) = kw.simulated_population_monitor_timeseries(
        1.0e6,
        2,
        2,
        1.0,
        80.0e3,
        70.0e3,
        90.0e3,
        0.0,
        1.0e30,
        0.0,
        0.5,
        11,
        2.0e-6,
        0.05,
        2.0,
        96,
        5.0e-2,
        0.12,
        0.0,
        False,
        True,
        0.5,
        0.5,
        3.0e-9,
        0.04,
        160,
        101_325.0,
        998.0,
        1481.0,
        1.0e-3,
        0.0725,
        2330.0,
        1.4,
    )

    t = np.asarray(t)
    signal = np.asarray(signal)
    power_pct = np.asarray(power_pct)
    cumulative = np.asarray(cumulative)
    stable = np.asarray(stable)
    broadband = np.asarray(broadband)

    np.testing.assert_allclose(t, np.array([0.0, 1.0]))
    assert signal.shape == stable.shape == broadband.shape == (2,)
    assert np.all(np.isfinite(signal))
    assert np.all(signal >= 0.0)
    np.testing.assert_allclose(signal, stable + broadband)
    np.testing.assert_allclose(power_pct, np.full(2, (80.0 / 90.0) ** 2 * 100.0))
    np.testing.assert_allclose(cumulative[0], 0.0)
    np.testing.assert_allclose(cumulative[1], 0.5 * (signal[0] + signal[1]))
    np.testing.assert_allclose(goal, 0.5 * cumulative[-1])


def test_chapter24_simulated_monitor_helper_routes_trace_to_rust():
    source = (BOOK_DIR / "cavitation_dose_monitor.py").read_text(encoding="utf-8")
    start = source.index("def simulated_monitor_timeseries(")
    end = source.index("\ndef simulate_raster_pulsing", start)
    body = source[start:end]

    assert "kw.simulated_population_monitor_timeseries" in body
    forbidden_tokens = (
        "np.random.default_rng(seed)",
        "for k in range(n_pulses):",
        "simulate_population_emission(",
        "kw.cavitation_controller_pressure(",
        "kw.cumulative_cavitation_dose(sig",
    )
    for token in forbidden_tokens:
        assert token not in body, token


def test_chapter29_layout_helpers_report_skin_and_focused_bowl_clearance():
    import ch29_theranostic_fwi_platforms as ch29

    kidney = {
        "anatomy": "kidney",
        "placement_metrics": {
            "skin_contact_to_nearest_aperture_m": 0.003,
            "min_body_clearance_m": 0.003,
        },
    }
    brain = {
        "anatomy": "brain",
        "placement_metrics": {
            "skin_contact_to_nearest_aperture_m": 0.016,
            "min_body_clearance_m": 0.015,
        },
    }

    x_limits = ch29.axis_limits(
        [-0.04, 0.04],
        np.asarray([-0.11, 0.09]),
        np.asarray([-0.043]),
    )
    y_limits = ch29.axis_limits([-0.05, 0.05], np.asarray([]), np.asarray([]))

    assert "skin gap 3.0 mm" == ch29.placement_label(kidney)
    assert "focused-bowl clearance 15.0 mm" == ch29.placement_label(brain)
    assert x_limits[0] < -0.11
    assert x_limits[1] > 0.09
    assert y_limits[0] < -0.05
    assert y_limits[1] > 0.05


def test_chapter29_fig02_reconstruction_grid_starts_with_ct_context():
    import ch29_theranostic_fwi_platforms as ch29

    first_key, first_cmap, first_title = ch29.RECONSTRUCTION_FIGURE_COLUMNS[0]
    second_key = ch29.RECONSTRUCTION_FIGURE_COLUMNS[1][0]
    rendered_keys = [key for key, _, _ in ch29.RECONSTRUCTION_FIGURE_COLUMNS]
    reconstruction_keys = [key for key, _ in ch29.RECONSTRUCTION_CHANNELS]

    assert first_key == "ct_hu"
    assert first_cmap == "gray"
    assert first_title == "CT + target + tx/rx"
    assert second_key == "exposure"
    assert rendered_keys[-len(reconstruction_keys) :] == reconstruction_keys
    assert "elastic_shear_reconstruction" in reconstruction_keys
    assert all("FWI" not in title for _, title in ch29.RECONSTRUCTION_CHANNELS)
    assert all("FWI" not in title for _, _, title in ch29.RECONSTRUCTION_FIGURE_COLUMNS)


def test_chapter29_controlled_comparison_grid_starts_with_ct_context():
    import ch29_theranostic_fwi_platforms as ch29

    first_key, first_cmap, first_title = ch29.CONTROLLED_COMPARISON_COLUMNS[0]
    second_key = ch29.CONTROLLED_COMPARISON_COLUMNS[1][0]

    assert first_key == "placement_ct_hu"
    assert first_cmap == "gray"
    assert first_title == "CT + target + tx/rx"
    assert second_key == "common_target"
    assert any(key == "elastic_shear" for key, _, _ in ch29.CONTROLLED_COMPARISON_COLUMNS)


def test_chapter29_fig05_defaults_to_fig02_case_grids(monkeypatch):
    import ch29_theranostic_fwi_platforms as ch29

    for key in (
        "KWAVERS_CH29_NONLINEAR_GRID",
        "KWAVERS_CH29_BRAIN_NONLINEAR_GRID",
        "KWAVERS_CH29_KIDNEY_NONLINEAR_GRID",
        "KWAVERS_CH29_LIVER_NONLINEAR_GRID",
    ):
        monkeypatch.delenv(key, raising=False)

    default_grids = {case["name"]: ch29.nonlinear_grid_size(case) for case in ch29.CASES}

    assert default_grids == {"brain": 56, "kidney": 56, "liver": 56}

    monkeypatch.setenv("KWAVERS_CH29_NONLINEAR_GRID", "40")
    assert {case["name"]: ch29.nonlinear_grid_size(case) for case in ch29.CASES} == {
        "brain": 40,
        "kidney": 40,
        "liver": 40,
    }

    monkeypatch.setenv("KWAVERS_CH29_KIDNEY_NONLINEAR_GRID", "56")
    kidney = next(case for case in ch29.CASES if case["name"] == "kidney")

    assert ch29.nonlinear_grid_size(kidney) == 56


def test_chapter29_output_directory_can_be_overridden(monkeypatch):
    monkeypatch.setenv("KWAVERS_CH29_OUT_DIR", "D:/kwavers/target/ch29-test-output")

    import importlib
    import ch29_theranostic_fwi_platforms as ch29

    reloaded = importlib.reload(ch29)

    assert reloaded.OUT_DIR.as_posix().endswith("target/ch29-test-output")


def test_chapter29_loader_rejects_stale_nonlinear_extension_signature():
    import ch29_theranostic_fwi_platforms as ch29

    def inverse_stub():
        return None

    def current_nonlinear_stub():
        return None

    def stale_nonlinear_stub():
        return None

    class CurrentExtension:
        run_theranostic_inverse_from_ritk = staticmethod(inverse_stub)
        run_theranostic_nonlinear_3d_from_ritk = staticmethod(current_nonlinear_stub)

    class StaleExtension:
        run_theranostic_inverse_from_ritk = staticmethod(inverse_stub)
        run_theranostic_nonlinear_3d_from_ritk = staticmethod(stale_nonlinear_stub)

    current_nonlinear_stub.__text_signature__ = (
        "(ct_nifti_path, treatment_window_radius_m=0.04, min_points_per_wavelength=6.0)"
    )
    stale_nonlinear_stub.__text_signature__ = (
        "(ct_nifti_path, lesion_delta_c_m_s=-35.0)"
    )

    assert ch29.pykwavers_extension_is_current(CurrentExtension)
    assert not ch29.pykwavers_extension_is_current(StaleExtension)


def test_chapter29_nonlinear_defaults_to_histotripsy_drive(monkeypatch):
    import ch29_theranostic_fwi_platforms as ch29

    monkeypatch.delenv("KWAVERS_CH29_NONLINEAR_SOURCE_PRESSURE_PA", raising=False)
    monkeypatch.delenv("KWAVERS_CH29_NONLINEAR_FREQUENCY_HZ", raising=False)

    frequencies = {case["name"]: ch29.nonlinear_frequency_hz(case) for case in ch29.CASES}

    assert frequencies == {"brain": 650_000.0, "kidney": 500_000.0, "liver": 500_000.0}

    for case in ch29.CASES:
        pressure = ch29.nonlinear_source_pressure_pa(case)
        frequency_mhz = ch29.nonlinear_frequency_hz(case) * 1.0e-6
        mi = pressure * 1.0e-6 / np.sqrt(frequency_mhz)

        assert pressure >= float(case["pressure"])
        assert mi >= ch29.INERTIAL_MI_THRESHOLD

    monkeypatch.setenv("KWAVERS_CH29_NONLINEAR_SOURCE_PRESSURE_PA", "3.0e6")
    assert ch29.nonlinear_source_pressure_pa(ch29.CASES[0]) == 3.0e6


def test_chapter29_ct_context_draws_transducer_locations():
    import matplotlib.pyplot as plt

    import ch29_theranostic_fwi_platforms as ch29

    target = np.zeros((5, 5), dtype=bool)
    target[2, 2] = True
    result = {
        "anatomy": "kidney",
        "device_model": "focused_bowl_256_element_skin_coupled_arc",
        "element_count": 2,
        "placement_ct_hu": np.zeros((5, 5), dtype=float),
        "placement_spacing_m": (0.01, 0.01),
        "placement_target_mask": target,
        "placement_body_mask": np.ones((5, 5), dtype=bool),
        "placement_therapy_points_m": np.asarray([[-0.03, 0.02], [0.03, 0.02]], dtype=float),
        "placement_imaging_points_m": np.asarray([[0.0, 0.035]], dtype=float),
        "placement_focus_m": [0.0, 0.0],
        "placement_skin_contact_m": [0.0, 0.02],
        "placement_context_skin_gap_m": 0.003,
        "placement_metrics": {"min_body_clearance_m": 0.003},
    }

    fig, ax = plt.subplots()
    try:
        ch29.plot_placement_ct(ax, result, show_legend=False)
        scatter_offsets = [
            np.asarray(collection.get_offsets(), dtype=float)
            for collection in ax.collections
            if hasattr(collection, "get_offsets") and len(collection.get_offsets()) > 0
        ]
        offsets = np.vstack(scatter_offsets)

        assert np.any(np.all(np.isclose(offsets, [-0.03, 0.02]), axis=1))
        assert np.any(np.all(np.isclose(offsets, [0.03, 0.02]), axis=1))
        assert np.any(np.all(np.isclose(offsets, [0.0, 0.035]), axis=1))
        assert ax.get_xlim()[0] < -0.03
        assert ax.get_xlim()[1] > 0.03
    finally:
        plt.close(fig)


def test_active_book_focused_bowl_artifacts_use_generic_source_labels():
    checked_paths = [
        BOOK_DIR / "ch31_clinical_device_geometry.py",
        DOCS_DIR / "clinical_device_geometry.md",
        DOCS_DIR / "figures" / "ch31" / "metrics.json",
        DOCS_DIR / "figures" / "ch29" / "metrics.json",
    ]
    forbidden_source_identity_tokens = (
        "HistoSonics",
        "InSightec",
        "Exablate",
        "histosonics_like",
        "insightec_like",
        "brain_helmet",
        "helmet",
    )

    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_source_identity_tokens:
            assert token not in text, (path, token)

    ch31_metrics = json.loads((DOCS_DIR / "figures" / "ch31" / "metrics.json").read_text(encoding="utf-8"))
    assert ch31_metrics["inverse_results"]["brain"]["device_model"] == "transcranial_focused_bowl_projection"
    assert (
        ch31_metrics["inverse_results"]["liver"]["device_model"]
        == "focused_bowl_256_element_skin_coupled_arc"
    )
    assert (
        ch31_metrics["inverse_results"]["kidney"]["device_model"]
        == "focused_bowl_256_element_skin_coupled_arc"
    )


def test_chapter31_image_then_treat_helpers_use_value_semantic_thresholds():
    import ch31_clinical_device_geometry as ch31

    body = np.array(
        [
            [True, True, True],
            [True, True, False],
            [False, True, True],
        ],
        dtype=bool,
    )
    target = np.zeros((3, 3), dtype=bool)
    target[0, 1] = True
    target[1, 1] = True
    image = np.array(
        [
            [0.0, 9.0, 2.0],
            [1.0, 7.0, 99.0],
            [99.0, 3.0, 4.0],
        ],
        dtype=float,
    )
    disconnected_body = np.zeros((4, 4), dtype=bool)
    disconnected_body[0, 1] = True
    disconnected_body[1, 1] = True
    disconnected_body[3, 3] = True
    disconnected_target = np.zeros((4, 4), dtype=bool)
    disconnected_target[0, 1] = True
    boundary_body = np.ones((5, 5), dtype=bool)
    boundary_target = np.zeros((5, 5), dtype=bool)
    boundary_target[2, 2] = True

    display, peak = ch31._normalised_masked_display(image, body)
    threshold = ch31._equal_area_threshold(image, target, body)
    connected = ch31._target_connected_body_mask(disconnected_body, disconnected_target)
    therapy_mask = ch31._therapy_display_mask(boundary_body, boundary_target)

    assert peak == 9.0
    assert display.mask[1, 2]
    assert display.mask[2, 0]
    assert np.isclose(display[0, 1], 1.0)
    assert threshold == 7.0
    assert connected[0, 1]
    assert connected[1, 1]
    assert not connected[3, 3]
    assert therapy_mask[2, 2]
    assert not therapy_mask[0, 0]
    assert not therapy_mask[0, 2]
    assert ch31._pressure_label(2.5e6) == "2.50 MPa"
    assert ch31._pressure_label(38_000.0) == "38.0 kPa"


def test_chapter29_reconstruction_diagnostics_quantify_outside_target_sidelobes():
    import ch29_theranostic_fwi_platforms as ch29

    target = np.zeros((4, 4), dtype=bool)
    target[1, 1] = True
    body = np.ones((4, 4), dtype=bool)
    image = np.zeros((4, 4), dtype=float)
    image[1, 1] = 1.0
    image[2, 2] = 0.1
    result = {
        "target_mask": target,
        "body_mask": body,
        **{key: image for key, _ in ch29.RECONSTRUCTION_CHANNELS},
    }

    diagnostics = ch29.reconstruction_diagnostics(result)
    active = diagnostics["active_lesion_reconstruction"]

    assert np.isclose(ch29.normalized_db(image)[2, 2], -20.0)
    assert np.isclose(active["outside_peak_ratio"], 0.1)
    assert np.isclose(active["outside_peak_db"], -20.0)
    assert np.isclose(active["outside_energy_fraction"], 0.01 / 1.01)


def test_chapter29_controlled_comparison_uses_common_target_and_records_histories():
    import ch29_theranostic_fwi_platforms as ch29

    target = np.zeros((4, 4, 3), dtype=bool)
    target[1, 1, 1] = True
    target[1, 2, 1] = True
    linear_target = np.max(target, axis=2).astype(float)
    nonlinear_fusion = np.zeros_like(target, dtype=float)
    nonlinear_fusion[1, 1, 1] = 1.0
    nonlinear_fusion[2, 2, 1] = 0.5
    pressure = target.astype(float)
    pressure[3, 3, 1] = 5.0
    placement_target = np.zeros((6, 6), dtype=bool)
    placement_target[2, 2] = True
    placement_target[2, 3] = True
    linear = {
        "anatomy": "brain",
        "fused_reconstruction": linear_target,
        "elastic_shear_reconstruction": linear_target,
        "active_lesion_reconstruction": linear_target,
        "exposure": linear_target,
        "spacing_m": 1.0,
        "focus_m": (0.0, 0.0),
        "therapy_x_m": np.asarray([-1.0, 1.0]),
        "therapy_y_m": np.asarray([0.0, 0.0]),
        "placement_ct_hu": np.zeros((6, 6), dtype=float),
        "placement_spacing_m": [1.0, 1.0],
        "placement_target_mask": placement_target,
        "placement_body_mask": np.ones((6, 6), dtype=bool),
        "placement_therapy_points_m": np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=float),
        "placement_imaging_points_m": np.asarray([[0.0, 1.0]], dtype=float),
        "placement_focus_m": [0.0, 0.0],
        "placement_skin_contact_m": [0.0, 1.0],
    }
    nonlinear = {
        "anatomy": "brain",
        "target_mask": target,
        "inversion_mask": target,
        "crop_bounds_index": [0, 3, 0, 3, 0, 2],
        "source_dimensions": [4, 4, 3],
        "source_spacing_m": [1.0, 1.0, 1.0],
        "body_mask": np.ones_like(target, dtype=bool),
        "westervelt_peak_pressure_pa": pressure,
        "frequency_hz": 1.0e6,
        "source_pressure_pa": 1.0,
        "source_scale": 1.0,
        "inertial_mi_threshold": 1.9,
        "multiparameter_fwi_score": nonlinear_fusion,
        "cavitation_source_density": nonlinear_fusion,
        "reconstructed_cavitation_density": nonlinear_fusion,
        "nonlinear_fusion_score": nonlinear_fusion,
        "therapy_points_m": np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "electronic_steering_metrics": {
            "nominal_focus_index": [1, 1, 1],
            "calibration_hotspot_index": [1, 2, 1],
            "steering_focus_index": [1, 1, 1],
            "correction_grid_cells": [0, 0, 0],
            "calibration_hotspot_distance_grid_cells": 1.0,
            "steering_applied": False,
        },
        "fwi_objective_history": np.asarray([2.0, 1.25]),
        "cavitation_objective_history": np.asarray([1.0, 0.8]),
    }

    comparison = ch29.build_controlled_comparison([linear], [nonlinear])[0]

    assert comparison["common_grid_shape"] == [6, 6]
    assert comparison["fields"]["placement_ct_hu"].shape == (6, 6)
    assert np.array_equal(comparison["fields"]["common_target"], placement_target)
    assert comparison["fields"]["linear_fusion"].shape == (6, 6)
    assert comparison["fields"]["elastic_shear"].shape == (6, 6)
    assert comparison["fields"]["nonlinear_fusion"].shape == (6, 6)
    assert comparison["fields"]["nonlinear_pressure_window"].shape == (6, 6)
    assert comparison["fields"]["nonlinear_pressure_raw"].shape == (6, 6)
    assert comparison["fields"]["ct_frame_linear_fusion"].shape == (6, 6)
    assert comparison["fields"]["ct_frame_elastic_shear"].shape == (6, 6)
    assert comparison["fields"]["ct_frame_nonlinear_fusion"].shape == (6, 6)
    assert comparison["fields"]["ct_frame_common_target"].dtype == np.bool_
    assert comparison["geometry"]["comparison_frame"] == "full_ct_placement_xy_projection"
    assert comparison["geometry"]["common_target_voxels"] == 2
    assert comparison["comparison_metrics"]["linear_fusion"]["dice_equal_area"] == 1.0
    assert comparison["comparison_metrics"]["elastic_shear"]["dice_equal_area"] == 1.0
    assert comparison["comparison_metrics"]["nonlinear_pressure"]["target_peak"] == 1.0
    assert comparison["comparison_metrics"]["nonlinear_pressure"]["outside_peak"] == 0.0
    assert comparison["comparison_metrics"]["nonlinear_pressure"]["window_ct_field_target_peak"] == 1.0
    assert comparison["comparison_metrics"]["nonlinear_pressure"]["raw_peak_pressure_pa"] == 5.0
    assert (
        comparison["comparison_metrics"]["nonlinear_pressure"]["raw_ct_field_hotspot_distance_to_target_grid_cells"]
        > 0.0
    )
    assert comparison["comparison_metrics"]["nonlinear_pressure"]["ct_frame_pressure_hotspot_distance_m"] >= 0.0
    assert comparison["comparison_metrics"]["nonlinear_pressure"]["ct_frame_pressure_hotspot_cross_axis_offset_m"] >= 0.0
    assert comparison["comparison_metrics"]["nonlinear_pressure"]["electronic_steering_steering_applied"] is False
    assert comparison["geometry"]["planned_to_nonlinear_aperture_axis_angle_deg"] >= 0.0
    assert comparison["comparison_metrics"]["nonlinear_cavitation_source"]["target_peak"] == 1.0
    assert comparison["objective_history"]["nonlinear_fwi"] == [2.0, 1.25]
    assert "linear fusion Dice" in comparison["technical_explanation"]
    assert "pressure hotspot offset" in comparison["technical_explanation"]


def test_chapter29_controlled_comparison_linear_extent_uses_crop_metadata():
    import ch29_controlled_comparison as comparison

    result = {
        "fused_reconstruction": np.zeros((3, 3), dtype=float),
        "spacing_m": 1.0,
        "source_dimensions": [7, 5],
        "source_spacing_m": [0.002, 0.003],
        "crop_bounds_index": [2, 4, 1, 3],
    }

    assert comparison._linear_extent(result) == [-0.002, 0.002, -0.003, 0.003]


def test_chapter29_pressure_hotspot_metrics_project_onto_beam_axis():
    import ch29_pressure_localization as localization

    target = np.zeros((5, 5), dtype=bool)
    target[2, 2] = True
    pressure = np.zeros((5, 5), dtype=float)
    pressure[3, 2] = 2.0
    geometry = {"planned_beam_axis_unit": [1.0, 0.0]}

    metrics = localization.pressure_hotspot_physical_metrics(pressure, target, [-2.0, 2.0, -2.0, 2.0], geometry)

    assert metrics["ct_frame_target_centroid_m"] == [0.0, 0.0]
    assert metrics["ct_frame_pressure_hotspot_m"] == [1.0, 0.0]
    assert np.isclose(metrics["ct_frame_pressure_hotspot_distance_m"], 1.0)
    assert np.isclose(metrics["ct_frame_pressure_hotspot_axis_offset_m"], 1.0)
    assert np.isclose(metrics["ct_frame_pressure_hotspot_cross_axis_offset_m"], 0.0)
    assert metrics["pressure_hotspot_is_postfocal"]
    assert not metrics["pressure_hotspot_is_prefocal"]


def test_chapter29_pressure_diagnostics_accept_projected_2d_fields():
    import ch29_pressure_diagnostics as diagnostics

    pressure = np.zeros((3, 3), dtype=float)
    pressure[1, 2] = 2.0
    target = np.zeros((3, 3), dtype=bool)
    target[1, 1] = True

    metrics = diagnostics.pressure_field_diagnostics(
        pressure,
        target,
        body_mask=np.ones_like(target, dtype=bool),
        frequency_hz=1.0e6,
        source_pressure_pa=1.0,
        source_scale=1.0,
        inertial_mi_threshold=1.9,
    )

    assert metrics["raw_peak_pressure_pa"] == 2.0
    assert metrics["peak_mechanical_index"] == 2.0e-6
    assert metrics["raw_hotspot_x_index"] == 1.0
    assert metrics["raw_hotspot_y_index"] == 2.0
    assert metrics["raw_hotspot_z_index"] == 0.0
    assert metrics["target_centroid_z_index"] == 0.0


def test_chapter29_pressure_diagnostics_uses_rust_mechanical_index():
    source = (BOOK_DIR / "ch29_pressure_diagnostics.py").read_text(
        encoding="utf-8", errors="ignore"
    )

    forbidden = [
        token
        for token in (
            "_HAS_PYKWAVERS",
            "except ImportError",
            "kw = None",
            "pressure_pa * 1.0e-6",
            "np.sqrt(frequency_mhz)",
        )
        if token in source
    ]

    assert forbidden == []
    assert "import pykwavers as kw" in source
    assert "kw.mechanical_index" in source


def test_chapter29_nonlinear_projection_expands_to_full_ct_frame():
    import ch29_theranostic_fwi_platforms as ch29

    crop = np.zeros((2, 2), dtype=float)
    crop[0, 0] = 1.0

    projected = ch29.project_to_ct_frame(
        crop,
        [-0.5, 0.5, -0.5, 0.5],
        (4, 4),
        [-1.5, 1.5, -1.5, 1.5],
    )
    projected_mask = ch29.project_to_ct_frame(
        crop,
        [-0.5, 0.5, -0.5, 0.5],
        (4, 4),
        [-1.5, 1.5, -1.5, 1.5],
        binary=True,
    )

    assert projected.shape == (4, 4)
    assert projected[1, 1] == 1.0
    assert np.count_nonzero(projected) == 1
    assert projected_mask.dtype == np.bool_
    assert projected_mask[1, 1]


def test_chapter29_fig05_westervelt_panel_masks_to_target_support():
    import ch29_theranostic_fwi_platforms as ch29

    target = np.zeros((3, 3, 3), dtype=bool)
    target[1, 1, 1] = True
    pressure = np.zeros_like(target, dtype=float)
    pressure[1, 1, 1] = 2.0
    pressure[2, 2, 1] = 9.0

    display = ch29.nonlinear_target_pressure_volume({
        "target_mask": target,
        "body_mask": np.ones_like(target, dtype=bool),
        "westervelt_peak_pressure_pa": pressure,
    })

    assert display[1, 1, 1] == 2.0
    assert display[2, 2, 1] == 0.0
    assert float(np.max(display)) == 2.0


def test_chapter29_fig05_uses_nonlinear_beams_on_pressure_panels(monkeypatch):
    import ch29_theranostic_fwi_platforms as ch29

    target = np.zeros((3, 3, 3), dtype=bool)
    target[1, 1, 1] = True
    nonlinear_points = np.asarray([[0.20, 0.30, 0.0], [0.25, 0.35, 0.0]], dtype=float)
    planned_points = np.asarray([[-0.20, -0.30, 0.0], [-0.25, -0.35, 0.0]], dtype=float)
    volume = target.astype(float)
    result = {
        "anatomy": "liver",
        "target_mask": target,
        "body_mask": np.ones_like(target, dtype=bool),
        "ct_hu": np.zeros_like(volume),
        "westervelt_peak_pressure_pa": volume,
        "multiparameter_fwi_score": volume,
        "reconstructed_delta_beta": volume,
        "cavitation_source_density": volume,
        "reconstructed_cavitation_density": volume,
        "nonlinear_fusion_score": volume,
        "therapy_points_m": nonlinear_points,
        "crop_bounds_index": [0, 2, 0, 2, 0, 2],
        "source_dimensions": [3, 3, 3],
        "source_spacing_m": [1.0, 1.0, 1.0],
        "wavelength_min_m": 1.0,
        "points_per_wavelength_min": 3.0,
        "min_points_per_wavelength": 2.0,
        "resolution_meets_min_ppw": True,
        "metrics": {
            "fwi": {"dice_equal_area": 1.0},
            "rayleigh_plesset_cavitation": {"dice_equal_area": 1.0},
            "fusion": {"dice_equal_area": 1.0},
        },
    }
    placement_target = np.zeros((5, 5), dtype=bool)
    placement_target[2, 2] = True
    placement = {
        "anatomy": "liver",
        "placement_ct_hu": np.zeros((5, 5), dtype=float),
        "placement_spacing_m": [1.0, 1.0],
        "placement_target_mask": placement_target,
        "placement_body_mask": np.ones((5, 5), dtype=bool),
        "placement_therapy_points_m": planned_points,
        "placement_imaging_points_m": np.asarray([[0.0, 0.0]], dtype=float),
        "placement_focus_m": [0.0, 0.0],
        "placement_skin_contact_m": [0.0, -1.0],
        "lesion_target": placement_target.astype(float),
        "exposure": placement_target.astype(float),
        "placement_metrics": {"min_body_clearance_m": 0.0},
    }
    calls = []

    def capture_beams(ax, therapy_points, focus):
        calls.append((np.asarray(therapy_points, dtype=float).copy(), np.asarray(focus, dtype=float).copy()))

    monkeypatch.setattr(ch29, "plot_beam_paths_2d", capture_beams)
    monkeypatch.setattr(ch29, "save_figure", lambda fig, path, dpi=None: None)

    ch29.render_nonlinear_3d([result], [placement])

    assert np.array_equal(calls[0][0], planned_points)
    assert np.array_equal(calls[1][0], planned_points[:, :2])
    assert np.array_equal(calls[2][0], nonlinear_points[:, :2])

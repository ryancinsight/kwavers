from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))

import ch30_intravascular_ultrasound as ivus  # noqa: E402


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

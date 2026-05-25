import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _examples_root() -> Path:
    return Path(__file__).resolve().parents[1] / "examples"


def _load_example():
    module_path = _examples_root() / "replicate_ali2025_breast_fwi.py"
    spec = importlib.util.spec_from_file_location("replicate_ali2025_breast_fwi", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["replicate_ali2025_breast_fwi"] = module
    spec.loader.exec_module(module)
    return module


def _load_support_module(name: str):
    root = _examples_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(f"ali2025_breast_fwi.{name}")


def _load_pykwavers():
    return importlib.import_module("pykwavers")


def test_center_crop_and_decimation_are_value_exact():
    kw = _load_pykwavers()
    volume = np.arange(6 * 8 * 4, dtype=np.float64).reshape(6, 8, 4) + 1400.0

    decimated = kw.prepare_breast_fwi_reduced_phantom(
        volume,
        5.0e-4,
        (10, 10, 10),
        2,
        "/phantom/sound_speed_m_s",
        "fixture.mat",
    )["sound_speed_m_s"]
    np.testing.assert_array_equal(decimated, volume[::2, ::2, ::2])

    cropped = kw.prepare_breast_fwi_reduced_phantom(
        volume,
        5.0e-4,
        (4, 2, 2),
        1,
        "/phantom/sound_speed_m_s",
        "fixture.mat",
    )["sound_speed_m_s"]
    np.testing.assert_array_equal(cropped, volume[1:5, 3:5, 1:3])


def test_prepare_reduced_phantom_preserves_spacing_and_initial_model():
    kw = _load_pykwavers()
    source = np.linspace(1450.0, 1550.0, 6 * 6 * 4, dtype=np.float64).reshape(6, 6, 4)

    reduced = kw.prepare_breast_fwi_reduced_phantom(
        source,
        5.0e-4,
        (2, 2, 2),
        2,
        "/phantom/sound_speed_m_s",
        "fixture.mat",
    )

    assert reduced["original_shape"] == (6, 6, 4)
    assert reduced["reduced_shape"] == (2, 2, 2)
    assert reduced["source_spacing_m"] == 5.0e-4
    assert reduced["effective_spacing_m"] == 1.0e-3
    assert reduced["dataset_path"] == "/phantom/sound_speed_m_s"
    assert np.all(reduced["initial_sound_speed_m_s"] == np.median(reduced["sound_speed_m_s"]))


def test_reconstruction_metrics_identity_and_affine_relation():
    module = _load_support_module("metrics")
    reference = np.arange(27, dtype=np.float64).reshape(3, 3, 3) + 1450.0
    shifted = reference + 5.0

    identity = module.reconstruction_metrics(reference, reference.copy())
    shifted_metrics = module.reconstruction_metrics(reference, shifted)

    assert identity["rmse_m_s"] == 0.0
    assert identity["normalized_rmse"] == 0.0
    assert abs(identity["pearson_correlation"] - 1.0) < 1.0e-12
    assert abs(shifted_metrics["rmse_m_s"] - 5.0) < 1.0e-12
    assert abs(shifted_metrics["pearson_correlation"] - 1.0) < 1.0e-12


def test_table1_parity_uses_published_3d_fwi_thresholds():
    module = _load_support_module("metrics")
    metrics = {"rmse_m_s": 31.0, "pearson_correlation": 0.8848 * 0.95}

    parity = module.table1_parity(metrics, phantom_index=1, rmse_multiplier=2.0, pcc_fraction=0.95)

    assert parity["table1_3d_rmse_m_s"] == 15.5
    assert parity["table1_3d_pearson_correlation"] == 0.8848
    assert parity["rmse_threshold_m_s"] == 31.0
    assert parity["pcc_threshold"] == 0.8848 * 0.95
    assert parity["passes"] is True

    failed = module.table1_parity(
        {"rmse_m_s": 31.01, "pearson_correlation": 0.1},
        phantom_index=1,
        rmse_multiplier=2.0,
        pcc_fraction=0.95,
    )
    assert failed["passes"] is False
    with pytest.raises(ValueError, match="phantom_index"):
        module.table1_parity(metrics, phantom_index=4, rmse_multiplier=2.0, pcc_fraction=0.95)


def test_metrics_reject_invalid_domains():
    module = _load_support_module("metrics")
    reference = np.ones((2, 2, 2), dtype=np.float64) * 1500.0
    estimate = reference.copy()

    with pytest.raises(ValueError, match="nonconstant"):
        module.pearson_correlation(reference, estimate)
    with pytest.raises(ValueError, match="shape mismatch"):
        module.rmse_m_s(reference, estimate[:, :, :1])
    with pytest.raises(ValueError, match="strictly positive"):
        module.rmse_m_s(reference, np.zeros_like(reference))


def test_parse_contracts_and_reduced_geometry():
    module = _load_example()
    kw = _load_pykwavers()

    assert module.parse_shape("4,5,6") == (4, 5, 6)
    assert module.parse_frequency_list("200000, 300000") == [200000.0, 300000.0]
    with pytest.raises(argparse.ArgumentTypeError):
        module.parse_shape("4,5")
    with pytest.raises(argparse.ArgumentTypeError):
        module.parse_frequency_list("0")

    geometry = kw.derive_breast_fwi_reduced_array_geometry(
        (10, 12, 4),
        1.0e-3,
        2,
        None,
        None,
    )
    diameter = geometry["diameter_m"]
    row_spacing = geometry["row_spacing_m"]
    assert abs(diameter - 0.80 * 9.0e-3) < 1.0e-15
    assert row_spacing > 0.0
    plan = kw.derive_breast_fwi_reduced_array_plan(
        (24, 24, 12),
        3.2e-3,
        "table1_parity_interior",
        None,
        None,
        None,
    )
    assert plan["rows"] == 10
    assert plan["row_policy"] == "table1_parity_interior"
    assert plan["row_spacing_m"] == 3.2e-3
    with pytest.raises(ValueError, match="no larger than"):
        kw.derive_breast_fwi_reduced_array_geometry((10, 10, 4), 1.0e-3, 1, 0.1, None)
    with pytest.raises(ValueError, match="unknown breast FWI reduced-array row_policy"):
        kw.derive_breast_fwi_reduced_array_plan((10, 10, 4), 1.0e-3, "invalid", None, None, None)
    pstd_spectral = kw.FrequencyDomainFwiConfig(
        propagation_model="pstd_spectral_convergent_born",
        pstd_time_step_s=1.0e-7,
        absorbing_boundary="polynomial",
        absorbing_thickness_cells=0,
    )
    assert pstd_spectral.propagation_model == "pstd_spectral_convergent_born"
    assert (
        kw.FrequencyDomainFwiConfig.spectral_convergent_born(
            absorbing_thickness_cells=0,
        ).propagation_model
        == "spectral_convergent_born"
    )
    assert (
        kw.FrequencyDomainFwiConfig.pstd_spectral_convergent_born(
            absorbing_thickness_cells=0,
        ).propagation_model
        == "pstd_spectral_convergent_born"
    )


def test_homogeneous_direct_field_reports_passive_residual_deltas():
    kw = _load_pykwavers()
    model = np.full((12, 12, 3), 1482.0, dtype=np.float64)
    array = kw.MultiRowRingArray(4, 1, 0.006, 0.0)
    config = kw.BreastFwiPstdDatasetConfig(
        1.0e-3,
        1.0e-7,
        1,
        1,
        1.0e3,
        1000.0,
        0,
    )

    diagnostics = dict(
        kw.diagnose_breast_fwi_homogeneous_direct_field(
            model,
            array,
            [200_000.0],
            config,
        )
    )
    source_kappa = dict(diagnostics["source_kappa_filtered"])
    pstd_periodic = dict(diagnostics["pstd_periodic"])

    assert np.isfinite(diagnostics["source_kappa_filtered_passive_residual_delta"])
    assert np.isfinite(diagnostics["pstd_periodic_passive_residual_delta"])
    assert (
        abs(
            diagnostics["source_kappa_filtered_passive_residual_delta"]
            - (
                source_kappa["passive_only_normalized_l2_residual"]
                - diagnostics["passive_only_normalized_l2_residual"]
            )
        )
        <= 1.0e-15
    )
    assert (
        abs(
            diagnostics["pstd_periodic_passive_residual_delta"]
            - (
                pstd_periodic["passive_only_normalized_l2_residual"]
                - diagnostics["passive_only_normalized_l2_residual"]
            )
        )
        <= 1.0e-15
    )


def test_orthographic_slices_use_center_planes():
    module = _load_support_module("volume")
    volume = np.arange(5 * 7 * 3, dtype=np.float64).reshape(5, 7, 3) + 1400.0
    sx, sy, sz = module.orthographic_slices(volume)

    np.testing.assert_array_equal(sx, volume[2, :, :])
    np.testing.assert_array_equal(sy, volume[:, 3, :])
    np.testing.assert_array_equal(sz, volume[:, :, 1])


def test_identifiability_reports_reduced_probe_rank_bound():
    module = _load_support_module("identifiability")

    report = module.acquisition_identifiability(
        (8, 8, 4),
        [200_000.0],
        transmissions=4,
        receivers=4,
        source_scaling_policy=module.SourceScalingPolicy.ESTIMATED,
    )

    assert report["unknown_voxels"] == 256
    assert report["complex_observations"] == 16
    assert report["real_observation_dof"] == 32
    assert report["estimated_source_scale_real_dof"] == 8
    assert report["informative_real_dof_upper_bound"] == 24
    assert report["informative_dof_to_unknown_ratio"] == 24 / 256
    assert report["underdetermined_by_rank_upper_bound"] is True
    with pytest.raises(ValueError, match="rank-underdetermined"):
        module.require_determined_acquisition(report)


def test_identifiability_accepts_determined_reduced_probe():
    module = _load_support_module("identifiability")

    report = module.acquisition_identifiability(
        (4, 3, 2),
        [200_000.0],
        transmissions=4,
        receivers=4,
        source_scaling_policy=module.SourceScalingPolicy.ESTIMATED,
    )

    assert report["unknown_voxels"] == 24
    assert report["informative_real_dof_upper_bound"] == 24
    assert report["underdetermined_by_rank_upper_bound"] is False
    module.require_determined_acquisition(report)


def test_forward_consistency_recovers_row_source_scale():
    module = _load_support_module("forward_consistency")
    predicted = np.array(
        [[[1.0 + 0.0j, 0.0 + 2.0j], [2.0 - 1.0j, 0.5 + 0.25j]]],
        dtype=np.complex128,
    )
    scale = 3.0 - 2.0j
    observed = scale * predicted

    metrics = module.scaled_observation_residual_metrics(predicted, observed)

    assert metrics["row_count"] == 2
    assert metrics["normalized_l2_residual"] <= 1.0e-14
    assert abs(metrics["source_scale_magnitude_min"] - abs(scale)) <= 1.0e-14
    assert abs(metrics["source_scale_magnitude_max"] - abs(scale)) <= 1.0e-14


def test_forward_consistency_source_receiver_mask_matches_cylindrical_topology():
    module = _load_support_module("forward_consistency")

    mask = module.source_receiver_mask((2, 4, 8), circumferential_elements=4, rows=2)
    passive = module.passive_receiver_mask((2, 4, 8), circumferential_elements=4, rows=2)

    assert int(np.count_nonzero(mask)) == 2 * 4 * 2
    assert int(np.count_nonzero(passive)) == 2 * 4 * 6
    for frequency_index in range(2):
        for transmit_index in range(4):
            active_indices = np.flatnonzero(mask[frequency_index, transmit_index])
            np.testing.assert_array_equal(active_indices, [transmit_index, transmit_index + 4])
    np.testing.assert_array_equal(np.logical_not(mask), passive)


def test_forward_consistency_passive_mask_isolates_source_channel_mismatch():
    module = _load_support_module("forward_consistency")
    predicted = np.array(
        [[[1.0 + 0.0j, 2.0 - 1.0j, 3.0 + 0.5j, 4.0 + 1.5j],
          [2.0 + 0.5j, 5.0 - 0.5j, 7.0 + 0.25j, 11.0 - 1.0j]]],
        dtype=np.complex128,
    )
    scale = 2.0 + 0.5j
    observed = scale * predicted
    active_mask = module.source_receiver_mask(predicted.shape, 2, 2)
    passive_mask = np.logical_not(active_mask)
    observed = observed.copy()
    observed[active_mask] += 10.0 - 3.0j

    all_metrics = module.scaled_observation_residual_metrics(predicted, observed)
    passive_metrics = module.scaled_observation_residual_metrics(
        predicted,
        observed,
        passive_mask,
    )
    diagnostics = module.source_channel_residual_diagnostics(predicted, observed, 2, 2)

    assert all_metrics["normalized_l2_residual"] > 0.1
    assert passive_metrics["normalized_l2_residual"] <= 1.0e-14
    assert diagnostics["passive_only_normalized_l2_residual"] <= 1.0e-14
    assert diagnostics["active_receiver_count_per_row"] == 2
    assert diagnostics["passive_receiver_count_per_row"] == 2
    assert diagnostics["active_full_scale_residual_energy_fraction"] > 0.0
    energy_fraction_sum = (
        diagnostics["active_full_scale_residual_energy_fraction"]
        + diagnostics["passive_full_scale_residual_energy_fraction"]
    )
    assert abs(energy_fraction_sum - 1.0) <= 1.0e-12


def test_forward_consistency_rejects_shape_and_zero_energy_defects():
    module = _load_support_module("forward_consistency")
    observed = np.ones((1, 1, 2), dtype=np.complex128)

    with pytest.raises(ValueError, match="shape mismatch"):
        module.scaled_observation_residual_metrics(np.ones((1, 2, 2)), observed)
    with pytest.raises(ValueError, match="zero energy"):
        module.scaled_observation_residual_metrics(np.zeros((1, 1, 2)), observed)
    with pytest.raises(ValueError, match="at least one receiver"):
        module.scaled_observation_residual_metrics(
            np.ones((1, 1, 2), dtype=np.complex128),
            observed,
            np.zeros((1, 1, 2), dtype=bool),
        )


def test_source_excitation_reports_uniform_scalar_source_contract():
    module = _load_support_module("source_excitation")
    predicted = np.array(
        [
            [[1.0 + 0.0j, 2.0 + 0.5j], [3.0 - 1.0j, 4.0 + 0.25j]],
            [[2.0 + 1.0j, 1.0 - 0.5j], [5.0 + 0.5j, 7.0 - 0.75j]],
        ],
        dtype=np.complex128,
    )
    source_amplitude = 5.0
    frequencies = [100.0, 125.0]
    time_steps = [40, 40]
    bin_starts = [20, 20]
    observed = predicted.copy()
    for frequency_index, frequency_hz in enumerate(frequencies):
        coeff = module.sine_frequency_bin_coefficient(frequency_hz, 0.001, 40, 20)
        observed[frequency_index] *= source_amplitude * coeff * (2.0 + frequency_index)

    diagnostics = module.source_excitation_diagnostics(
        predicted,
        observed,
        frequencies,
        source_amplitude,
        0.001,
        time_steps,
        bin_starts,
    )

    assert diagnostics["frequency_count"] == 2
    assert diagnostics["transmission_count"] == 2
    assert diagnostics["max_source_scale_magnitude_coefficient_of_variation"] <= 1.0e-14
    assert diagnostics["max_source_scale_phase_circular_variance"] <= 1.0e-14
    assert diagnostics["max_source_scale_phase_span_rad"] <= 1.0e-14


def test_source_excitation_detects_transmit_scale_dispersion():
    module = _load_support_module("source_excitation")
    predicted = np.array(
        [[[1.0 + 0.0j, 2.0 + 0.5j], [3.0 - 1.0j, 4.0 + 0.25j]]],
        dtype=np.complex128,
    )
    coeff = module.sine_frequency_bin_coefficient(100.0, 0.001, 40, 20)
    observed = predicted.copy()
    observed[0, 0, :] *= 4.0 * coeff
    observed[0, 1, :] *= 8.0j * coeff

    diagnostics = module.source_excitation_diagnostics(
        predicted,
        observed,
        [100.0],
        4.0,
        0.001,
        [40],
        [20],
    )

    assert diagnostics["max_source_scale_magnitude_coefficient_of_variation"] > 0.3
    assert diagnostics["max_source_scale_phase_span_rad"] > 1.0
    with pytest.raises(ValueError, match="length must match"):
        module.source_excitation_diagnostics(predicted, observed, [], 4.0, 0.001, [40], [20])


def test_operator_equivalence_selects_lowest_residual_model():
    excitation = _load_support_module("source_excitation")
    operators = _load_support_module("operator_equivalence")
    accurate = np.array(
        [[[1.0 + 0.0j, 2.0 + 0.5j], [3.0 - 1.0j, 4.0 + 0.25j]]],
        dtype=np.complex128,
    )
    distorted = accurate.copy()
    distorted[0, :, 1] *= np.array([0.25 + 0.0j, -0.5 + 1.0j])
    coefficient = excitation.sine_frequency_bin_coefficient(100.0, 0.001, 40, 20)
    observed = 4.0 * coefficient * accurate

    diagnostics = operators.operator_equivalence_diagnostics(
        {"distorted": distorted, "accurate": accurate},
        observed,
        [100.0],
        4.0,
        0.001,
        [40],
        [20],
    )

    assert diagnostics["model_count"] == 2
    assert diagnostics["receiver_channel_policy"] == "all"
    assert diagnostics["best_model"] == "accurate"
    assert diagnostics["best_normalized_l2_residual"] <= 1.0e-14
    assert diagnostics["worst_model"] == "distorted"
    assert diagnostics["residual_spread"] > 0.0
    with pytest.raises(ValueError, match="must not be empty"):
        operators.operator_equivalence_diagnostics({}, observed, [100.0], 4.0, 0.001, [40], [20])


def test_operator_equivalence_receiver_policy_changes_ranking():
    operators = _load_support_module("operator_equivalence")
    observed = np.ones((1, 2, 4), dtype=np.complex128)
    active_distorted = observed.copy()
    passive_distorted = observed.copy()
    for transmit in range(2):
        for receiver in range(4):
            distortion = 2.0j if receiver // 2 == 0 else 3.0 + 0.0j
            if receiver % 2 == transmit:
                active_distorted[0, transmit, receiver] = distortion
            else:
                passive_distorted[0, transmit, receiver] = distortion

    passive = operators.operator_equivalence_diagnostics(
        {
            "active_distorted": active_distorted,
            "passive_distorted": passive_distorted,
        },
        observed,
        [100.0],
        1.0,
        0.001,
        [40],
        [20],
        operators.ReceiverChannelPolicy.PASSIVE_ONLY,
    )
    active = operators.operator_equivalence_diagnostics(
        {
            "active_distorted": active_distorted,
            "passive_distorted": passive_distorted,
        },
        observed,
        [100.0],
        1.0,
        0.001,
        [40],
        [20],
        "active_only",
    )

    assert passive["receiver_channel_policy"] == "passive_only"
    assert passive["best_model"] == "active_distorted"
    assert passive["best_normalized_l2_residual"] <= 1.0e-14
    assert active["receiver_channel_policy"] == "active_only"
    assert active["best_model"] == "passive_distorted"
    assert active["best_normalized_l2_residual"] <= 1.0e-14


def test_scattering_increment_diagnostics_identify_exact_increment_model():
    operators = _load_support_module("operator_equivalence")
    baseline = np.ones((1, 2, 4), dtype=np.complex128)
    increment = np.array(
        [
            [
                [1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 2.0j, 0.0 - 2.0j],
                [0.5 + 0.0j, -0.5 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j],
            ]
        ],
        dtype=np.complex128,
    )
    scale = 2.0 - 0.5j
    observed = scale * baseline + increment
    exact = baseline + increment / scale
    half = baseline + increment / (2.0 * scale)

    diagnostics = operators.scattering_increment_diagnostics(
        baseline,
        {"baseline": baseline, "half_increment": half, "exact_increment": exact},
        observed,
    )

    assert diagnostics["model_count"] == 3
    assert diagnostics["receiver_channel_policy"] == "all"
    assert diagnostics["best_model"] == "exact_increment"
    assert diagnostics["best_normalized_increment_residual"] <= 1.0e-14
    by_model = {row["model"]: row for row in diagnostics["per_model"]}
    assert abs(by_model["baseline"]["normalized_increment_residual"] - 1.0) <= 1.0e-14
    assert abs(by_model["half_increment"]["normalized_increment_residual"] - 0.5) <= 1.0e-14


def test_operator_prediction_builder_uses_all_models_and_frequencies():
    operators = _load_support_module("operator_equivalence")

    class FakeKw:
        @staticmethod
        def simulate_breast_fwi_frequency_observation(_model, _array, frequency_hz, config):
            return np.full((2, 2), frequency_hz * config, dtype=np.complex128)

    predictions = operators.simulate_forward_predictions(
        FakeKw,
        np.ones((1, 1, 1), dtype=np.float64),
        object(),
        [2.0, 3.0],
        {"a": 10.0, "b": 20.0},
    )

    assert sorted(predictions) == ["a", "b"]
    np.testing.assert_array_equal(predictions["a"][:, 0, 0], [20.0 + 0.0j, 30.0 + 0.0j])
    np.testing.assert_array_equal(predictions["b"][:, 0, 0], [40.0 + 0.0j, 60.0 + 0.0j])
    with pytest.raises(ValueError, match="must not be empty"):
        operators.simulate_forward_predictions(FakeKw, np.ones((1, 1, 1)), object(), [2.0], {})

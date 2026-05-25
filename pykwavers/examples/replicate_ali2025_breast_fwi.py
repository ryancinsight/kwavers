"""Reduced-grid Ali 2025 breast UST FWI replication entry point.

This script keeps the clinical replication path on the production bindings:
Rust decodes the published MATLAB-5 MRI phantom or HDF5 sound-speed phantom,
Rust reduces the clinical domain and plans the array geometry, Rust generates
PSTD frequency-domain data, and Rust runs the frequency-domain FWI solver.
Python owns CLI orchestration, report serialization, and visualization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ali2025_breast_fwi.identifiability import (
    SourceScalingPolicy,
    acquisition_identifiability,
    require_determined_acquisition,
)
from ali2025_breast_fwi.direct_field import homogeneous_direct_field_probe
from ali2025_breast_fwi.metrics import reconstruction_metrics, table1_parity
from ali2025_breast_fwi.operator_equivalence import (
    ReceiverChannelPolicy,
    make_configs_by_model,
    operator_equivalence_diagnostics,
    scattering_increment_diagnostics,
    simulate_forward_predictions,
)
from ali2025_breast_fwi.visualization import write_orthographic_plot


ROOT = Path(__file__).resolve().parents[2]
PYKWAVERS_PACKAGE_ROOT = ROOT / "pykwavers" / "python"
PHANTOM_URL = (
    "https://github.com/rehmanali1994/3D-FWI-MultiRowRingArrayUST/"
    "releases/download/v1.0.0/BreastPhantomFromMRI.mat"
)
PHANTOM_FILE_NAME = "BreastPhantomFromMRI.mat"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "ali2025_breast_fwi"


def bootstrap_pykwavers() -> Any:
    if str(PYKWAVERS_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PYKWAVERS_PACKAGE_ROOT))
    import pykwavers as kw

    return kw


def default_phantom_dir() -> Path:
    configured = os.getenv("KWAVERS_ALI2025_PHANTOM_DIR")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        return Path(r"D:\3D-FWI-MultiRowRingArrayUST\phantoms")
    return ROOT / "data" / "ali2025_breast_fwi" / "phantoms"


def resolve_phantom_path(path: Path | None, directory: Path, url: str) -> Path:
    if path is not None:
        resolved = path.expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"phantom path does not exist: {resolved}")
        return resolved

    target = directory.expanduser() / PHANTOM_FILE_NAME
    if target.exists() and target.stat().st_size > 0:
        return target
    return download_phantom(url, target)


def download_phantom(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        if partial.stat().st_size == 0:
            raise RuntimeError(f"downloaded empty phantom from {url}")
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return target


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must have form nx,ny,nz")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape entries must be integers") from exc
    if any(axis <= 0 for axis in shape):
        raise argparse.ArgumentTypeError("shape entries must be positive")
    return shape  # type: ignore[return-value]


def parse_frequency_list(value: str) -> list[float]:
    frequencies = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        frequency = float(text)
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise argparse.ArgumentTypeError("frequencies must be positive and finite")
        frequencies.append(frequency)
    if not frequencies:
        raise argparse.ArgumentTypeError("at least one frequency is required")
    return frequencies


def run_reduced_replication(args: argparse.Namespace) -> dict[str, Any]:
    kw = bootstrap_pykwavers()
    phantom_path = resolve_phantom_path(args.phantom_path, args.phantom_dir, args.phantom_url)
    phantom = kw.load_ali_2025_breast_fwi_phantom(
        str(phantom_path),
        sound_speed_dataset_path=args.dataset_path,
        spacing_m=args.spacing_m,
        sound_speed_unit=args.sound_speed_unit,
        storage_order=args.storage_order,
        file_format=args.file_format,
        mat5_output_shape=tuple(axis * args.decimation for axis in args.max_shape),
        mat5_grid_spacing_m=args.mat5_grid_spacing_m,
        mat5_breast_side=args.mat5_breast_side,
        mat5_mri_variable_name=args.mat5_mri_variable_name,
        mat5_tissue_threshold=args.mat5_tissue_threshold,
    )
    reduced = kw.prepare_breast_fwi_reduced_phantom(
        phantom["sound_speed_m_s"],
        float(phantom["spacing_m"]),
        args.max_shape,
        args.decimation,
        str(phantom["dataset_path"]),
        str(phantom["source_path"]),
    )
    reduced_sound_speed = np.asarray(reduced["sound_speed_m_s"], dtype=np.float64)
    initial_sound_speed = np.asarray(reduced["initial_sound_speed_m_s"], dtype=np.float64)
    reduced_shape = tuple(int(axis) for axis in reduced["reduced_shape"])
    original_shape = tuple(int(axis) for axis in reduced["original_shape"])
    effective_spacing_m = float(reduced["effective_spacing_m"])
    source_spacing_m = float(reduced["source_spacing_m"])
    row_policy = (
        "explicit"
        if args.rows is not None
        else "table1_parity_interior"
        if args.require_table1_parity
        else "smoke_single_ring"
    )
    geometry = kw.derive_breast_fwi_reduced_array_plan(
        reduced_shape,
        effective_spacing_m,
        row_policy,
        args.rows,
        args.diameter_m,
        args.row_spacing_m,
    )
    effective_rows = int(geometry["rows"])
    diameter_m = float(geometry["diameter_m"])
    row_spacing_m = float(geometry["row_spacing_m"])

    array = kw.MultiRowRingArray(
        args.circumferential_elements,
        effective_rows,
        diameter_m,
        row_spacing_m,
    )
    if args.snap_array_to_grid:
        array = kw.snap_breast_fwi_array_to_grid(
            array,
            reduced_shape,
            effective_spacing_m,
        )
    identifiability = acquisition_identifiability(
        reduced_shape,
        args.frequencies_hz,
        args.circumferential_elements,
        int(array.element_count),
        SourceScalingPolicy.ESTIMATED,
    )
    if args.require_determined_acquisition:
        require_determined_acquisition(identifiability)

    dataset_config = kw.BreastFwiPstdDatasetConfig(
        spacing_m=effective_spacing_m,
        time_step_s=args.time_step_s,
        cycles_per_frequency=args.cycles_per_frequency,
        frequency_bin_cycles=args.frequency_bin_cycles,
        source_amplitude_pa=args.source_amplitude_pa,
        density_kg_m3=args.density_kg_m3,
        cpml_thickness_cells=args.cpml_thickness_cells,
    )
    reference_speed = float(initial_sound_speed.ravel()[0])
    configs_by_model = make_configs_by_model(kw, args, reference_speed, effective_spacing_m)
    fwi_config = configs_by_model["pstd_spectral_convergent_born"]

    dataset = kw.generate_breast_fwi_pstd_dataset(
        reduced_sound_speed,
        array,
        args.frequencies_hz,
        dataset_config,
    )
    observed_pressure = np.asarray(dataset["observed_pressure"], dtype=np.complex128)
    forward_predictions = simulate_forward_predictions(
        kw,
        reduced_sound_speed,
        array,
        args.frequencies_hz,
        configs_by_model,
    )
    homogeneous_pstd_baseline = simulate_forward_predictions(
        kw,
        initial_sound_speed,
        array,
        args.frequencies_hz,
        {"pstd_spectral_convergent_born": fwi_config},
    )["pstd_spectral_convergent_born"]
    truth_forward = forward_predictions["pstd_spectral_convergent_born"]
    operator_equivalence = operator_equivalence_diagnostics(
        forward_predictions,
        observed_pressure,
        args.frequencies_hz,
        args.source_amplitude_pa,
        args.time_step_s,
        dataset["time_steps_per_frequency"],
        dataset["frequency_bin_start_steps_per_frequency"],
    )
    operator_equivalence_receiver_policies = {
        policy.value: operator_equivalence_diagnostics(
            forward_predictions,
            observed_pressure,
            args.frequencies_hz,
            args.source_amplitude_pa,
            args.time_step_s,
            dataset["time_steps_per_frequency"],
            dataset["frequency_bin_start_steps_per_frequency"],
            policy,
        )
        for policy in ReceiverChannelPolicy
    }
    scattering_increment = scattering_increment_diagnostics(
        homogeneous_pstd_baseline,
        forward_predictions,
        observed_pressure,
    )
    scattering_increment_receiver_policies = {
        policy.value: scattering_increment_diagnostics(
            homogeneous_pstd_baseline,
            forward_predictions,
            observed_pressure,
            policy,
        )
        for policy in ReceiverChannelPolicy
    }
    observation_diagnostics = kw.diagnose_breast_fwi_observation_pair(
        truth_forward,
        observed_pressure,
        array,
        args.frequencies_hz,
        dataset_config,
        dataset["time_steps_per_frequency"],
        dataset["frequency_bin_start_steps_per_frequency"],
    )
    forward_consistency = observation_diagnostics["forward_consistency"]
    source_channel_consistency = observation_diagnostics["source_channel_consistency"]
    source_excitation = observation_diagnostics["source_excitation"]
    homogeneous_direct_field = homogeneous_direct_field_probe(
        kw,
        initial_sound_speed,
        array,
        args.frequencies_hz,
        dataset_config,
    )
    inversion = kw.invert_breast_fwi(
        args.frequencies_hz,
        observed_pressure,
        array,
        initial_sound_speed,
        fwi_config,
    )
    reconstruction = np.asarray(inversion["sound_speed_m_s"], dtype=np.float64)
    metrics = reconstruction_metrics(reduced_sound_speed, reconstruction)
    parity = table1_parity(
        metrics,
        args.phantom_index,
        args.table1_rmse_multiplier,
        args.table1_pcc_fraction,
    )
    if args.require_table1_parity and not parity["passes"]:
        raise RuntimeError(
            "Ali 2025 Table 1 parity failed: "
            f"RMSE {metrics['rmse_m_s']:.6g} m/s <= {parity['rmse_threshold_m_s']:.6g}, "
            f"PCC {metrics['pearson_correlation']:.6g} >= {parity['pcc_threshold']:.6g}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot:
        write_orthographic_plot(
            reduced_sound_speed,
            reconstruction,
            args.output_dir / "ali2025_breast_fwi_slices.png",
        )

    report = {
        "model_family": "ali_2025_reduced_grid_breast_ust_fwi_replication",
        "phantom_url": args.phantom_url,
        "phantom_path": str(phantom_path),
        "dataset_path": str(reduced["dataset_path"]),
        "original_shape": original_shape,
        "reduced_shape": reduced_shape,
        "source_spacing_m": source_spacing_m,
        "effective_spacing_m": effective_spacing_m,
        "frequencies_hz": args.frequencies_hz,
        "array": {
            "circumferential_elements": args.circumferential_elements,
            "rows": effective_rows,
            "row_policy": str(geometry["row_policy"]),
            "diameter_m": diameter_m,
            "row_spacing_m": row_spacing_m,
            "element_count": array.element_count,
            "snapped_to_grid": bool(args.snap_array_to_grid),
        },
        "dataset": {
            "transmissions": int(dataset["transmissions"]),
            "receivers": int(dataset["receivers"]),
            "time_steps_per_frequency": list(dataset["time_steps_per_frequency"]),
            "frequency_bin_start_steps_per_frequency": list(
                dataset["frequency_bin_start_steps_per_frequency"]
            ),
            "model_family": str(dataset["model_family"]),
        },
        "inversion": {
            "objective_history": [
                float(value) for value in np.asarray(inversion["objective_history"]).ravel()
            ],
            "frequencies_used": int(inversion["frequencies_used"]),
            "transmissions_used": int(inversion["transmissions_used"]),
            "receivers_used": int(inversion["receivers_used"]),
            "model_family": str(inversion["model_family"]),
            "solver_model_family": str(inversion["solver_model_family"]),
        },
        "identifiability": identifiability,
        "forward_consistency": forward_consistency,
        "source_channel_consistency": source_channel_consistency,
        "source_excitation": source_excitation,
        "operator_equivalence": operator_equivalence,
        "operator_equivalence_receiver_policies": operator_equivalence_receiver_policies,
        "scattering_increment": scattering_increment,
        "scattering_increment_receiver_policies": scattering_increment_receiver_policies,
        "homogeneous_direct_field": homogeneous_direct_field,
        "metrics": metrics,
        "table1_parity": parity,
    }
    metrics_path = args.output_dir / "ali2025_breast_fwi_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phantom-path", type=Path, default=None)
    parser.add_argument("--phantom-dir", type=Path, default=default_phantom_dir())
    parser.add_argument("--phantom-url", default=PHANTOM_URL)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--spacing-m", type=float, default=None)
    parser.add_argument("--sound-speed-unit", default="meters_per_second")
    parser.add_argument("--storage-order", default="fortran_contiguous")
    parser.add_argument("--file-format", choices=("auto", "hdf5", "mat5"), default="auto")
    parser.add_argument("--mat5-grid-spacing-m", type=float, default=4.0e-4)
    parser.add_argument("--mat5-breast-side", choices=("left", "right"), default="right")
    parser.add_argument("--mat5-mri-variable-name", default="breast_mri")
    parser.add_argument("--mat5-tissue-threshold", type=float, default=40.0)
    parser.add_argument("--phantom-index", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--table1-rmse-multiplier", type=float, default=2.0)
    parser.add_argument("--table1-pcc-fraction", type=float, default=0.95)
    parser.add_argument("--require-table1-parity", action="store_true")
    parser.add_argument("--require-determined-acquisition", action="store_true")
    parser.add_argument("--max-shape", type=parse_shape, default=parse_shape("24,24,12"))
    parser.add_argument("--decimation", type=int, default=8)
    parser.add_argument("--frequencies-hz", type=parse_frequency_list, default=[200_000.0])
    parser.add_argument("--circumferential-elements", type=int, default=4)
    parser.add_argument("--rows", type=int, default=None,
                        help="ring rows; when omitted, Rust selects smoke or Table 1 parity policy")
    parser.add_argument("--diameter-m", type=float, default=None)
    parser.add_argument("--row-spacing-m", type=float, default=None)
    parser.add_argument("--snap-array-to-grid", dest="snap_array_to_grid", action="store_true", default=True)
    parser.add_argument("--no-snap-array-to-grid", dest="snap_array_to_grid", action="store_false")
    parser.add_argument("--time-step-s", type=float, default=1.0e-7)
    parser.add_argument("--cycles-per-frequency", type=int, default=1)
    parser.add_argument("--frequency-bin-cycles", type=int, default=1)
    parser.add_argument("--source-amplitude-pa", type=float, default=1.0e3)
    parser.add_argument("--density-kg-m3", type=float, default=1000.0)
    parser.add_argument("--cpml-thickness-cells", type=int, default=0)
    parser.add_argument("--fwi-iterations", type=int, default=5)
    parser.add_argument("--initial-step-s-per-m", type=float, default=5.0e-5)
    parser.add_argument("--min-sound-speed-m-s", type=float, default=1350.0)
    parser.add_argument("--max-sound-speed-m-s", type=float, default=1700.0)
    parser.add_argument("--tikhonov-weight", type=float, default=0.0)
    parser.add_argument("--cbs-iterations", type=int, default=64)
    parser.add_argument("--cbs-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--absorbing-thickness-cells", type=int, default=0)
    parser.add_argument("--absorbing-strength-nepers", type=float, default=1.5)
    parser.add_argument("--absorbing-order", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot", dest="plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_reduced_replication(args)
    print(
        json.dumps(
            {
                "metrics": report["metrics"],
                "table1_parity": report["table1_parity"],
                "identifiability": report["identifiability"],
                "forward_consistency": report["forward_consistency"],
                "source_channel_consistency": report["source_channel_consistency"],
                "source_excitation": report["source_excitation"],
                "operator_equivalence": report["operator_equivalence"],
                "operator_equivalence_receiver_policies": report[
                    "operator_equivalence_receiver_policies"
                ],
                "homogeneous_direct_field": report["homogeneous_direct_field"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

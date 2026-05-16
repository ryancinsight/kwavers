"""Chapter 29: same-device therapy, finite-frequency inverse, and RTM monitoring.

The computation is owned by kwavers through the PyO3 wrapper
``run_theranostic_inverse_from_ritk``. Python only selects the public CT/NIfTI
inputs, runs the wrapper, and renders figures.
"""

from __future__ import annotations

import json
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import numpy as np

from ch29_controlled_comparison import (
    build_controlled_comparison,
    controlled_comparison_payload,
    render_controlled_comparison,
    write_controlled_comparison_fields,
    write_controlled_comparison_metrics,
)


BOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BOOK_DIR.parents[2]
OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch29"
PY_PACKAGE = REPO_ROOT / "pykwavers" / "python"

if "PYKWAVERS_EXTENSION_PATH" not in os.environ:
    for candidate in (
        REPO_ROOT / "target" / "release" / "pykwavers.dll",
        REPO_ROOT / "target" / "maturin" / "pykwavers.dll",
        REPO_ROOT / "target" / "debug" / "pykwavers.dll",
    ):
        if candidate.exists():
            os.environ["PYKWAVERS_EXTENSION_PATH"] = str(candidate)
            break
if str(BOOK_DIR) not in sys.path:
    sys.path.insert(0, str(BOOK_DIR))
if str(PY_PACKAGE) not in sys.path:
    sys.path.insert(0, str(PY_PACKAGE))

from transcranial_planning.scene import CANONICAL_BRAIN_SCENE, BrainSceneDefinition  # noqa: E402


def load_pykwavers_extension():
    for module_name in ("_pykwavers", "pykwavers._pykwavers", "pykwavers"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "run_theranostic_inverse_from_ritk"):
            return module
    extension_path = os.environ.get("PYKWAVERS_EXTENSION_PATH")
    if extension_path and Path(extension_path).exists():
        loader = importlib.machinery.ExtensionFileLoader("_pykwavers", extension_path)
        spec = importlib.util.spec_from_file_location("_pykwavers", extension_path, loader=loader)
        if spec is None:
            raise ImportError(f"cannot load pykwavers extension from {extension_path}")
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module
    import pykwavers as module  # noqa: PLC0415

    return module


kw = load_pykwavers_extension()  # noqa: E402


INERTIAL_MI_THRESHOLD = float(os.environ.get("KWAVERS_CH29_INERTIAL_MI_THRESHOLD", "1.9"))
HISTOTRIPSY_SOURCE_PRESSURE_PA = float(os.environ.get("KWAVERS_CH29_HISTOTRIPSY_SOURCE_PRESSURE_PA", "28.0e6"))
BRAIN_HISTOTRIPSY_SOURCE_PRESSURE_PA = float(
    os.environ.get(
        "KWAVERS_CH29_BRAIN_HISTOTRIPSY_SOURCE_PRESSURE_PA",
        str(
            max(
                HISTOTRIPSY_SOURCE_PRESSURE_PA,
                2.0
                * INERTIAL_MI_THRESHOLD
                * np.sqrt(CANONICAL_BRAIN_SCENE.transducer.frequency_hz * 1.0e-6)
                * 1.0e6,
            )
        ),
    )
)


CASES = (
    {
        "name": "brain",
        "title": "Brain helmet",
        "ct": REPO_ROOT / "data" / "rire_patient_109" / "patient_109_ct.nii.gz",
        "seg": None,
        "grid": int(os.environ.get("KWAVERS_CH29_BRAIN_GRID", "48")),
        "elements": CANONICAL_BRAIN_SCENE.transducer.element_count,
        "freq": [220_000.0, CANONICAL_BRAIN_SCENE.transducer.frequency_hz],
        "offsets": [256, 384, 512, 640],
        "pressure": BRAIN_HISTOTRIPSY_SOURCE_PRESSURE_PA,
        "scene": CANONICAL_BRAIN_SCENE,
    },
    {
        "name": "kidney",
        "title": "Kidney histotripsy head",
        "ct": REPO_ROOT / "data" / "kits19_sample" / "case_00000.nii.gz",
        "seg": REPO_ROOT / "data" / "kits19_sample" / "segmentation_00000.nii.gz",
        "grid": int(os.environ.get("KWAVERS_CH29_ABDOMEN_GRID", "52")),
        "elements": 256,
        "freq": [250_000.0, 500_000.0, 750_000.0],
        "offsets": [32, 64, 96, 128],
        "pressure": HISTOTRIPSY_SOURCE_PRESSURE_PA,
    },
    {
        "name": "liver",
        "title": "Liver histotripsy head",
        "ct": REPO_ROOT / "data" / "lits17_sample" / "volume-0.nii",
        "seg": REPO_ROOT / "data" / "lits17_sample" / "segmentation-0.nii",
        "grid": int(os.environ.get("KWAVERS_CH29_ABDOMEN_GRID", "52")),
        "elements": 256,
        "freq": [250_000.0, 500_000.0, 750_000.0],
        "offsets": [32, 64, 96, 128],
        "pressure": HISTOTRIPSY_SOURCE_PRESSURE_PA,
    },
)
CASES_BY_NAME = {str(case["name"]): case for case in CASES}
NONLINEAR_CROP_EXTENT_CACHE: dict[str, list[float]] = {}

RECONSTRUCTION_CHANNELS = (
    ("active_lesion_reconstruction", "active Born inverse"),
    ("waveform_rtm_reconstruction", "linear acoustic RTM"),
    ("subharmonic_reconstruction", "subharmonic inverse"),
    ("harmonic_reconstruction", "harmonic inverse"),
    ("ultraharmonic_reconstruction", "ultraharmonic inverse"),
    ("fused_reconstruction", "fusion"),
)

RECONSTRUCTION_FIGURE_COLUMNS = (
    ("ct_hu", "gray", "CT + target + tx/rx"),
    ("exposure", "magma", "simulated exposure"),
    ("lesion_target", "magma", "lesion target"),
    ("active_lesion_reconstruction", "viridis", "active Born inverse"),
    ("waveform_rtm_reconstruction", "viridis", "linear acoustic RTM"),
    ("subharmonic_reconstruction", "viridis", "subharmonic inverse"),
    ("harmonic_reconstruction", "viridis", "harmonic inverse"),
    ("ultraharmonic_reconstruction", "viridis", "ultraharmonic inverse"),
    ("fused_reconstruction", "viridis", "fusion"),
)


def placement_arrays(result: dict[str, object]) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
    ct = np.asarray(result["placement_ct_hu"], dtype=float)
    spacing = tuple(float(v) for v in result["placement_spacing_m"])
    extent = image_extent_xy(ct, spacing)
    therapy_points = np.asarray(result["placement_therapy_points_m"], dtype=float)
    imaging_points = np.asarray(result["placement_imaging_points_m"], dtype=float)
    return ct, extent, therapy_points, imaging_points


def placement_axis_limits(result: dict[str, object]) -> tuple[tuple[float, float], tuple[float, float]]:
    _, extent, therapy_points, imaging_points = placement_arrays(result)
    therapy_x = therapy_points[:, 0]
    therapy_y = therapy_points[:, 1]
    imaging_x = imaging_points[:, 0] if imaging_points.size else np.array([], dtype=float)
    imaging_y = imaging_points[:, 1] if imaging_points.size else np.array([], dtype=float)
    return (
        axis_limits(extent[:2], therapy_x, imaging_x),
        axis_limits(extent[2:], therapy_y, imaging_y),
    )


def nonlinear_crop_extent(result: dict[str, object]) -> list[float]:
    if all(key in result for key in ("crop_bounds_index", "source_dimensions", "source_spacing_m")):
        bounds = np.asarray(result["crop_bounds_index"], dtype=float)
        dims = np.asarray(result["source_dimensions"], dtype=float)
        spacing = np.asarray(result["source_spacing_m"], dtype=float)
        return nonlinear_extent_from_metadata(bounds, dims, spacing)
    anatomy = str(result["anatomy"])
    if anatomy not in NONLINEAR_CROP_EXTENT_CACHE:
        NONLINEAR_CROP_EXTENT_CACHE[anatomy] = nonlinear_crop_extent_from_case(CASES_BY_NAME[anatomy])
    return NONLINEAR_CROP_EXTENT_CACHE[anatomy]


def nonlinear_extent_from_metadata(bounds: np.ndarray, dims: np.ndarray, spacing: np.ndarray) -> list[float]:
    center_x = 0.5 * (dims[0] - 1.0)
    center_y = 0.5 * (dims[1] - 1.0)
    return [
        (bounds[0] - center_x) * spacing[0],
        (bounds[1] - center_x) * spacing[0],
        (bounds[2] - center_y) * spacing[1],
        (bounds[3] - center_y) * spacing[1],
    ]


def nonlinear_crop_extent_from_case(case: dict[str, object]) -> list[float]:
    import nibabel as nib  # noqa: PLC0415

    image = nib.load(str(case["ct"]))
    ct = np.asarray(image.get_fdata(), dtype=float)
    spacing_mm = np.asarray(image.header.get_zooms()[:3], dtype=float)
    label = None
    if case["seg"] is not None:
        label = np.rint(np.asarray(nib.load(str(case["seg"])).get_fdata(), dtype=float)).astype(np.int16)
    anatomy = str(case["name"])
    if anatomy == "brain":
        body = ct > -300.0
        bounds = body_cube_bounds(body, spacing_mm)
    else:
        if label is None:
            raise ValueError(f"{anatomy} nonlinear crop requires segmentation")
        body = (ct > -450.0) | (label > 0)
        target = label == 2
        bounds = path_cube_bounds(body, target, spacing_mm)
    dims = np.asarray(ct.shape, dtype=float)
    spacing_m = spacing_mm * 1.0e-3
    return nonlinear_extent_from_metadata(np.asarray(bounds, dtype=float), dims, spacing_m)


def body_cube_bounds(body: np.ndarray, spacing_mm: np.ndarray) -> list[int]:
    bounds = mask_bounds(body)
    center = np.array(
        [
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        ],
        dtype=float,
    )
    spacing_m = spacing_mm * 1.0e-3
    radius_m = max(
        0.5 * (bounds[1] - bounds[0] + 1) * spacing_m[0],
        0.5 * (bounds[3] - bounds[2] + 1) * spacing_m[1],
        0.5 * (bounds[5] - bounds[4] + 1) * spacing_m[2],
    ) + 0.01
    return cube_from_center_radius(body.shape, center, radius_m, spacing_m)


def path_cube_bounds(body: np.ndarray, target: np.ndarray, spacing_mm: np.ndarray) -> list[int]:
    spacing_m = spacing_mm * 1.0e-3
    focus = centroid_float(target)
    skin = nearest_boundary(body, focus, spacing_m)
    center = 0.5 * (focus + skin)
    target_bounds = mask_bounds(target)
    target_radius_m = max_distance_to_bounds(center, target_bounds, spacing_m)
    skin_distance_m = physical_distance(focus, skin, spacing_m)
    radius_m = max(target_radius_m, 0.55 * skin_distance_m) + 0.025
    return cube_from_center_radius(body.shape, center, radius_m, spacing_m)


def mask_bounds(mask: np.ndarray) -> list[int]:
    coordinates = np.argwhere(np.asarray(mask, dtype=bool))
    if coordinates.size == 0:
        raise ValueError("mask support is empty")
    mins = coordinates.min(axis=0)
    maxs = coordinates.max(axis=0)
    return [int(mins[0]), int(maxs[0]), int(mins[1]), int(maxs[1]), int(mins[2]), int(maxs[2])]


def centroid_float(mask: np.ndarray) -> np.ndarray:
    coordinates = np.argwhere(np.asarray(mask, dtype=bool))
    if coordinates.size == 0:
        raise ValueError("mask support is empty")
    return coordinates.mean(axis=0)


def nearest_boundary(body: np.ndarray, focus: np.ndarray, spacing_m: np.ndarray) -> np.ndarray:
    active = np.asarray(body, dtype=bool)
    padded = np.pad(active, 1, mode="constant", constant_values=False)
    enclosed = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    boundary = active & ~enclosed
    coordinates = np.argwhere(boundary)
    if coordinates.size == 0:
        raise ValueError("body mask has no boundary")
    delta_m = (coordinates.astype(float) - focus[np.newaxis, :]) * spacing_m[np.newaxis, :]
    return coordinates[np.argmin(np.sum(delta_m * delta_m, axis=1))].astype(float)


def cube_from_center_radius(
    dims: tuple[int, int, int],
    center: np.ndarray,
    radius_m: float,
    spacing_m: np.ndarray,
) -> list[int]:
    radius_index = np.ceil(radius_m / spacing_m).astype(int)
    rounded = np.rint(center).astype(int)
    lower = np.maximum(rounded - radius_index, 0)
    upper = np.minimum(rounded + radius_index, np.asarray(dims, dtype=int) - 1)
    return [int(lower[0]), int(upper[0]), int(lower[1]), int(upper[1]), int(lower[2]), int(upper[2])]


def max_distance_to_bounds(center: np.ndarray, bounds: list[int], spacing_m: np.ndarray) -> float:
    max_distance = 0.0
    for x in (bounds[0], bounds[1]):
        for y in (bounds[2], bounds[3]):
            for z in (bounds[4], bounds[5]):
                max_distance = max(max_distance, physical_distance(center, np.array([x, y, z], dtype=float), spacing_m))
    return max_distance


def physical_distance(a: np.ndarray, b: np.ndarray, spacing_m: np.ndarray) -> float:
    return float(np.linalg.norm((a - b) * spacing_m))


def plot_placement_ct(
    ax: plt.Axes,
    result: dict[str, object],
    *,
    show_legend: bool,
    target_color: str = "yellow",
    show_beams: bool = False,
) -> AxesImage:
    ct, extent, therapy_points, imaging_points = placement_arrays(result)
    therapy_x = therapy_points[:, 0]
    therapy_y = therapy_points[:, 1]
    imaging_x = imaging_points[:, 0] if imaging_points.size else np.array([], dtype=float)
    imaging_y = imaging_points[:, 1] if imaging_points.size else np.array([], dtype=float)

    im = ax.imshow(ct.T, cmap="gray", origin="lower", extent=extent, vmin=-200, vmax=300)
    contour_mask(ax, np.asarray(result["placement_target_mask"], dtype=bool), extent, target_color, 1.1)
    contour_mask(ax, np.asarray(result["placement_body_mask"], dtype=bool), extent, "cyan", 0.8)
    focus = np.asarray(result["placement_focus_m"], dtype=float)
    if show_beams:
        plot_beam_paths_2d(ax, therapy_points, focus)
    ax.scatter(therapy_x, therapy_y, s=2.0, c="#e74c3c", alpha=0.50, label="therapy tx/rx")
    if imaging_x.size > 0:
        ax.scatter(imaging_x, imaging_y, s=6.0, c="#2e86de", alpha=0.80, label="central imaging rx")
    skin = result["placement_skin_contact_m"]
    ax.scatter([focus[0]], [focus[1]], marker="x", s=45, c="white", linewidths=1.6)
    ax.scatter([skin[0]], [skin[1]], marker="o", s=24, c="lime", edgecolors="black", linewidths=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(*axis_limits(extent[:2], therapy_x, imaging_x))
    ax.set_ylim(*axis_limits(extent[2:], therapy_y, imaging_y))
    if show_legend:
        ax.legend(loc="lower right", fontsize=7, frameon=True)
    return im


def plot_beam_paths_2d(ax: plt.Axes, therapy_points: np.ndarray, focus: np.ndarray) -> None:
    if therapy_points.size == 0:
        return
    max_paths = int(os.environ.get("KWAVERS_CH29_FIG05_BEAM_PATHS", "64"))
    count = min(max_paths, therapy_points.shape[0])
    if count == 0:
        return
    indices = np.linspace(0, therapy_points.shape[0] - 1, count, dtype=int)
    sampled = therapy_points[indices]
    for point in sampled:
        ax.plot(
            [point[0], focus[0]],
            [point[1], focus[1]],
            color="#ffd166",
            linewidth=0.35,
            alpha=0.24,
            solid_capstyle="round",
)


def case_scene(case: dict[str, object]) -> BrainSceneDefinition | None:
    scene = case.get("scene")
    if scene is None:
        return None
    if not isinstance(scene, BrainSceneDefinition):
        raise TypeError("case scene must be a BrainSceneDefinition")
    return scene


def scene_target_kwargs(scene: BrainSceneDefinition | None) -> dict[str, object]:
    return {} if scene is None else {"target_fraction_xyz": scene.target.fraction_xyz}


def run_case(case: dict[str, object]) -> dict[str, object]:
    if not Path(case["ct"]).exists():
        raise FileNotFoundError(case["ct"])
    seg = case["seg"]
    if seg is not None and not Path(seg).exists():
        raise FileNotFoundError(seg)
    scene = case_scene(case)
    scene_kwargs = scene_target_kwargs(scene)
    return kw.run_theranostic_inverse_from_ritk(
        str(case["ct"]),
        None if seg is None else str(seg),
        anatomy=str(case["name"]),
        grid_size=int(case["grid"]),
        element_count=int(case["elements"]),
        iterations=int(os.environ.get("KWAVERS_CH29_ITERATIONS", "10")),
        frequencies_hz=list(case["freq"]),
        receiver_offsets=list(case["offsets"]),
        source_pressure_pa=float(case["pressure"]),
        noise_fraction=float(os.environ.get("KWAVERS_CH29_NOISE_FRACTION", "0.012")),
        inverse_encoding_rows_per_code=int(os.environ.get("KWAVERS_CH29_INVERSE_ENCODING_ROWS_PER_CODE", "2")),
        **scene_kwargs,
    )


def run_controlled_linear_case(case: dict[str, object]) -> dict[str, object]:
    if not Path(case["ct"]).exists():
        raise FileNotFoundError(case["ct"])
    seg = case["seg"]
    if seg is not None and not Path(seg).exists():
        raise FileNotFoundError(seg)
    scene = case_scene(case)
    scene_kwargs = scene_target_kwargs(scene)
    return kw.run_theranostic_inverse_from_ritk(
        str(case["ct"]),
        None if seg is None else str(seg),
        anatomy=str(case["name"]),
        grid_size=nonlinear_grid_size(case),
        element_count=nonlinear_element_count(case),
        iterations=int(
            os.environ.get(
                "KWAVERS_CH29_CONTROLLED_LINEAR_ITERATIONS",
                os.environ.get("KWAVERS_CH29_ITERATIONS", "10"),
            )
        ),
        frequencies_hz=[nonlinear_frequency_hz(case)],
        receiver_offsets=list(case["offsets"]),
        source_pressure_pa=nonlinear_source_pressure_pa(case),
        noise_fraction=float(os.environ.get("KWAVERS_CH29_NOISE_FRACTION", "0.012")),
        inverse_encoding_rows_per_code=int(os.environ.get("KWAVERS_CH29_INVERSE_ENCODING_ROWS_PER_CODE", "2")),
        **scene_kwargs,
    )


def run_nonlinear_case(case: dict[str, object]) -> dict[str, object]:
    if not Path(case["ct"]).exists():
        raise FileNotFoundError(case["ct"])
    seg = case["seg"]
    if seg is not None and not Path(seg).exists():
        raise FileNotFoundError(seg)
    grid = nonlinear_grid_size(case)
    iterations = int(os.environ.get("KWAVERS_CH29_NONLINEAR_ITERATIONS", "3"))
    scene = case_scene(case)
    scene_kwargs = scene_target_kwargs(scene)
    return kw.run_theranostic_nonlinear_3d_from_ritk(
        str(case["ct"]),
        None if seg is None else str(seg),
        anatomy=str(case["name"]),
        grid_size=grid,
        element_count=nonlinear_element_count(case),
        receiver_count=nonlinear_receiver_count(case),
        source_encoding_count=int(os.environ.get("KWAVERS_CH29_NONLINEAR_ENCODINGS", "3")),
        checkpoint_interval_steps=int(os.environ.get("KWAVERS_CH29_CHECKPOINT_INTERVAL", "128")),
        iterations=iterations,
        frequency_hz=nonlinear_frequency_hz(case),
        source_pressure_pa=nonlinear_source_pressure_pa(case),
        cycles=float(os.environ.get("KWAVERS_CH29_NONLINEAR_CYCLES", "3.0")),
        lesion_delta_c_m_s=float(os.environ.get("KWAVERS_CH29_NONLINEAR_DELTA_C", "-35.0")),
        lesion_delta_beta=float(os.environ.get("KWAVERS_CH29_NONLINEAR_DELTA_BETA", "0.85")),
        sound_speed_regularization=float(os.environ.get("KWAVERS_CH29_NONLINEAR_C_REG", "0.002")),
        nonlinearity_regularization=float(os.environ.get("KWAVERS_CH29_NONLINEAR_BETA_REG", "0.001")),
        gradient_smoothing_steps=int(os.environ.get("KWAVERS_CH29_NONLINEAR_SMOOTHING", "2")),
        bubble_time_steps_per_period=int(os.environ.get("KWAVERS_CH29_RP_STEPS_PER_PERIOD", "64")),
        inertial_mi_threshold=INERTIAL_MI_THRESHOLD,
        cavitation_iterations=int(os.environ.get("KWAVERS_CH29_CAVITATION_ITERATIONS", "24")),
        **scene_kwargs,
    )


def nonlinear_frequency_hz(case: dict[str, object]) -> float:
    return float(os.environ.get("KWAVERS_CH29_NONLINEAR_FREQUENCY_HZ", str(case["freq"][-1])))


def nonlinear_source_pressure_pa(case: dict[str, object]) -> float:
    return float(os.environ.get("KWAVERS_CH29_NONLINEAR_SOURCE_PRESSURE_PA", str(case["pressure"])))


def nonlinear_grid_size(case: dict[str, object]) -> int:
    case_grid_key = f"KWAVERS_CH29_{str(case['name']).upper()}_NONLINEAR_GRID"
    return int(
        os.environ.get(
            case_grid_key,
            os.environ.get("KWAVERS_CH29_NONLINEAR_GRID", str(case["grid"])),
        )
    )


def nonlinear_element_count(case: dict[str, object]) -> int:
    case_key = f"KWAVERS_CH29_{str(case['name']).upper()}_NONLINEAR_ELEMENTS"
    return int(
        os.environ.get(
            case_key,
            os.environ.get("KWAVERS_CH29_NONLINEAR_ELEMENTS", str(case["elements"])),
        )
    )


def nonlinear_receiver_count(case: dict[str, object]) -> int:
    case_key = f"KWAVERS_CH29_{str(case['name']).upper()}_NONLINEAR_RECEIVERS"
    element_count = nonlinear_element_count(case)
    return int(
        os.environ.get(
            case_key,
            os.environ.get("KWAVERS_CH29_NONLINEAR_RECEIVERS", str(element_count)),
        )
    )


def render_layouts(results: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(15, 4.8), constrained_layout=True)
    for ax, result in zip(axes, results):
        plot_placement_ct(ax, result, show_legend=True)
        ax.set_title(
            f"{result['anatomy']}: {short_device_name(result)}\n"
            f"full CT slice, {result['element_count']} elements, {placement_label(result)}"
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    path = OUT_DIR / "fig01_device_placement_on_ct.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def render_reconstructions(results: list[dict[str, object]]) -> Path:
    columns = RECONSTRUCTION_FIGURE_COLUMNS
    fig, axes = plt.subplots(len(results), len(columns), figsize=(23.5, 8.4), constrained_layout=True)
    for row, result in enumerate(results):
        for col, (key, cmap, title) in enumerate(columns):
            ax = axes[row, col]
            if key == "ct_hu":
                im = plot_placement_ct(ax, result, show_legend=False, target_color="white")
                ax.set_xlabel(short_device_name(result), fontsize=7)
            else:
                image = np.asarray(result[key], dtype=float)
                im = ax.imshow(image.T, cmap=cmap, origin="lower", vmin=0.0, vmax=max(float(np.max(image)), 1.0e-12))
                ax.contour(np.asarray(result["target_mask"], dtype=bool).T, levels=[0.5], colors="white", linewidths=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{result['anatomy']} {title}" if col == 0 else title, fontsize=9)
            if col == len(columns) - 1:
                metrics = result["metrics"]["fusion"]
                ax.set_xlabel(f"Dice={metrics['dice_equal_area']:.2f}, CNR={metrics['cnr']:.2f}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    path = OUT_DIR / "fig02_exposure_and_reconstruction.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def render_nonlinear_3d(results: list[dict[str, object]], placement_results: list[dict[str, object]]) -> Path:
    columns = (
        ("ct_hu", "gray", "CT + target + tx/rx + beams"),
        ("planned_exposure", "magma", "planned exposure"),
        ("westervelt_peak_pressure_pa", "magma", "Westervelt FDTD peak"),
        ("target_mask", "magma", "planned lesion target"),
        ("multiparameter_fwi_score", "viridis", "source-encoded Westervelt FWI"),
        ("reconstructed_delta_beta", "viridis", "nonlinear beta inverse"),
        ("cavitation_source_density", "viridis", "RP cavitation source"),
        ("reconstructed_cavitation_density", "viridis", "passive cavitation inverse"),
        ("nonlinear_fusion_score", "viridis", "fusion"),
    )
    if len(results) != len(placement_results):
        raise ValueError("nonlinear and placement result counts differ")
    fig, axes = plt.subplots(len(results), len(columns), figsize=(28.0, 9.2), constrained_layout=True)
    for row, (result, placement) in enumerate(zip(results, placement_results)):
        if str(result["anatomy"]) != str(placement["anatomy"]):
            raise ValueError(f"case order mismatch: {result['anatomy']} != {placement['anatomy']}")
        z = target_slice_index(np.asarray(result["target_mask"], dtype=bool))
        nonlinear_extent = nonlinear_crop_extent(result)
        _, placement_extent, _, _ = placement_arrays(placement)
        xlim, ylim = placement_axis_limits(placement)
        slab = target_slab_bounds(np.asarray(result["target_mask"], dtype=bool), z)
        target = np.max(np.asarray(result["target_mask"], dtype=bool)[:, :, slab[0] : slab[1]], axis=2)
        expected_lesion = np.asarray(placement["lesion_target"], dtype=float)
        expected_lesion_mask = lesion_mask(expected_lesion)
        placement_target_outline = np.asarray(placement["target_mask"], dtype=bool)
        for col, (key, cmap, title) in enumerate(columns):
            ax = axes[row, col]
            if key == "ct_hu":
                im = plot_placement_ct(ax, placement, show_legend=False, target_color="white", show_beams=True)
            elif key == "planned_exposure":
                image = np.asarray(placement["exposure"], dtype=float)
                im = ax.imshow(
                    image.T,
                    cmap=cmap,
                    origin="lower",
                    extent=placement_extent,
                    vmin=0.0,
                    vmax=max(float(np.max(image)), 1.0e-12),
                )
                contour_mask(ax, resize_mask(placement_target_outline, image.shape), placement_extent, "white", 0.8)
                contour_mask(ax, expected_lesion_mask, placement_extent, "yellow", 0.9)
            elif key == "target_mask":
                image = expected_lesion
                im = ax.imshow(image.T, cmap=cmap, origin="lower", extent=placement_extent, vmin=0.0, vmax=1.0)
                contour_mask(ax, resize_mask(placement_target_outline, image.shape), placement_extent, "white", 0.8)
                contour_mask(ax, expected_lesion_mask, placement_extent, "yellow", 1.1)
            elif key in {"multiparameter_fwi_score", "nonlinear_fusion_score"}:
                image = slab_projection(np.asarray(result[key], dtype=float), slab, mode="max")
                im = ax.imshow(image.T, cmap=cmap, origin="lower", extent=nonlinear_extent, vmin=0.0, vmax=1.0)
            elif key == "reconstructed_delta_beta":
                image = slab_projection(np.maximum(np.asarray(result[key], dtype=float), 0.0), slab, mode="max")
                im = ax.imshow(
                    image.T,
                    cmap=cmap,
                    origin="lower",
                    extent=nonlinear_extent,
                    vmin=0.0,
                    vmax=max(float(np.max(image)), 1.0e-12),
                )
            else:
                volume = np.asarray(result[key], dtype=float)
                image = slab_projection(volume, slab, mode="max")
                im = ax.imshow(image.T, cmap=cmap, origin="lower", extent=nonlinear_extent)
            if key not in {"ct_hu", "planned_exposure"}:
                contour_mask(ax, target, nonlinear_extent, "white", 0.8)
            if key not in {"ct_hu", "target_mask", "planned_exposure"}:
                contour_mask(ax, expected_lesion_mask, placement_extent, "yellow", 0.9)
            if key != "ct_hu":
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{result['anatomy']} {title}" if col == 0 else title, fontsize=8.5)
            if col == len(columns) - 1:
                fwi = result["metrics"]["fwi"]
                cav = result["metrics"]["rayleigh_plesset_cavitation"]
                fusion = result["metrics"]["fusion"]
                ax.set_xlabel(
                    f"FWI={fwi['dice_equal_area']:.2f}; RP={cav['dice_equal_area']:.2f}; "
                    f"fusion={fusion['dice_equal_area']:.2f}"
                )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    path = OUT_DIR / "fig05_nonlinear_3d_westervelt_rp_cavitation.png"
    save_figure(fig, path, dpi=int(os.environ.get("KWAVERS_CH29_FIG05_DPI", "360")))
    plt.close(fig)
    return path


def render_controlled_linear_nonlinear(
    linear_results: list[dict[str, object]],
    nonlinear_results: list[dict[str, object]],
) -> tuple[Path, Path, Path, list[dict[str, object]]]:
    comparisons = build_controlled_comparison(linear_results, nonlinear_results)
    fields = write_controlled_comparison_fields(comparisons, OUT_DIR)
    figure = render_controlled_comparison(comparisons, OUT_DIR, save_figure)
    metrics = write_controlled_comparison_metrics(comparisons, OUT_DIR, figure, fields)
    return figure, metrics, fields, comparisons


def render_brain_helmet_3d(placement: dict[str, object]) -> Path:
    head = np.asarray(placement["head_surface_points_m"], dtype=float)
    skull = np.asarray(placement["skull_surface_points_m"], dtype=float)
    elements = np.asarray(placement["therapy_elements_m"], dtype=float)
    starts = np.asarray(placement["beam_start_points_m"], dtype=float)
    ends = np.asarray(placement["beam_end_points_m"], dtype=float)
    intersections = np.asarray(placement["skull_intersections_m"], dtype=float)
    focus = np.asarray(placement["focus_m"], dtype=float)

    fig = plt.figure(figsize=(13.5, 6.2), constrained_layout=True)
    ax_a = fig.add_subplot(1, 2, 1, projection="3d")
    ax_b = fig.add_subplot(1, 2, 2, projection="3d")
    for ax in (ax_a, ax_b):
        ax.scatter(head[:, 0], head[:, 1], head[:, 2], s=0.9, c="#b8c0cc", alpha=0.16, depthshade=False)
        ax.scatter(skull[:, 0], skull[:, 1], skull[:, 2], s=1.5, c="#f2d7a0", alpha=0.38, depthshade=False)
        ax.scatter(elements[:, 0], elements[:, 1], elements[:, 2], s=5.0, c="#d94f45", alpha=0.62, depthshade=False)
        if intersections.size:
            ax.scatter(
                intersections[:, 0],
                intersections[:, 1],
                intersections[:, 2],
                s=16,
                c="#ffff33",
                edgecolors="black",
                linewidths=0.25,
                depthshade=False,
            )
        ax.scatter([focus[0]], [focus[1]], [focus[2]], marker="x", s=80, c="white", linewidths=2.0)
        for start, end in zip(starts, ends):
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                c="#ff8c00",
                lw=0.55,
                alpha=0.42,
            )
        set_equal_3d_limits(ax, [head, skull, elements])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
    ax_a.view_init(elev=18, azim=-68)
    ax_a.set_title(
        "1024-element helmet around CT head\n"
        "red: tx/rx elements, yellow: CT skull entry points",
        fontsize=10,
    )
    ax_b.view_init(elev=4, azim=-8)
    ax_b.set_title(
        "calvarial beam paths into brain target\n"
        f"skull-intersection fraction={float(placement['intersection_fraction']):.2f}",
        fontsize=10,
    )
    path = OUT_DIR / "fig03_brain_helmet_3d_placement.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def render_dynamic_range_diagnostics(results: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(
        len(results),
        len(RECONSTRUCTION_CHANNELS),
        figsize=(15.0, 8.2),
        constrained_layout=True,
    )
    last_image = None
    for row, result in enumerate(results):
        diagnostics = reconstruction_diagnostics(result)
        target = np.asarray(result["target_mask"], dtype=bool)
        for col, (key, title) in enumerate(RECONSTRUCTION_CHANNELS):
            ax = axes[row, col]
            image = np.asarray(result[key], dtype=float)
            db_image = normalized_db(image)
            last_image = ax.imshow(db_image.T, cmap="magma", origin="lower", vmin=-40.0, vmax=0.0)
            ax.contour(target.T, levels=[0.5], colors="white", linewidths=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            channel = diagnostics[key]
            ax.set_title(f"{result['anatomy']} {title}" if col == 0 else title, fontsize=8.5)
            ax.set_xlabel(
                f"outside peak {channel['outside_peak_db']:.1f} dB\n"
                f"outside energy {100.0 * channel['outside_energy_fraction']:.1f}%"
            )
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01, label="relative amplitude [dB]")
    path = OUT_DIR / "fig04_reconstruction_dynamic_range_diagnostics.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def image_extent(image: np.ndarray, spacing_m: float) -> list[float]:
    nx, ny = image.shape
    return [
        -0.5 * (nx - 1) * spacing_m,
        0.5 * (nx - 1) * spacing_m,
        -0.5 * (ny - 1) * spacing_m,
        0.5 * (ny - 1) * spacing_m,
    ]


def save_figure(fig: plt.Figure, path: Path, *, dpi: int | None = None) -> None:
    fig.savefig(path, dpi=dpi or int(os.environ.get("KWAVERS_CH29_FIGURE_DPI", "260")))
    try:
        fig.savefig(path.with_suffix(".pdf"))
    except PermissionError:
        pass


def image_extent_xy(image: np.ndarray, spacing_m: tuple[float, float]) -> list[float]:
    nx, ny = image.shape
    return [
        -0.5 * (nx - 1) * spacing_m[0],
        0.5 * (nx - 1) * spacing_m[0],
        -0.5 * (ny - 1) * spacing_m[1],
        0.5 * (ny - 1) * spacing_m[1],
    ]


def target_slice_index(mask: np.ndarray) -> int:
    counts = np.sum(mask, axis=(0, 1))
    return int(np.argmax(counts))


def target_slab_bounds(mask: np.ndarray, z_index: int) -> tuple[int, int]:
    half_width = max(1, min(3, mask.shape[2] // 8))
    z0 = max(0, z_index - half_width)
    z1 = min(mask.shape[2], z_index + half_width + 1)
    return z0, z1


def slab_projection(volume: np.ndarray, slab: tuple[int, int], *, mode: str) -> np.ndarray:
    data = np.asarray(volume[:, :, slab[0] : slab[1]], dtype=float)
    if mode == "mean":
        return np.mean(data, axis=2)
    return np.max(data, axis=2)


def lesion_mask(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    return np.isfinite(values) & (values > 0.0)


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.shape == shape:
        return values
    x_idx = np.rint(np.linspace(0, values.shape[0] - 1, shape[0])).astype(int)
    y_idx = np.rint(np.linspace(0, values.shape[1] - 1, shape[1])).astype(int)
    return values[np.ix_(x_idx, y_idx)]


def plot_projected_3d_points(ax: plt.Axes, result: dict[str, object], z_index: int) -> None:
    spacing = float(result["spacing_m"])
    z_m = (z_index - 0.5 * (np.asarray(result["ct_hu"]).shape[2] - 1)) * spacing
    for key, color, size, alpha in (
        ("therapy_points_m", "#e74c3c", 4.0, 0.45),
        ("receiver_points_m", "#2e86de", 7.0, 0.75),
    ):
        points = np.asarray(result[key], dtype=float)
        if points.size == 0:
            continue
        near = np.abs(points[:, 2] - z_m) <= 2.5 * spacing
        if np.any(near):
            ax.scatter(points[near, 0], points[near, 1], s=size, c=color, alpha=alpha)


def short_device_name(result: dict[str, object]) -> str:
    name = str(result["device_model"])
    if "helmet" in name:
        return "INSIGHTEC-like helmet"
    return "HistoSonics-like skin arc"


def placement_label(result: dict[str, object]) -> str:
    metrics = result["placement_metrics"]
    gap_m = result.get(
        "placement_context_skin_gap_m",
        metrics["skin_contact_to_nearest_aperture_m"],
    )
    gap_mm = 1.0e3 * float(gap_m)
    clearance_mm = 1.0e3 * float(metrics["min_body_clearance_m"])
    if str(result["anatomy"]) == "brain":
        return f"helmet clearance {clearance_mm:.1f} mm"
    return f"skin gap {gap_mm:.1f} mm"


def axis_limits(
    image_limits: list[float] | tuple[float, float],
    therapy_values: np.ndarray,
    imaging_values: np.ndarray,
) -> tuple[float, float]:
    values = [float(image_limits[0]), float(image_limits[1])]
    if therapy_values.size > 0:
        values.extend([float(np.min(therapy_values)), float(np.max(therapy_values))])
    if imaging_values.size > 0:
        values.extend([float(np.min(imaging_values)), float(np.max(imaging_values))])
    low = min(values)
    high = max(values)
    margin = max(0.04 * (high - low), 5.0e-3)
    return low - margin, high + margin


def contour_mask(ax: plt.Axes, mask: np.ndarray, extent: list[float], color: str, width: float) -> None:
    x = np.linspace(extent[0], extent[1], mask.shape[0])
    y = np.linspace(extent[2], extent[3], mask.shape[1])
    ax.contour(x, y, mask.T.astype(float), levels=[0.5], colors=color, linewidths=width)


def set_equal_3d_limits(ax: plt.Axes, clouds: list[np.ndarray]) -> None:
    stacked = np.vstack([cloud for cloud in clouds if cloud.size])
    mins = np.min(stacked, axis=0)
    maxs = np.max(stacked, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.52 * float(np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def normalized_db(image: np.ndarray, floor_db: float = -40.0) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    peak = float(np.max(np.abs(values)))
    if peak <= 0.0 or not np.isfinite(peak):
        return np.full(values.shape, floor_db, dtype=float)
    ratio = np.maximum(np.abs(values) / peak, 10.0 ** (floor_db / 20.0))
    return np.maximum(20.0 * np.log10(ratio), floor_db)


def reconstruction_diagnostics(result: dict[str, object]) -> dict[str, dict[str, float]]:
    target = np.asarray(result["target_mask"], dtype=bool)
    body = np.asarray(result.get("body_mask", np.ones_like(target)), dtype=bool)
    outside = body & ~target
    diagnostics: dict[str, dict[str, float]] = {}
    for key, _ in RECONSTRUCTION_CHANNELS:
        image = np.asarray(result[key], dtype=float)
        peak = float(np.max(np.abs(image)))
        outside_peak = float(np.max(np.abs(image[outside]))) if np.any(outside) else 0.0
        total_energy = float(np.sum(image[body] ** 2)) if np.any(body) else 0.0
        outside_energy = float(np.sum(image[outside] ** 2)) if np.any(outside) else 0.0
        outside_peak_ratio = outside_peak / peak if peak > 0.0 else 0.0
        outside_energy_fraction = outside_energy / total_energy if total_energy > 0.0 else 0.0
        diagnostics[key] = {
            "peak": peak,
            "outside_peak_ratio": outside_peak_ratio,
            "outside_peak_db": ratio_to_db(outside_peak_ratio),
            "outside_energy_fraction": outside_energy_fraction,
        }
    return diagnostics


def ratio_to_db(ratio: float, floor_db: float = -120.0) -> float:
    if ratio <= 0.0 or not np.isfinite(ratio):
        return floor_db
    return max(20.0 * float(np.log10(ratio)), floor_db)


def write_metrics(
    results: list[dict[str, object]],
    nonlinear_results: list[dict[str, object]],
    figures: list[Path],
    brain_helmet_3d: dict[str, object],
    controlled_comparisons: list[dict[str, object]] | None = None,
    controlled_figure: Path | None = None,
    controlled_fields: Path | None = None,
) -> Path:
    payload = {
        "chapter": 29,
        "analysis": "same-device ultrasound treatment plus finite-frequency inverse and source-encoded linear RTM monitoring",
        "simulation_type": "RITK-loaded CT/NIfTI, kwavers PyO3 theranostic inverse",
        "brain_scene": CANONICAL_BRAIN_SCENE.to_manifest(),
        "figures": [str(path) for path in figures],
        "brain_helmet_3d": {
            "geometry_model": brain_helmet_3d["geometry_model"],
            "element_count": int(brain_helmet_3d["element_count"]),
            "helmet_radius_m": float(brain_helmet_3d["helmet_radius_m"]),
            "beam_probe_count": int(np.asarray(brain_helmet_3d["beam_start_points_m"]).shape[0]),
            "skull_intersection_count": int(np.asarray(brain_helmet_3d["skull_intersections_m"]).shape[0]),
            "skull_intersection_fraction": float(brain_helmet_3d["intersection_fraction"]),
            "skull_hu_threshold": float(brain_helmet_3d["skull_hu_threshold"]),
            "target_fraction_xyz": [float(v) for v in brain_helmet_3d.get("target_fraction_xyz", ())],
            "scene_radius_m": float(brain_helmet_3d.get("scene_radius_m", CANONICAL_BRAIN_SCENE.transducer.radius_m)),
        },
        "cases": [
            {
                "anatomy": result["anatomy"],
                "device_model": result["device_model"],
                "geometry_model": result["geometry_model"],
                "placement_context_model": result["placement_context_model"],
                "operator_model": result["operator_model"],
                "operator_backend": result["operator_backend"],
                "operator_storage_values": int(result["operator_storage_values"]),
                "dense_operator_values": int(result["dense_operator_values"]),
                "inverse_model_family": result["inverse_model_family"],
                "is_full_wave_inversion": bool(result["is_full_wave_inversion"]),
                "uses_nonlinear_wave_propagation": bool(result["uses_nonlinear_wave_propagation"]),
                "waveform_model": result["waveform_model"],
                "waveform_misfit": result["waveform_misfit"],
                "waveform_misfit_scale": float(result["waveform_misfit_scale"]),
                "waveform_objective": float(result["waveform_objective"]),
                "waveform_receiver_count": int(result["waveform_receiver_count"]),
                "waveform_time_steps": int(result["waveform_time_steps"]),
                "waveform_dt_s": float(result["waveform_dt_s"]),
                "waveform_residual_energy": float(result["waveform_residual_energy"]),
                "waveform_observed_energy": float(result["waveform_observed_energy"]),
                "element_count": int(result["element_count"]),
                "source_pressure_pa": float(result["source_pressure_pa"]),
                "target_fraction_xyz": [float(v) for v in result.get("target_fraction_xyz", ())],
                "measurements": int(result["measurements"]),
                "encoded_measurements": int(result["encoded_measurements"]),
                "unencoded_measurements": int(result["unencoded_measurements"]),
                "inverse_encoding_rows_per_code": int(result["inverse_encoding_rows_per_code"]),
                "active_voxels": int(result["active_voxels"]),
                "spacing_m": float(result["spacing_m"]),
                "placement_slice_index": int(result["placement_slice_index"]),
                "placement_spacing_m": [float(v) for v in result["placement_spacing_m"]],
                "placement_context_skin_gap_m": float(result["placement_context_skin_gap_m"]),
                "placement_context_surface_points": int(result["placement_context_surface_points"]),
                "placement_metrics": result["placement_metrics"],
                "metrics": result["metrics"],
                "reconstruction_diagnostics": reconstruction_diagnostics(result),
            }
            for result in results
        ],
        "nonlinear_3d_cases": [
            {
                "anatomy": result["anatomy"],
                "model_family": result["model_family"],
                "propagator_model": result["propagator_model"],
                "cavitation_inverse_model": result["cavitation_inverse_model"],
                "aperture_model": result["aperture_model"],
                "is_full_wave_inversion": bool(result["is_full_wave_inversion"]),
                "uses_nonlinear_wave_propagation": bool(result["uses_nonlinear_wave_propagation"]),
                "uses_rayleigh_plesset": bool(result["uses_rayleigh_plesset"]),
                "grid_size": int(result["grid_size"]),
                "time_steps": int(result["time_steps"]),
                "dt_s": float(result["dt_s"]),
                "active_voxels": int(result["active_voxels"]),
                "source_dimensions": [int(v) for v in result["source_dimensions"]],
                "source_spacing_m": [float(v) for v in result["source_spacing_m"]],
                "crop_bounds_index": [int(v) for v in result["crop_bounds_index"]],
                "element_count": int(result["element_count"]),
                "receiver_count": int(result["receiver_count"]),
                "source_encoding_count": int(result["source_encoding_count"]),
                "checkpoint_interval_steps": int(result["checkpoint_interval_steps"]),
                "frequency_hz": float(result["frequency_hz"]),
                "source_pressure_pa": float(result["source_pressure_pa"]),
                "inertial_mi_threshold": float(result.get("inertial_mi_threshold", INERTIAL_MI_THRESHOLD)),
                "bubble_time_steps_per_period": int(
                    result.get(
                        "bubble_time_steps_per_period",
                        os.environ.get("KWAVERS_CH29_RP_STEPS_PER_PERIOD", "64"),
                    )
                ),
                "target_fraction_xyz": [float(v) for v in result.get("target_fraction_xyz", ())],
                "lesion_delta_c_m_s": float(result["lesion_delta_c_m_s"]),
                "lesion_delta_beta": float(result["lesion_delta_beta"]),
                "sound_speed_regularization": float(result["sound_speed_regularization"]),
                "nonlinearity_regularization": float(result["nonlinearity_regularization"]),
                "gradient_smoothing_steps": int(result["gradient_smoothing_steps"]),
                "fwi_objective_history": [float(v) for v in np.asarray(result["fwi_objective_history"], dtype=float)],
                "cavitation_objective_history": [
                    float(v) for v in np.asarray(result["cavitation_objective_history"], dtype=float)
                ],
                "metrics": result["metrics"],
            }
            for result in nonlinear_results
        ],
        "controlled_linear_nonlinear_comparison": None
        if controlled_comparisons is None
        else controlled_comparison_payload(controlled_comparisons, controlled_figure, controlled_fields),
    }
    path = OUT_DIR / "metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_case(case) for case in CASES]
    controlled_linear_results = [run_controlled_linear_case(case) for case in CASES]
    nonlinear_results = [run_nonlinear_case(case) for case in CASES]
    helmet_kwargs = CANONICAL_BRAIN_SCENE.helmet_pykwavers_kwargs()
    brain_helmet_3d = kw.plan_brain_helmet_placement_from_ritk_ct(
        str(CASES[0]["ct"]),
        surface_stride=int(os.environ.get("KWAVERS_CH29_3D_SURFACE_STRIDE", "7")),
        **helmet_kwargs,
    )
    figures = [
        render_layouts(results),
        render_reconstructions(results),
        render_brain_helmet_3d(brain_helmet_3d),
        render_dynamic_range_diagnostics(results),
        render_nonlinear_3d(nonlinear_results, results),
    ]
    comparison_figure, _, comparison_fields, comparisons = render_controlled_linear_nonlinear(
        controlled_linear_results,
        nonlinear_results,
    )
    figures.append(comparison_figure)
    metrics = write_metrics(
        results,
        nonlinear_results,
        figures,
        brain_helmet_3d,
        comparisons,
        comparison_figure,
        comparison_fields,
    )
    return {"figures": [str(path) for path in figures], "metrics": str(metrics)}


def run_fig05_only() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    placement_results: list[dict[str, object]] = []
    for case in CASES:
        print(f"fig05 linear placement {case['name']} start", flush=True)
        placement_results.append(run_case(case))
        print(f"fig05 linear placement {case['name']} done", flush=True)

    nonlinear_results: list[dict[str, object]] = []
    for case in CASES:
        print(f"fig05 nonlinear {case['name']} start", flush=True)
        nonlinear_results.append(run_nonlinear_case(case))
        print(f"fig05 nonlinear {case['name']} done", flush=True)

    controlled_linear_results: list[dict[str, object]] = []
    for case in CASES:
        print(f"fig05 controlled linear {case['name']} start", flush=True)
        controlled_linear_results.append(run_controlled_linear_case(case))
        print(f"fig05 controlled linear {case['name']} done", flush=True)

    figure = render_nonlinear_3d(nonlinear_results, placement_results)
    comparison_figure, comparison_metrics, _, _ = render_controlled_linear_nonlinear(
        controlled_linear_results,
        nonlinear_results,
    )
    print(f"fig05 rendered {figure}", flush=True)
    print(f"controlled comparison rendered {comparison_figure}", flush=True)
    return {"figures": [str(figure), str(comparison_figure)], "metrics": str(comparison_metrics)}


def run_comparison_only() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    controlled_linear_results: list[dict[str, object]] = []
    nonlinear_results: list[dict[str, object]] = []
    for case in CASES:
        print(f"comparison controlled linear {case['name']} start", flush=True)
        controlled_linear_results.append(run_controlled_linear_case(case))
        print(f"comparison controlled linear {case['name']} done", flush=True)
    for case in CASES:
        print(f"comparison nonlinear {case['name']} start", flush=True)
        nonlinear_results.append(run_nonlinear_case(case))
        print(f"comparison nonlinear {case['name']} done", flush=True)
    figure, metrics, fields, _ = render_controlled_linear_nonlinear(controlled_linear_results, nonlinear_results)
    print(f"controlled comparison rendered {figure}", flush=True)
    return {"figures": [str(figure)], "metrics": str(metrics), "fields": str(fields)}


if __name__ == "__main__" or __name__ == "ch29":
    scope = os.environ.get("KWAVERS_CH29_RENDER_SCOPE", "all").lower()
    if scope == "fig05":
        run_fig05_only()
    elif scope == "comparison":
        run_comparison_only()
    else:
        run()

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from seismic_fwi.histotripsy_plots import ReportConfig, plot_metrics, plot_passive_bands, plot_scenarios, write_metrics

REPO_ROOT = Path(__file__).resolve().parents[4]
for candidate in (
    REPO_ROOT / "target" / "release" / "pykwavers.dll",
    REPO_ROOT / "target" / "maturin" / "pykwavers.dll",
    REPO_ROOT / "target" / "debug" / "pykwavers.dll",
):
    if candidate.exists():
        os.environ["PYKWAVERS_EXTENSION_PATH"] = str(candidate)
        break
PY_PACKAGE = REPO_ROOT / "pykwavers" / "python"
if str(PY_PACKAGE) not in sys.path:
    sys.path.insert(0, str(PY_PACKAGE))
import pykwavers as kw  # noqa: E402

OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch27"
CT_PATH = REPO_ROOT / "data" / "rire_patient_109" / "patient_109_ct.nii.gz"
GRID_SIZE = int(os.environ.get("KWAVERS_CH27_HIST_GRID_SIZE", "64"))
ELEMENT_COUNT = 1024
RADIUS_M = float(os.environ.get("KWAVERS_CH27_HIST_RADIUS_M", "0.11"))
ACTIVE_FREQUENCIES_HZ = tuple(float(v) for v in os.environ.get("KWAVERS_CH27_HIST_ACTIVE_FREQ_HZ", "110000,160000,220000").split(","))
PASSIVE_FREQUENCIES_HZ = tuple(float(v) for v in os.environ.get("KWAVERS_CH27_HIST_PASSIVE_FREQ_HZ", "110000,220000,440000").split(","))
RECEIVER_OFFSETS = tuple(int(v) for v in os.environ.get("KWAVERS_CH27_HIST_RECEIVER_OFFSETS", "512,384,640,256").split(","))
NOISE_SNR_DB = float(os.environ.get("KWAVERS_CH27_HIST_NOISE_SNR_DB", "36"))
GAIN_JITTER_STD = float(os.environ.get("KWAVERS_CH27_HIST_GAIN_JITTER_STD", "0.02"))
PHASE_JITTER_RAD = float(os.environ.get("KWAVERS_CH27_HIST_PHASE_JITTER_RAD", "0.025"))
NOISE_SEED = int(os.environ.get("KWAVERS_CH27_HIST_NOISE_SEED", "1729"))
C_REF = 1540.0
RHO_REF = 1000.0
BETA_REF = 4.5
SOURCE_PRESSURE_PA = 0.15e6
SOFT_ATTENUATION_NP_PER_M_MHZ = 0.5 * 100.0 * np.log(10.0) / 20.0
SKULL_ATTENUATION_NP_PER_M_MHZ = 70.0
HUBER_GAUSSIAN_EFFICIENCY_K = 1.345


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    center_shift_mm: tuple[float, float]
    radii_mm: tuple[float, float]
    rotation_deg: float
    delta_c_m_s: float
    attenuation_delta: float
    beta_delta: float
    passive_exponent: float


@dataclass(frozen=True)
class Operators:
    fundamental: np.ndarray
    attenuation: np.ndarray
    harmonic_speed: np.ndarray
    harmonic_beta: np.ndarray
    frequency_slices: list[slice]
    passive: np.ndarray
    passive_bands: list[slice]


@dataclass
class ScenarioResult:
    scenario: Scenario
    lesion: np.ndarray
    active_rtm: np.ndarray
    linear_fwi: np.ndarray
    multiparameter_fwi: np.ndarray
    nonlinear_fwi: np.ndarray
    subharmonic_source: np.ndarray
    fused: np.ndarray
    objective_linear: list[float]
    objective_multiparameter: list[float]
    objective_nonlinear: list[float]
    objective_subharmonic: list[float]
    metrics: dict[str, dict[str, float]]
    passive_bands: list[np.ndarray]


def load_baseline() -> dict:
    if not CT_PATH.exists():
        raise FileNotFoundError(f"RIRE CT not found: {CT_PATH}")
    return kw.run_seismic_helmet_fwi_from_ritk_ct(
        str(CT_PATH),
        grid_size=GRID_SIZE,
        iterations=1,
        frequencies_hz=[max(ACTIVE_FREQUENCIES_HZ)],
        receiver_offsets=list(RECEIVER_OFFSETS),
        frequency_continuation=False,
        attenuation_model=True,
        nonlinear_harmonic_model=False,
    )


def centered_coordinates(shape: tuple[int, int], spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(shape[0], dtype=float) - 0.5 * (shape[0] - 1)) * spacing_m
    y = (np.arange(shape[1], dtype=float) - 0.5 * (shape[1] - 1)) * spacing_m
    return np.meshgrid(x, y, indexing="ij")


def ring_geometry() -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, ELEMENT_COUNT, endpoint=False)
    return RADIUS_M * np.cos(theta), RADIUS_M * np.sin(theta)


def bilinear_sample(image: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, spacing_m: float) -> np.ndarray:
    nx, ny = image.shape
    ix = x_m / spacing_m + 0.5 * (nx - 1)
    iy = y_m / spacing_m + 0.5 * (ny - 1)
    inside = (ix >= 0.0) & (iy >= 0.0) & (ix <= nx - 1) & (iy <= ny - 1)
    x0 = np.clip(np.floor(ix).astype(np.int64), 0, nx - 1)
    y0 = np.clip(np.floor(iy).astype(np.int64), 0, ny - 1)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    tx = ix - x0
    ty = iy - y0
    values = (1.0 - tx) * (1.0 - ty) * image[x0, y0] + tx * (1.0 - ty) * image[x1, y0]
    values += (1.0 - tx) * ty * image[x0, y1] + tx * ty * image[x1, y1]
    return np.where(inside, values, 0.0)


def attenuation_map(ct_hu: np.ndarray, skull_mask: np.ndarray) -> np.ndarray:
    alpha = np.full(ct_hu.shape, SOFT_ATTENUATION_NP_PER_M_MHZ, dtype=np.float32)
    alpha[ct_hu < -300.0] = 0.002
    alpha[skull_mask] = SKULL_ATTENUATION_NP_PER_M_MHZ
    return alpha


def attenuation_integrals(alpha: np.ndarray, active_xy_m: np.ndarray, spacing_m: float, samples: int = 10) -> np.ndarray:
    ex, ey = ring_geometry()
    vx, vy = active_xy_m[:, 0], active_xy_m[:, 1]
    out = np.zeros((ELEMENT_COUNT, active_xy_m.shape[0]), dtype=np.float32)
    for element in range(ELEMENT_COUNT):
        dx = vx - ex[element]
        dy = vy - ey[element]
        length = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        integral = np.zeros(active_xy_m.shape[0], dtype=np.float32)
        for step in range(samples):
            t = (step + 0.5) / samples
            integral += bilinear_sample(alpha, ex[element] + t * dx, ey[element] + t * dy, spacing_m).astype(np.float32)
        out[element, :] = integral * (length / samples)
    return out


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    np.divide(matrix, np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1.0e-12), out=matrix)
    return matrix


def build_active_operators(active_xy_m: np.ndarray, integrals: np.ndarray, spacing_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[slice]]:
    sx, sy = ring_geometry()
    nvox = active_xy_m.shape[0]
    rows_per_frequency = ELEMENT_COUNT * len(RECEIVER_OFFSETS)
    rows = rows_per_frequency * len(ACTIVE_FREQUENCIES_HZ)
    fundamental = np.empty((rows, nvox), dtype=np.float32)
    attenuation = np.empty((rows, nvox), dtype=np.float32)
    harmonic_speed = np.empty((rows, nvox), dtype=np.float32)
    harmonic_beta = np.empty((rows, nvox), dtype=np.float32)
    vx, vy = active_xy_m[:, 0], active_xy_m[:, 1]
    row = 0
    slices: list[slice] = []
    for frequency_hz in ACTIVE_FREQUENCIES_HZ:
        start = row
        frequency_mhz = frequency_hz * 1.0e-6
        k1 = 2.0 * np.pi * frequency_hz / C_REF
        omega = 2.0 * np.pi * frequency_hz
        shock_distance = RHO_REF * C_REF**3 / (BETA_REF * omega * SOURCE_PRESSURE_PA)
        for source in range(ELEMENT_COUNT):
            ds = np.sqrt((vx - sx[source]) ** 2 + (vy - sy[source]) ** 2)
            for offset in RECEIVER_OFFSETS:
                receiver = (source + offset) % ELEMENT_COUNT
                dr = np.sqrt((vx - sx[receiver]) ** 2 + (vy - sy[receiver]) ** 2)
                total_distance = ds + dr
                spread = np.maximum(np.sqrt(ds * dr), 1.0e-6)
                path = integrals[source] + integrals[receiver]
                base = spacing_m * spacing_m * np.exp(-path * frequency_mhz) * np.cos(k1 * total_distance) / spread
                h2 = 0.25 * total_distance / shock_distance
                harmonic = spacing_m * spacing_m * h2 * np.exp(-path * (2.0 * frequency_mhz)) * np.cos(2.0 * k1 * total_distance) / spread
                fundamental[row, :] = base
                attenuation[row, :] = -path * frequency_mhz * base
                harmonic_speed[row, :] = harmonic
                harmonic_beta[row, :] = harmonic * (1.0 + 0.15 * h2)
                row += 1
        slices.append(slice(start, row))
    return tuple(row_normalize(m) for m in (fundamental, attenuation, harmonic_speed, harmonic_beta)) + (slices,)


def build_passive_matrix(active_xy_m: np.ndarray, integrals: np.ndarray, spacing_m: float) -> tuple[np.ndarray, list[slice]]:
    sx, sy = ring_geometry()
    vx, vy = active_xy_m[:, 0], active_xy_m[:, 1]
    rows = ELEMENT_COUNT * len(PASSIVE_FREQUENCIES_HZ)
    matrix = np.empty((rows, active_xy_m.shape[0]), dtype=np.float32)
    band_slices: list[slice] = []
    row = 0
    for frequency_hz in PASSIVE_FREQUENCIES_HZ:
        start = row
        k = 2.0 * np.pi * frequency_hz / C_REF
        frequency_mhz = frequency_hz * 1.0e-6
        for receiver in range(ELEMENT_COUNT):
            dr = np.sqrt((vx - sx[receiver]) ** 2 + (vy - sy[receiver]) ** 2)
            attenuation = np.exp(-integrals[receiver] * frequency_mhz)
            matrix[row, :] = spacing_m * spacing_m * attenuation * np.cos(k * dr) / np.maximum(np.sqrt(dr), 1.0e-6)
            row += 1
        band_slices.append(slice(start, row))
    return row_normalize(matrix), band_slices


def build_operators(active_xy_m: np.ndarray, integrals: np.ndarray, spacing_m: float) -> Operators:
    fundamental, attenuation, harmonic_speed, harmonic_beta, frequency_slices = build_active_operators(active_xy_m, integrals, spacing_m)
    passive, passive_bands = build_passive_matrix(active_xy_m, integrals, spacing_m)
    return Operators(fundamental, attenuation, harmonic_speed, harmonic_beta, frequency_slices, passive, passive_bands)


def smooth_block(values: np.ndarray, active: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = np.zeros(shape, dtype=np.float32)
    image[active[:, 0], active[:, 1]] = values.astype(np.float32)
    smoothed = image.copy()
    count = np.ones(shape, dtype=np.float32)
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        smoothed += np.roll(image, shift=shift, axis=axis)
        count += 1.0
    return (smoothed / count)[active[:, 0], active[:, 1]]


def continuation_rows(stage: int, frequency_slices: list[slice]) -> np.ndarray:
    return np.r_[tuple(np.arange(s.start, s.stop) for s in frequency_slices[: stage + 1])]


def robust_weights(residual: np.ndarray) -> tuple[np.ndarray, float]:
    centered = residual - np.median(residual)
    sigma = np.median(np.abs(centered)) / 0.6744897501960817
    c = max(HUBER_GAUSSIAN_EFFICIENCY_K * sigma, 1.0e-8)
    absr = np.abs(residual)
    weights = np.where(absr <= c, 1.0, c / np.maximum(absr, 1.0e-12)).astype(np.float32)
    objective = np.where(absr <= c, 0.5 * residual * residual, c * (absr - 0.5 * c))
    return weights, float(np.sum(objective))


def solve_grouped_continuation(
    groups: list[tuple[list[np.ndarray | None], np.ndarray, float]],
    frequency_slices: list[slice],
    active: np.ndarray,
    shape: tuple[int, int],
    parameter_count: int,
    iterations_per_stage: int,
    regularization: float,
) -> tuple[np.ndarray, list[float]]:
    nvox = next(block for block in groups[0][0] if block is not None).shape[1]
    model = np.zeros(parameter_count * nvox, dtype=np.float32)
    history: list[float] = []
    for stage in range(len(frequency_slices)):
        rows = continuation_rows(stage, frequency_slices)
        for _ in range(iterations_per_stage):
            gradient = np.zeros_like(model)
            objective = 0.5 * regularization * float(model @ model)
            for blocks, data, scale in groups:
                prediction = np.zeros(rows.size, dtype=np.float32)
                for p, block in enumerate(blocks):
                    if block is not None:
                        prediction += scale * (block[rows] @ model[p * nvox : (p + 1) * nvox])
                residual = scale * data[rows] - prediction
                weights, robust_objective = robust_weights(residual)
                objective += robust_objective
                weighted = weights * residual
                for p, block in enumerate(blocks):
                    if block is not None:
                        gradient[p * nvox : (p + 1) * nvox] += scale * (block[rows].T @ weighted)
            gradient -= regularization * model
            direction = np.zeros_like(model)
            for p in range(parameter_count):
                block = gradient[p * nvox : (p + 1) * nvox]
                direction[p * nvox : (p + 1) * nvox] = 0.65 * block + 0.35 * smooth_block(block, active, shape)
            numerator = float(direction @ gradient)
            denominator = regularization * float(direction @ direction)
            for blocks, data, scale in groups:
                projected = np.zeros(rows.size, dtype=np.float32)
                residual = scale * data[rows]
                for p, block in enumerate(blocks):
                    if block is not None:
                        residual -= scale * (block[rows] @ model[p * nvox : (p + 1) * nvox])
                        projected += scale * (block[rows] @ direction[p * nvox : (p + 1) * nvox])
                weights, _ = robust_weights(residual)
                denominator += float(weights @ (projected * projected))
            history.append(objective)
            if denominator <= 0.0 or numerator <= 0.0:
                break
            model += np.float32(numerator / denominator) * direction
    return model, history


def vector_to_image(values: np.ndarray, active: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = np.zeros(shape, dtype=np.float32)
    image[active[:, 0], active[:, 1]] = values.astype(np.float32)
    return image


def normalize_positive(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = image[mask]
    lo = float(np.percentile(values, 2.0))
    hi = float(np.percentile(values, 98.0))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.where(mask, np.clip((image - lo) / (hi - lo), 0.0, 1.0), 0.0).astype(np.float32)


def noisy(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = np.linalg.norm(data) / max(np.sqrt(data.size), 1.0) * 10.0 ** (-NOISE_SNR_DB / 20.0)
    gain = 1.0 + rng.normal(0.0, GAIN_JITTER_STD, data.shape)
    phase = rng.normal(0.0, PHASE_JITTER_RAD, data.shape) * np.maximum(np.abs(data), sigma)
    return (gain * data + phase + rng.normal(0.0, sigma, data.shape)).astype(np.float32)


def scenario_field(scenario: Scenario, mask: np.ndarray, spacing_m: float) -> np.ndarray:
    xx, yy = centered_coordinates(mask.shape, spacing_m)
    points = np.argwhere(mask)
    center = points.mean(axis=0)
    cx = (center[0] - 0.5 * (mask.shape[0] - 1)) * spacing_m + scenario.center_shift_mm[0] * 1.0e-3
    cy = (center[1] - 0.5 * (mask.shape[1] - 1)) * spacing_m + scenario.center_shift_mm[1] * 1.0e-3
    theta = np.deg2rad(scenario.rotation_deg)
    xr = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
    yr = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
    rx, ry = scenario.radii_mm[0] * 1.0e-3, scenario.radii_mm[1] * 1.0e-3
    return np.where(mask, np.exp(-0.5 * ((xr / rx) ** 2 + (yr / ry) ** 2)), 0.0).astype(np.float32)


def rtm_image(matrix: np.ndarray, data: np.ndarray, active: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    diag = np.sum(matrix * matrix, axis=0) + 1.0e-6
    return vector_to_image((matrix.T @ data) / diag, active, shape)


def passive_correlation_images(matrix: np.ndarray, data: np.ndarray, band_slices: list[slice], active: np.ndarray, shape: tuple[int, int], mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    bands = [normalize_positive(rtm_image(matrix[band], data[band], active, shape), mask) for band in band_slices]
    product = np.ones(shape, dtype=np.float32)
    for image in bands:
        product *= np.maximum(image, 1.0e-6)
    return normalize_positive(product ** (1.0 / len(bands)), mask), bands


def auprc(score: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    y = target[mask].astype(bool)
    s = score[mask].astype(float)
    order = np.argsort(-s)
    y = y[order]
    positives = max(int(y.sum()), 1)
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    recall = np.concatenate(([0.0], tp / positives))
    precision = np.concatenate(([1.0], tp / np.maximum(tp + fp, 1)))
    return float(np.trapezoid(precision, recall))


def dice_at_target_area(score: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    y = target & mask
    count = int(y.sum())
    if count == 0:
        return 0.0
    threshold = np.partition(score[mask], -count)[-count]
    pred = (score >= threshold) & mask
    return float(2 * np.logical_and(pred, y).sum() / max(pred.sum() + y.sum(), 1))


def cnr(score: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    lesion = score[target & mask]
    background = score[(~target) & mask]
    return float((lesion.mean() - background.mean()) / max(background.std(), 1.0e-12))


def evaluate_maps(maps: dict[str, np.ndarray], target: np.ndarray, mask: np.ndarray) -> dict[str, dict[str, float]]:
    return {name: {"auprc": auprc(image, target, mask), "dice_equal_area": dice_at_target_area(image, target, mask), "cnr": cnr(image, target, mask)} for name, image in maps.items()}


def subharmonic_rows(passive_bands: list[slice]) -> tuple[np.ndarray, float]:
    reference = max(ACTIVE_FREQUENCIES_HZ)
    matches = [(f, b) for f, b in zip(PASSIVE_FREQUENCIES_HZ, passive_bands) if np.isclose(f, 0.5 * reference, rtol=0.02)]
    if not matches:
        raise ValueError("No passive subharmonic band matches the highest active carrier")
    return np.r_[tuple(np.arange(b.start, b.stop) for _, b in matches)], max(f for f, _ in matches)


def reconstruct_scenario(index: int, scenario: Scenario, mask: np.ndarray, spacing_m: float, active: np.ndarray, operators: Operators) -> ScenarioResult:
    nvox = active.shape[0]
    lesion = scenario_field(scenario, mask, spacing_m)
    target = lesion >= 0.45
    speed = (scenario.delta_c_m_s * lesion)[active[:, 0], active[:, 1]].astype(np.float32) / C_REF
    alpha = (scenario.attenuation_delta * lesion)[active[:, 0], active[:, 1]].astype(np.float32)
    beta = (scenario.beta_delta * lesion)[active[:, 0], active[:, 1]].astype(np.float32)
    cavitation = (lesion[active[:, 0], active[:, 1]] ** scenario.passive_exponent * (1.0 + 0.25 * np.maximum(beta, 0.0))).astype(np.float32)
    clean_fundamental = operators.fundamental @ speed + operators.attenuation @ alpha
    clean_harmonic = operators.harmonic_speed @ speed + operators.harmonic_beta @ beta
    clean_passive = operators.passive @ cavitation
    rng = np.random.default_rng(NOISE_SEED + index)
    d1, d2, dp = noisy(clean_fundamental, rng), noisy(clean_harmonic, rng), noisy(clean_passive, rng)
    active_rtm = normalize_positive(-rtm_image(operators.fundamental, d1, active, mask.shape), mask)
    linear, h_linear = solve_grouped_continuation([([operators.fundamental], d1, 1.0)], operators.frequency_slices, active, mask.shape, 1, 8, 1.0e-3)
    multiparam, h_multi = solve_grouped_continuation([([operators.fundamental, operators.attenuation], d1, 1.0)], operators.frequency_slices, active, mask.shape, 2, 8, 8.0e-4)
    scale = np.linalg.norm(d1) / max(np.linalg.norm(d2), 1.0e-12)
    nonlinear, h_non = solve_grouped_continuation(
        [([operators.fundamental, operators.attenuation, None], d1, 1.0), ([operators.harmonic_speed, None, operators.harmonic_beta], d2, float(scale))],
        operators.frequency_slices,
        active,
        mask.shape,
        3,
        10,
        7.0e-4,
    )
    sub_rows, sub_frequency = subharmonic_rows(operators.passive_bands)
    sub_model, h_sub = solve_grouped_continuation([([operators.passive[sub_rows]], dp[sub_rows], 1.0)], [slice(0, sub_rows.size)], active, mask.shape, 1, 20, 5.0e-4)
    passive_rtm, band_maps = passive_correlation_images(operators.passive, dp, operators.passive_bands, active, mask.shape, mask)
    linear_map = normalize_positive(-vector_to_image(linear, active, mask.shape), mask)
    speed_multi = normalize_positive(-vector_to_image(multiparam[:nvox], active, mask.shape), mask)
    attenuation_multi = normalize_positive(vector_to_image(multiparam[nvox:], active, mask.shape), mask)
    multiparameter_map = normalize_positive(0.75 * speed_multi + 0.25 * attenuation_multi, mask)
    speed_non = normalize_positive(-vector_to_image(nonlinear[:nvox], active, mask.shape), mask)
    beta_non = normalize_positive(vector_to_image(nonlinear[2 * nvox :], active, mask.shape), mask)
    nonlinear_map = normalize_positive(0.65 * speed_non + 0.35 * beta_non, mask)
    sub_map = normalize_positive(vector_to_image(sub_model, active, mask.shape), mask)
    gate_floor = np.sqrt(sub_frequency / max(ACTIVE_FREQUENCIES_HZ))
    fused = normalize_positive(nonlinear_map * (gate_floor + (1.0 - gate_floor) * sub_map), mask)
    maps = {"active_rtm": active_rtm, "linear_fwi": linear_map, "multiparameter_fwi": multiparameter_map, "nonlinear_fwi": nonlinear_map, "passive_rtm": passive_rtm, "subharmonic_source": sub_map, "fused": fused}
    return ScenarioResult(scenario, lesion, active_rtm, linear_map, multiparameter_map, nonlinear_map, sub_map, fused, h_linear, h_multi, h_non, h_sub, evaluate_maps(maps, target, mask), band_maps)


def run() -> list[ScenarioResult]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline()
    ct = np.asarray(baseline["ct_hu"], dtype=float)
    mask = np.asarray(baseline["brain_mask"], dtype=bool)
    skull = np.asarray(baseline["skull_mask"], dtype=bool)
    spacing_m = float(baseline["spacing_m"])
    xx, yy = centered_coordinates(mask.shape, spacing_m)
    active = np.argwhere(mask)
    active_xy_m = np.column_stack((xx[mask], yy[mask])).astype(np.float32)
    print("[ch27-hist] building CT-derived path attenuation integrals")
    integrals = attenuation_integrals(attenuation_map(ct, skull), active_xy_m, spacing_m)
    print("[ch27-hist] building 1024-element continuation, multiparameter, nonlinear, passive operators")
    operators = build_operators(active_xy_m, integrals, spacing_m)
    scenarios = [
        Scenario("compact", "compact intrinsic", (-7.0, -2.0), (4.8, 5.5), 20.0, -55.0, 0.18, 0.55, 1.0),
        Scenario("elongated", "shock elongated", (1.0, 0.0), (4.0, 13.0), -8.0, -72.0, 0.28, 0.95, 0.85),
        Scenario("multifocal", "multi-packet", (8.0, 4.0), (7.0, 4.0), 48.0, -48.0, 0.22, 0.70, 1.15),
    ]
    results = [reconstruct_scenario(i, scenario, mask, spacing_m, active, operators) for i, scenario in enumerate(scenarios)]
    report = ReportConfig(OUT_DIR, REPO_ROOT, CT_PATH, GRID_SIZE, ELEMENT_COUNT, ACTIVE_FREQUENCIES_HZ, PASSIVE_FREQUENCIES_HZ, RECEIVER_OFFSETS, NOISE_SNR_DB, GAIN_JITTER_STD, PHASE_JITTER_RAD)
    plot_scenarios(results, baseline, report)
    plot_metrics(results, report)
    plot_passive_bands(results[1], baseline, report)
    write_metrics(results, baseline, report)
    return results

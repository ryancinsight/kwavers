#!/usr/bin/env python
"""Generate the committed k-Wave differential-oracle reference fields.

The kwavers README states pseudospectral parity against k-Wave. This script is
the reproducible provenance of that claim: it runs `k-wave-python` (the Python
binding over the reference C++ `kspaceFirstOrder-OMP` binary) on a fixed set of
homogeneous-water initial-value problems and writes the resulting fields into
`crates/kwavers/tests/reference/kwave/` as NPZ archives with a JSON manifest.

The committed artifacts are consumed by the Rust differential test
`crates/kwavers/tests/kwave_reference_parity.rs`, which runs the kwavers PSTD
solver on the identical discretization and asserts value-semantic agreement.
The Rust test therefore runs in the default gate from a clean clone; this
script is required only to regenerate or extend the reference set.

Case design
-----------
Every case is a lossless homogeneous-water initial-value problem with a
band-limited Gaussian initial pressure. Three properties make the comparison an
oracle rather than a coincidence:

* Identical discretization. The manifest records the exact `dt` and step count
  that k-Wave used, so the Rust solver advances the same number of steps of the
  same size over the same grid. The only remaining difference is the scheme.
* Boundary-free comparison window. `t_end` is chosen so the wavefront travels a
  known distance that stays clear of the absorbing layer, and the manifest
  records the interior radius within which the two codes are compared. Neither
  code's perfectly matched layer enters the compared region, so a divergence
  there is a scheme divergence and not a boundary-treatment difference.
* No reference-side preprocessing. `smooth_p0` is disabled. k-Wave otherwise
  applies a Blackman window to `p0` before the first step, which would make the
  two codes solve different initial-value problems. The seed is already
  band-limited by construction (sigma = 3 dx), so the smoothing that option
  exists to provide is not needed.

Usage
-----
    python scripts/generate_kwave_reference.py [--out DIR] [--case NAME ...]

Requires `k-wave-python` and its `kspaceFirstOrder-OMP` binary; both are
external to this repository by design. Vendoring the reference solver is a
non-goal; reproducing and committing its output is the point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "crates" / "kwavers" / "tests" / "reference" / "kwave"

# Water at 20 degrees Celsius. Both codes are driven with exactly these values.
SOUND_SPEED_M_S = 1500.0
DENSITY_KG_M3 = 1000.0

# Uniform grid spacing for every case (0.1 mm).
DX_M = 1.0e-4

# Courant number handed to `kgrid.makeTime`. The k-Wave default is 0.3; 0.15 is
# used here so the same time step is also comfortably inside the kwavers
# three-dimensional k-space leapfrog stability bound dt <= dx / (c pi sqrt(3)),
# about 0.184 dx / c, making one shared time step valid for both codes in every
# case.
CFL = 0.15

# Gaussian seed width in cells. At sigma = 3 dx the spectrum is negligible above
# a tenth of the Nyquist wavenumber, so neither code is resolving content near
# the aliasing limit and reference-side smoothing is unnecessary.
SIGMA_CELLS = 3.0

# Peak initial pressure in pascals. The problem is linear and lossless, so this
# is a pure scale factor.
P0_PEAK_PA = 1.0

# Width in cells of the hyperbolic-tangent transition in a layered medium.
LAYER_TRANSITION_CELLS = 2.0


@dataclass(frozen=True)
class Case:
    """One reference case: a grid shape and the time the wave is propagated."""

    name: str
    shape: tuple[int, ...]
    # Propagation time in seconds, before rounding to a whole number of steps.
    t_end_s: float
    # Radius in cells, measured from the grid centre, inside which the Rust
    # differential test compares the two fields. Chosen so that the absorbing
    # layer of neither code overlaps the window at the final time.
    compare_radius_cells: int
    # Power-law absorption `alpha(f) = alpha_coeff * f**alpha_power`, in
    # k-Wave's own units of dB/(MHz**alpha_power cm). `None` is a lossless
    # medium; both codes take the coefficient in these units unconverted.
    alpha_coeff_db: float | None = None
    alpha_power: float | None = None
    # A layered medium: sound speed and density step from their water values to
    # these across a smoothed interface. `None` is a uniform medium.
    layer_sound_speed_m_s: float | None = None
    layer_density_kg_m3: float | None = None
    # Interface position, in cells along the first axis.
    layer_interface_cell: int | None = None
    # A time-varying point pressure source at this cell, driving a
    # Gaussian-windowed tone burst instead of an initial pressure.
    source_cell: tuple[int, ...] | None = None
    source_frequency_hz: float | None = None
    source_centre_s: float | None = None
    source_width_s: float | None = None


# k-wave-python exposes `kspaceFirstOrder2D`, `kspaceFirstOrder3D`, and the
# axisymmetric `kspaceFirstOrderAS`; it ships no one-dimensional solver, so a
# one-dimensional case has no reference to compare against here and is not
# listed. The axisymmetric case is a separate geometry rather than a further
# Cartesian dimension and is left to a follow-up.
CASES: tuple[Case, ...] = (
    # Two dimensions: 64 by 64 spans 6.4 mm. In 1.0 us the wavefront reaches
    # radius 15 cells; the 24-cell window contains it with 8 cells of clearance.
    Case("ivp_homogeneous_2d", (64, 64), 1.0e-6, 24),
    # Three dimensions: 32 cubed spans 3.2 mm. In 0.5 us the wavefront reaches
    # radius 7.5 cells; the 12-cell window clears the 4-cell edge margin.
    Case("ivp_homogeneous_3d", (32, 32, 32), 0.5e-6, 12),
    # The lossless two-dimensional case with power-law absorption switched on,
    # so the pair isolates the absorption model: identical grid, seed, time
    # step, and step count, one variable changed.
    #
    # `alpha_coeff` is deliberately far above tissue (which runs 0.5 to 1.5).
    # The seed's dominant content sits near 0.8 MHz and the wave travels only
    # 1.5 mm, so a tissue coefficient would attenuate by under one percent --
    # inside the noise of the comparison and unable to distinguish a correct
    # absorption model from none at all. At 40 the same path attenuates by
    # roughly a third, which the differential test asserts explicitly against
    # the lossless field. Both codes receive the identical coefficient, so the
    # value's realism is irrelevant to what the comparison establishes.
    Case("ivp_absorbing_2d", (64, 64), 1.0e-6, 24, alpha_coeff_db=40.0, alpha_power=1.5),
    # The lossless two-dimensional case with a layered medium, so this pair
    # isolates spatially varying sound speed and density exactly as the
    # absorbing pair isolates absorption.
    #
    # The interface sits 8 cells right of the seed. The wavefront reaches it at
    # about step 53 of 100, so the recorded field contains the transmitted wave,
    # the reflection travelling back through the seed, and the refraction of the
    # curved front -- all of which depend on both varying fields, and none of
    # which a solver that ignored either could produce.
    #
    # The step is smoothed over two cells rather than left sharp. Both codes are
    # pseudospectral and a discontinuity rings in both; smoothing keeps the
    # medium as band-limited as the seed, so the comparison measures the
    # heterogeneous propagation rather than two Gibbs phenomena.
    # The grid is deliberately non-square. Every other case is square, which
    # makes an axis transpose a no-op on the shape and therefore silent; k-Wave
    # returns the recorded field with its axes reversed, so a square case cannot
    # tell a correct orientation from a transposed one. Here it is a shape
    # error, checked at generation.
    Case(
        "ivp_layered_2d",
        (80, 64),
        1.0e-6,
        24,
        layer_sound_speed_m_s=1800.0,
        layer_density_kg_m3=1200.0,
        layer_interface_cell=48,
    ),
    # A time-varying point source instead of an initial pressure. Every other
    # case seeds `p0` and lets the field evolve, which never touches the source
    # injection path -- the mask indexing, the per-step signal lookup, and
    # k-Wave's source scaling and k-space source correction are all untested by
    # them, and they are the path a real driven simulation runs on.
    #
    # The source sits off-centre on both axes so the field is asymmetric in
    # every direction: a non-square grid catches a transposition, but only an
    # off-centre source catches a flip.
    Case(
        "src_tone_burst_2d",
        (96, 80),
        1.5e-6,
        30,
        source_cell=(40, 32),
        source_frequency_hz=3.0e6,
        source_centre_s=3.0e-7,
        source_width_s=1.0e-7,
    ),
)


def tone_burst(case: Case, dt_s: float, steps: int) -> np.ndarray:
    """Return the Gaussian-windowed sine the point source emits, one row.

    The envelope keeps the burst band-limited for the same reason the initial
    pressure is a Gaussian rather than a delta: at `sigma = 100 ns` the spectrum
    spans roughly 1.4 to 4.6 MHz, whose shortest wavelength is a little over
    three cells and so is resolved rather than aliased.
    """
    t = np.arange(steps, dtype=np.float64) * dt_s
    offset = t - case.source_centre_s
    envelope = np.exp(-((offset / case.source_width_s) ** 2))
    carrier = np.sin(2.0 * np.pi * case.source_frequency_hz * offset)
    return (P0_PEAK_PA * envelope * carrier).reshape(1, steps)


def layered_profile(case: Case, base: float, layer: float) -> np.ndarray:
    """Return the case's medium profile, stepping from `base` to `layer`.

    The transition is a hyperbolic tangent two cells wide, centred on the
    interface cell, so the profile carries no content above the seed's own band
    limit and neither code is asked to resolve a discontinuity.
    """
    axes = [np.arange(n, dtype=np.float64) for n in case.shape]
    grids = np.meshgrid(*axes, indexing="ij")
    interface = float(case.layer_interface_cell)
    blend = 0.5 * (1.0 + np.tanh((grids[0] - interface) / LAYER_TRANSITION_CELLS))
    return base + (layer - base) * blend


def gaussian_seed(shape: tuple[int, ...]) -> np.ndarray:
    """Return an isotropic Gaussian initial pressure centred on the grid.

    The centre is placed at ``n // 2`` on each axis, which is the same cell the
    Rust test seeds, so the two initial conditions are bit-identical rather than
    merely similar.
    """
    axes = [np.arange(n, dtype=np.float64) - float(n // 2) for n in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    radius_squared = sum(grid**2 for grid in grids)
    return P0_PEAK_PA * np.exp(-radius_squared / (2.0 * SIGMA_CELLS**2))


def run_case(case: Case) -> dict[str, object]:
    """Run one case through k-Wave and return its fields and discretization."""
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions

    dims = len(case.shape)
    if dims == 2:
        from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D as solver
    elif dims == 3:
        from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D as solver
    else:
        raise ValueError(f"unsupported dimensionality {dims}")

    kgrid = kWaveGrid(list(case.shape), [DX_M] * dims)
    if case.layer_interface_cell is None:
        sound_speed: float | np.ndarray = SOUND_SPEED_M_S
        density: float | np.ndarray = DENSITY_KG_M3
    else:
        sound_speed = layered_profile(case, SOUND_SPEED_M_S, case.layer_sound_speed_m_s)
        density = layered_profile(case, DENSITY_KG_M3, case.layer_density_kg_m3)

    medium = kWaveMedium(
        sound_speed=sound_speed,
        density=density,
        alpha_coeff=case.alpha_coeff_db,
        alpha_power=case.alpha_power,
    )
    # `makeTime` sizes the step from the fastest speed present, which is what
    # bounds stability; passing the array directly lets k-Wave take its maximum.
    kgrid.makeTime(medium.sound_speed, cfl=CFL, t_end=case.t_end_s)

    source = kSource()
    if case.source_cell is None:
        p0 = gaussian_seed(case.shape)
        source.p0 = p0
        signal = None
    else:
        # A driven case has no initial pressure; `p0` is stored as the source
        # mask so the manifest still records exactly what the run was given.
        p0 = np.zeros(case.shape, dtype=np.float64)
        p0[case.source_cell] = 1.0
        source.p_mask = p0
        # k-Wave indexes `source.p` by time step, so it needs one column per
        # step it will take, which is `Nt` -- one more than the propagation
        # intervals the manifest records.
        signal = tone_burst(case, float(np.asarray(kgrid.dt).item()),
                            int(np.asarray(kgrid.Nt).item()))
        source.p = signal
        source.p_mode = "additive"

    # Record the whole final field: a full-field oracle discriminates far more
    # than a sensor trace, and at these grid sizes it is still a small artifact.
    sensor = kSensor(mask=np.ones(case.shape, dtype=bool), record=["p_final"])

    simulation_options = SimulationOptions(
        pml_inside=False,
        smooth_p0=False,
        save_to_disk=True,
        data_cast="single",
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=False)

    output = solver(
        kgrid=kgrid,
        medium=medium,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=execution_options,
    )

    # k-Wave returns the recorded field with its axes reversed relative to the
    # grid it was given: a (Nx=64, Ny=48) problem comes back shaped (48, 64).
    # Reversing the axes restores the input order. A square case would hide the
    # difference behind a no-op reshape and store a transposed field, which is
    # invisible while the seed and medium are symmetric under transpose and
    # silently wrong the moment either is not -- so the orientation is asserted
    # rather than assumed.
    p_final = np.asarray(output["p_final"], dtype=np.float64)
    if p_final.shape == tuple(reversed(case.shape)):
        p_final = p_final.transpose()
    if p_final.shape != case.shape:
        raise ValueError(
            f"{case.name}: k-Wave returned {p_final.shape}, which is neither the "
            f"grid shape {case.shape} nor its reverse"
        )
    # How many propagation intervals separate the returned field from the start
    # depends on which kind of source drove it, and the two differ by one.
    #
    # `kgrid.Nt` is k-Wave's count of time *points*, spanning
    # `t_array = (0 : Nt - 1) * dt`. An initial-value case has a meaningful
    # state at `t = 0` -- the seed itself -- so the first of those points is the
    # initial condition and the returned field is `Nt - 1` intervals later.
    # Recording `Nt` instead over-propagates by one step, which costs five
    # orders of magnitude of agreement.
    #
    # A driven case starts from a zero field, which is not a state worth
    # counting: k-Wave performs an update for every one of its `Nt` points and
    # the returned field is `Nt` intervals later. Recording `Nt - 1` here
    # under-propagates by one step and costs two orders of agreement.
    #
    # Both were measured, not assumed. The manifest therefore records the
    # interval count for the case at hand rather than one rule for both, and
    # the differential test's step-count guard fails if either is wrong.
    return {
        "p0": p0,
        "p_signal": signal,
        "p_final": p_final,
        "dt_s": float(np.asarray(kgrid.dt).item()),
        "steps": (
            int(np.asarray(kgrid.Nt).item())
            if case.source_cell is not None
            else int(np.asarray(kgrid.Nt).item()) - 1
        ),
    }


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kwave_provenance() -> dict[str, str]:
    """Return the reference-solver identity recorded alongside the fields."""
    import kwave

    return {
        "package": "k-wave-python",
        "version": getattr(kwave, "__version__", "unknown"),
        "binary": "kspaceFirstOrder-OMP",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate k-Wave reference fields.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case", action="append", dest="cases", default=None)
    args = parser.parse_args()

    selected = CASES
    if args.cases:
        wanted = set(args.cases)
        selected = tuple(case for case in CASES if case.name in wanted)
        missing = wanted - {case.name for case in selected}
        if missing:
            parser.error(f"unknown case(s): {', '.join(sorted(missing))}")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest: dict[str, object] = {"cases": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["reference"] = kwave_provenance()
    manifest["medium"] = {
        "sound_speed_m_s": SOUND_SPEED_M_S,
        "density_kg_m3": DENSITY_KG_M3,
        "absorption": "per case; see alpha_coeff_db and alpha_power",
        "alpha_coeff_units": "dB/(MHz**alpha_power cm)",
    }
    manifest["seed"] = {
        "profile": "isotropic_gaussian",
        "sigma_cells": SIGMA_CELLS,
        "peak_pa": P0_PEAK_PA,
        "smooth_p0": False,
    }

    cases = manifest["cases"]
    assert isinstance(cases, dict)

    for case in selected:
        print(f"[kwave-reference] running {case.name} {case.shape}", flush=True)
        result = run_case(case)
        archive = args.out / f"{case.name}.npz"
        # `np.savez` records an array's memory order in the NPY header, and a
        # transposed array is Fortran-contiguous. Forcing C order keeps the
        # stored buffer row-major over the case's shape, which is the layout the
        # Rust reader states and checks.
        arrays = {
            "p0": np.ascontiguousarray(result["p0"], dtype=np.float64),
            "p_final": np.ascontiguousarray(result["p_final"], dtype=np.float64),
        }
        if result["p_signal"] is not None:
            arrays["p_signal"] = np.ascontiguousarray(
                result["p_signal"], dtype=np.float64
            )
        np.savez_compressed(archive, **arrays)
        cases[case.name] = {
            "archive": archive.name,
            "sha256": sha256_of(archive),
            "bytes": archive.stat().st_size,
            "shape": list(case.shape),
            "dx_m": DX_M,
            "cfl": CFL,
            "dt_s": result["dt_s"],
            "steps": result["steps"],
            "t_end_s": float(result["dt_s"]) * int(result["steps"]),
            "compare_radius_cells": case.compare_radius_cells,
            "alpha_coeff_db": case.alpha_coeff_db,
            "alpha_power": case.alpha_power,
            "layer_sound_speed_m_s": case.layer_sound_speed_m_s,
            "layer_density_kg_m3": case.layer_density_kg_m3,
            "layer_interface_cell": case.layer_interface_cell,
            "layer_transition_cells": (
                LAYER_TRANSITION_CELLS if case.layer_interface_cell is not None else None
            ),
            "source_cell": list(case.source_cell) if case.source_cell else None,
            "source_mode": "additive" if case.source_cell else None,
        }
        print(
            f"[kwave-reference] {case.name}: steps={result['steps']} "
            f"dt={result['dt_s']:.6e} s bytes={archive.stat().st_size}",
            flush=True,
        )

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[kwave-reference] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

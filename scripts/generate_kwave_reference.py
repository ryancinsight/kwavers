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
)


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
    medium = kWaveMedium(sound_speed=SOUND_SPEED_M_S, density=DENSITY_KG_M3)
    kgrid.makeTime(medium.sound_speed, cfl=CFL, t_end=case.t_end_s)

    p0 = gaussian_seed(case.shape)
    source = kSource()
    source.p0 = p0

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

    p_final = np.asarray(output["p_final"], dtype=np.float64).reshape(case.shape)
    # `kgrid.Nt` is k-Wave's count of time *points*, spanning
    # `t_array = (0 : Nt - 1) * dt`, so the returned final field has been
    # advanced `Nt - 1` propagation intervals from the initial condition. The
    # manifest records that interval count, because that is what a solver
    # driven step-by-step has to execute to reach the same instant. Recording
    # `Nt` instead over-propagates by one step, which for these cases costs
    # five orders of magnitude of agreement.
    return {
        "p0": p0,
        "p_final": p_final,
        "dt_s": float(np.asarray(kgrid.dt).item()),
        "steps": int(np.asarray(kgrid.Nt).item()) - 1,
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
        "absorption": "lossless",
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
        np.savez_compressed(
            archive,
            p0=np.asarray(result["p0"], dtype=np.float64),
            p_final=np.asarray(result["p_final"], dtype=np.float64),
        )
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

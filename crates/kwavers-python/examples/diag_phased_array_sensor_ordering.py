#!/usr/bin/env python3
"""Diagnostic: verify that the row-order of pykwavers SensorRecorder matches
the row-order expected by k-wave-python's NotATransducer.combine_sensor_data
for the phased-array geometry used in us_bmode_phased_array_compare.py.

Runs in seconds — no simulation. Builds the same active_elements_mask,
flattens it two ways:
  (A) pykwavers SensorRecorder iteration order (Fortran order on the 3D
      mask, which for fixed-x transducer reduces to y-fast, z-slow).
  (B) combine_sensor_data expectation order (indexed_active_elements_mask[0].T
      flattened in numpy C-order, which is z-outer, y-inner on the
      post-transpose (Nz, Ny) shape).

If (A) and (B) produce the same element-index sequence, the existing script
is correct. If they differ, the per-element grouping in combine_sensor_data
is aggregating wrong voxels and amplitude/harmonic parity can drift.

Writes a short report to stdout and dumps a CSV-ish summary to
``output/phased_array_sensor_ordering_report.txt``.
"""

from __future__ import annotations

import numpy as np

from example_parity_utils import DEFAULT_OUTPUT_DIR, bootstrap_example_paths
bootstrap_example_paths()

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
from kwave.utils.dotdictionary import dotdict
from kwave.utils.matlab import matlab_find
from kwave.utils.signals import tone_burst


PML_SIZE = Vector([15, 10, 10])
GRID_SIZE_POINTS = Vector([256, 256, 128]) - 2 * PML_SIZE
GRID_SIZE_METERS = 50e-3
GRID_SPACING_METERS = GRID_SIZE_METERS / Vector(
    [GRID_SIZE_POINTS.x, GRID_SIZE_POINTS.x, GRID_SIZE_POINTS.x]
)
C0 = 1540.0
TONE_BURST_FREQ = 1e6
TONE_BURST_CYCLES = 4


def main() -> None:
    kgrid = kWaveGrid(GRID_SIZE_POINTS, GRID_SPACING_METERS)
    t_end = (GRID_SIZE_POINTS.x * GRID_SPACING_METERS.x) * 2.2 / C0
    kgrid.makeTime(C0, t_end=t_end)
    input_signal = tone_burst(1 / kgrid.dt, TONE_BURST_FREQ, TONE_BURST_CYCLES)
    input_signal = (1e6 / (C0 * 1000.0)) * input_signal

    tr = dotdict()
    tr.number_elements = 64
    tr.element_width = 1
    tr.element_length = 40
    tr.element_spacing = 0
    tr.radius = float("inf")
    tw = tr.number_elements * tr.element_width + (tr.number_elements - 1) * tr.element_spacing
    tr.position = np.round(
        [1, GRID_SIZE_POINTS.y / 2 - tw / 2, GRID_SIZE_POINTS.z / 2 - tr.element_length / 2]
    )
    transducer = kWaveTransducerSimple(kgrid, **tr)

    nt = dotdict()
    nt.sound_speed = C0
    nt.focus_distance = 30e-3
    nt.elevation_focus_distance = 30e-3
    nt.steering_angle = 0.0
    nt.steering_angle_max = 32
    nt.transmit_apodization = "Rectangular"
    nt.receive_apodization = "Rectangular"
    nt.active_elements = np.ones((transducer.number_elements, 1))
    nt.input_signal = input_signal
    ntx = NotATransducer(transducer, kgrid, **nt)

    active_mask = np.asarray(ntx.active_elements_mask)
    indexed = np.asarray(ntx.indexed_active_elements_mask)

    # Basic sanity
    print(f"[diag] active_mask shape = {active_mask.shape}")
    print(f"[diag] indexed shape     = {indexed.shape}")
    print(f"[diag] n_active_voxels   = {int((active_mask > 0).sum())}")

    # ── Order (A): pykwavers SensorRecorder emission order ────────────────
    # np.argwhere on the raw 3D mask returns coordinates in C-order, i.e.
    # for each x outer, each y middle, each z inner. SensorRecorder's
    # Fortran-order iteration yields x-fastest: for fixed x this is the
    # INVERSE, i.e. y-inner, z-outer.
    #
    # Per feedback_sensor_ordering.md, for the fixed-x transducer plane
    # pykwavers emits rows in y-fast, z-slow order. Let's verify with the
    # explicit Fortran-order flatten of the 3D mask.
    a_flat_fortran = active_mask.flatten(order="F")
    # Row k ↔ voxel (ix, iy, iz) where ix = k // (Ny*Nz)? Actually Fortran
    # flatten visits first-axis fastest, which is x-fastest for shape
    # (Nx, Ny, Nz). For fixed x this degenerates to (y-fast, z-slow).
    # Collect active voxel coordinates in Fortran order:
    coords_fortran = np.column_stack(np.nonzero(active_mask.ravel(order="F"))[0:1])
    # That gives linear index; convert back to (x,y,z):
    lin_f = np.nonzero(active_mask.ravel(order="F"))[0]
    xyz_f = np.column_stack(np.unravel_index(lin_f, active_mask.shape, order="F"))
    # xyz_f[:,0] should all equal x_src (transducer plane)
    assert (xyz_f[:, 0] == xyz_f[0, 0]).all(), "multi-plane transducer?"

    # Element index for each row in order A (= indexed value):
    elem_A = indexed[xyz_f[:, 0], xyz_f[:, 1], xyz_f[:, 2]]

    # ── Order (B): combine_sensor_data expectation order ──────────────────
    # combine_sensor_data uses `indexed_active_elements_mask[0].T` flattened
    # via boolean indexing (numpy C-order). Replicate exactly:
    slice0 = indexed[0]          # (Ny, Nz)
    transposed = slice0.T        # (Nz, Ny)
    elem_B = transposed[transposed > 0]  # 1D in C-order on the transposed shape

    # Reconstruct (y, z) for each row in order B:
    mask_t = transposed > 0
    # C-order on (Nz, Ny): z outer, y inner
    zs_B, ys_B = np.nonzero(mask_t)
    # (zs_B[k], ys_B[k]) is the (z, y) of row k.

    # ── Compare ───────────────────────────────────────────────────────────
    n = elem_A.size
    match = int((elem_A == elem_B).sum())
    print(f"[diag] n_rows(A) = {elem_A.size}, n_rows(B) = {elem_B.size}")
    print(f"[diag] element-index matches between A and B: {match}/{n}")
    if match < n:
        # Print first few mismatches
        mm = np.where(elem_A != elem_B)[0]
        print(f"[diag] first 10 mismatches: rows={mm[:10].tolist()}")
        print(f"[diag]   elem_A[mm[:10]] = {elem_A[mm[:10]].tolist()}")
        print(f"[diag]   elem_B[mm[:10]] = {elem_B[mm[:10]].tolist()}")

    # Also check the raw (y,z) order of both:
    print(f"[diag] first 5 (y,z) in order A: "
          f"{[(int(xyz_f[k,1]), int(xyz_f[k,2])) for k in range(5)]}")
    print(f"[diag] first 5 (y,z) in order B: "
          f"{[(int(ys_B[k]), int(zs_B[k])) for k in range(5)]}")

    out = DEFAULT_OUTPUT_DIR / "phased_array_sensor_ordering_report.txt"
    with open(out, "w") as f:
        f.write(f"n_active_voxels={n}\n")
        f.write(f"matches={match}/{n}\n")
        f.write(f"first_5_A_yz={[(int(xyz_f[k,1]), int(xyz_f[k,2])) for k in range(5)]}\n")
        f.write(f"first_5_B_yz={[(int(ys_B[k]), int(zs_B[k])) for k in range(5)]}\n")
        f.write(f"elem_A[:16]={elem_A[:16].tolist()}\n")
        f.write(f"elem_B[:16]={elem_B[:16].tolist()}\n")
    print(f"[diag] wrote {out}")


if __name__ == "__main__":
    main()

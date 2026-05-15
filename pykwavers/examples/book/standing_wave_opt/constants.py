"""Physical and grid constants for Chapter 31 standing-wave suppression.

Geometry (default, overridable via ch31 environment variables):

  Source                Reflective layer
  column (x=11)        (x=90..96)
      |                     |
  ===========================|===  x
  12-element linear array    |
      |  <- focus (x=68,y=32)|
  ===========================|===

Domain: 128 × 64 cells at 0.75 mm → 96 mm × 48 mm.
Frequency: 250 kHz → λ = 6.16 mm → 8.2 cells/wavelength.
Reflection coefficient at layer: R ≈ 0.322 (Z_layer/Z_tissue = 1.95).
"""

from __future__ import annotations

C_REF_M_S: float = 1540.0       # background sound speed
RHO_REF_KG_M3: float = 1000.0   # background density
C_LAYER_M_S: float = 2000.0     # reflective layer sound speed (bone-like)
RHO_LAYER_KG_M3: float = 1500.0 # reflective layer density

F0_HZ: float = 250_000.0        # centre frequency
DX_M: float = 7.5e-4            # spatial step (0.75 mm)
CFL: float = 0.25               # Courant number

NX: int = 128                   # grid cells, x (propagation)
NY: int = 64                    # grid cells, y (lateral)
PML_CELLS: int = 10             # absorbing boundary width

LAYER_X_START: int = 90         # reflective layer start (cell index)
LAYER_X_END: int = 96           # reflective layer end (cell index)

SOURCE_X: int = 11              # source column (PML_CELLS + 1)
FOCUS_X: int = 68               # target focal cell, x
FOCUS_Y: int = 32               # target focal cell, y = NY // 2
FOCAL_RADIUS_CELLS: int = 3     # neighbourhood for peak pressure sampling

N_ELEMENTS: int = 12            # transducer elements
ELEMENT_Y_MIN: int = 12         # first element y-cell
ELEMENT_Y_MAX: int = 52         # last element y-cell

BURST_CYCLES: float = 5.0       # source burst length in cycles
ACCUM_SKIP_CYCLES: float = 2.0  # steady-state lock-in skip (cycles after burst)

SWI_AXIS_HALF_WIDTH: int = 2    # lateral half-width averaged for focal-axis profile
SWI_SMOOTH_SIGMA: float = 0.5   # Gaussian smoothing (cells) before spectral SWI

SWI_WEIGHT: float = 0.70        # objective weight for SWI term
FOCAL_WEIGHT: float = 0.30      # objective weight for focal pressure term

GRAD_DELTA_RAD: float = 0.05    # central-difference step for phase gradient
ARMIJO_C1: float = 0.01         # Armijo sufficient-decrease constant
LINE_SEARCH_ALPHA0: float = 1.0 # initial line-search step
LINE_SEARCH_BETA: float = 0.5   # line-search contraction factor
LINE_SEARCH_MAX: int = 12       # maximum line-search halvings
N_OPT_ITER: int = 25            # optimization iterations

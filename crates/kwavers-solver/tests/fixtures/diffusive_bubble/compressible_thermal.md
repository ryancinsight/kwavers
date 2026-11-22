# Compressible and thermal pulse fixture

`compressible_thermal.csv` projects the signed gas transfer and wall
state columns from the standalone benchmark's `model-high.csv` for its
eight model histories. The reference uses BDF4 at 256 steps per carrier
cycle, 1536 liquid intervals and, when enabled, 256 gas heat intervals.
Both solvers start with a uniform liquid at curved-wall saturation and
retain their histories through the two quiet intervals.

The standalone reference uses independent nodal solute differences
in physical distance from the moving wall and backward-difference
time stepping. The production comparison uses conservative material
finite volumes and implicit midpoint. The C gas temperature closure
uses an independent BDF time derivative and gas inventory predictor.

The SHA-256 of the complete source CSV is
`671f1f48fd64903e4920e6b09b01c5b56e9fc201a8daffc11274f2ca1f5934bf`.
Build the companion standalone C reference using
`cc -O3 -std=c11 -Wall -Wextra -Werror reference.c reference_energy.c -lm -o reference`
with the following source digests.

| Source | SHA-256 |
| --- | --- |
| `reference.c` | `90fc1a8f0e34708fe2896f462799794ba9baf7bcf0adf86699801dd6dd8a4923` |
| `reference_energy.c` | `f1c43ee068b9aa0c9de1e3715155b635bc09952d54156fdbef13aef595c6a86e` |
| `reference_energy.h` | `d946cda5179e7e17a3171dcb25c908a2359250d4a8058fd19010eede27741fbd` |

Regenerate using
`python3 model_sensitivity.py ./reference high model-high.csv`.
Project in the original row order to `frequency_hz,model,phase,elapsed_s,phase_ppm,total_ppm,r_end_m,u_end_m_s`.

The fixture is synthetic solver verification. Its modeled gas and
thermal properties do not characterize a measured acoustic exposure.

<a id="kw-phys-004"></a>

## KW-PHYS-004 — Opt-in Rayleigh–Plesset gas feedback [minor] — done

PR #1135 retains a liquid state for a prescribed spherical wall. The coupled
solver advances one clean spherical bubble with the Henry concentration
derived from its accepted noncondensable gas inventory. The same closure
sets the mechanical pressure and the diagnostic equilibrium radius.

The local implementation passes the conservation, rollback, independent C
and BDF references, and nine-case isothermal refinement matrix. The
[design](../crates/kwavers-solver/src/forward/bubble_dynamics/gas_transfer/DESIGN.md)
defines the verified operating range. The
[checklist](../crates/kwavers-solver/src/forward/bubble_dynamics/gas_transfer/checklist.md)
tracks the local acceptance and release boundary.

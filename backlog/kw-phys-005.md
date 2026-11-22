## KW-PHYS-005 — Compressible and thermal diffusive bubbles [minor]

Extend the existing isolated-bubble API with independent Keller–Miksis
and radial gas heat transfer choices. Preserve the default Rayleigh–Plesset
and fixed-polytrope behavior. Commit the wall, heat and dissolved-gas fields
only after one coupled step succeeds.

Accept the eight standalone 220 and 230 kHz model histories for two
10 ms bursts at 64% duty. Compare signed gas transfer and wall motion at
every burst and quiet boundary, check total gas conservation, and refine
time and both grids for the combined model. The
[design](../crates/kwavers-solver/src/forward/bubble_dynamics/gas_transfer/DESIGN.md)
and [checklist](../crates/kwavers-solver/src/forward/bubble_dynamics/gas_transfer/checklist.md)
record the equations and review gates.

PR #1142 integrates the candidate. Physics publishes before Solver, and
the fresh registry consumer verifies both 5.2.0 packages. The release
checklist records the archive checksums and acceptance results. Experimental
validation, large-amplitude collapse and compressible liquid mass
transport remain open.

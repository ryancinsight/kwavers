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

Next increment after review is integration on the selected base. The
release requires separate publication authorization. Experimental validation,
large-amplitude collapse and compressible liquid mass transport remain open.

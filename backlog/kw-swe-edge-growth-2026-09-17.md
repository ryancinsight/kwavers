<a id="kw-swe-edge-growth-2026-09-17"></a>

## KW-SWE-EDGE-GROWTH-2026-09-17 — Elastic displacement grows without bound when the initial field reaches the edges [patch] [fix] — todo

priority: correctness; needs: none; scope: `crates/kwavers-solver/src/forward/elastic/`

- **Finding.** 64 cubed, lambda = mu = 1 GPa, water density, default PML, `ux = sin`, `uy = cos` of `0.37 i + 0.53 j + 0.71 k` over the whole grid: the peak grows 1 to 9.2e3 in 400 steps at CFL 0.5, and the growth follows physical time, not step count (CFL 0.25 at step 200 equals CFL 0.5 at step 100), so it is not a timestep instability. A centred Gaussian pulse at the same settings decays into the PML at every CFL.
- **Question.** Whether this is the boundary closure (one-sided first-order derivatives at the walls feeding a non-symmetric operator), the PML acting on displacement, or an inadmissible initial condition the solver should reject. Unbounded growth is not physical in any of the three.
- **Acceptance:** a regression test that runs the whole-grid initial condition and bounds its energy, or a typed rejection of such initial data with its reason; the cause recorded here.
- **Next step:** vary the boundary closure and PML independently on the whole-grid probe to isolate the cause.

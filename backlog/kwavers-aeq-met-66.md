<a id="kwavers-aeq-met-66"></a>

## KWAVERS-AEQ-MET-66 — Type thermal-diffusion quantities [major] [arch] — blocked 2026-08-05

priority: architecture; needs: none; scope: public thermal-diffusion parameters and integration-time contracts, their solver/orchestrator callers, Python and simulation serialization boundaries, ADR 103

- **Outcome:** perfusion uses `ReciprocalTime`, blood density `MassDensity`, blood heat capacity `SpecificHeatCapacity`, arterial temperature `ThermodynamicTemperature`, relaxation and integration steps `Time`; scalar extraction only at storage and numerical formula boundaries. The Plugin trait's raw host timestep stays an explicit execution-boundary conversion to `Time`. Thermal dose storage and thresholds stay CEM43 domain values, not SI time.
- **Acceptance:** every direct constructor and update caller compiles against the typed contract without compatibility wrappers; analytical thermal and CEM43 value regressions pass; strict package gates, Nextest, doctests, Rustdoc, formatting and raw-unit scans pass.
- **Evidence:** overlay Nextest 2,404/2,404 (five configured skips); strict Clippy for physics, solver, simulation and Python; doctests pass. `RUSTSEC-2026-0235` is resolved by the Ritk Eunomia 0.8 cutover and rkyv 0.8.17 (no advisory ignore).
- **Blocker:** the clean package Nextest build fails on the Windows GNU linker missing `libLIBCMT.a` and `libOLDNAMES.a` for unrelated top-level test binaries; the hosted exact-head matrix was pending.
- **Re-open trigger:** a hosted exact-head matrix result or a linker environment that builds the top-level tests.

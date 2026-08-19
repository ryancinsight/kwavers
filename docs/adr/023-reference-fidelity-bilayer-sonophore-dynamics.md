# ADR 023 — Reference-fidelity transient bilayer-sonophore dynamics

**Status:** Proposed — in-session differential audit done (items 1, evidence below) and the
multi-amplitude validation harness committed (`bls_dynamic_deflection_strength_curve_vs_plaksin_fig1`);
the quantitative <5 % close is **blocked on external references** (digitised Plaksin Fig. 1 +
a PySONIC differential run) and remains the open [major] work.
**Change class:** [major] (changes a validated physical observable — the transient
leaflet deflection magnitude — and the public `BilayerSonophoreDynamic` behaviour).
**Date:** 2026-06-09
**Builds on:** the existing transient leaflet ODE
(`acoustics::therapy::neuromodulation::bls::dynamics::BilayerSonophoreDynamic`,
the PySONIC/Plaksin 2014 `derivatives` reproduction) and its quasi-static and
analytic siblings in `bls/`.

## Context

The neuromodulation module's transient bilayer-sonophore source integrates the
full leaflet Rayleigh–Plesset ODE (Plaksin et al. 2014, *Phys. Rev. X* 4, 011004,
Eq. 2; PySONIC `BilayerSonophore.derivatives`): state `[U, Z, n_g]` driven by the
total pressure `P_tot = P_M + P_g − P₀ − P_ac + P_E + P_V + P_elec`. It is
correct in structure, fast (O(1) `P_M(Z)` table + adaptive RK4), non-hanging
(steric-wall handling), and exercised end-to-end through NICE/SONIC.

It is, however, **quantitatively off the published reference** at the canonical
operating point (500 kPa / 0.5 MHz, cortical RS rest −71.9 mV). Pinned by the new
value-semantic test
`tests::sources::bls_dynamic_deflection_inertially_amplified_near_plaksin_fig1`:

| quantity                              | this model | Plaksin Fig. 1 |
|---------------------------------------|-----------:|---------------:|
| steady-cycle peak deflection          | 10.45 nm   | ≈ 12 nm        |
| quasi-static peak deflection (same P) | 9.71 nm    | —              |
| inertial amplification (peak/QS)      | ≈ 1.08×    | ≈ 2×           |

Two deviations, the second more diagnostic than the first:

1. **~13 % low on peak deflection.** Within engineering tolerance for a reduced
   model, but a real gap against the reference.
2. **Near-quasi-static behaviour.** The transient peak (10.45 nm) sits only ~8 %
   above the quasi-static balance (9.71 nm), whereas Plaksin's transient is ~2×
   the quasi-static value. The model **fails to reproduce the inertial/resonant
   amplification** that is the entire reason a transient ODE exists over the
   quasi-static solver. This points at a dynamics defect (over-damping or an
   effective-mass/inertia error), not merely a calibration constant.

**Evidence already gathered (this ADR is grounded, not speculative):**

- *Integration resolution is NOT the cause.* Refining `dt_max` from `T/200` to
  `T/1000` left the peak **bit-identical at 10.45 nm** — the adaptive step-doubling
  control had already converged the trajectory.
- *Settling is NOT a free knob.* Raising `N_SETTLE` 30 → 200 **collapsed** the peak
  to −0.21 nm: integrated in isolation, the slow gas-rectification drift
  (`dn_g/dt`) runs away; in the full NICE/SONIC loop it is bounded by the membrane
  coupling. 30 carrier cycles is the correct, deliberately-tuned settle window.
- *The deflection is steady from cycle 1 (NEW, decisive).* Instrumenting the settle
  loop shows the per-cycle peak deflection is **10.447 nm at the very first carrier
  cycle and rock-stable through cycle 30** — there is **no resonant build-up
  transient**. The leaflet is **overdamped / driven below its resonance**, so the
  transient sits only ~8 % above the quasi-static balance rather than the ~2× Plaksin
  reports. This rules out "under-settling" and localises the gap to the
  **damping/inertia balance** (it is intrinsic to the model as parameterised, not a
  numerical or settling artefact). Whether 10.45 nm is *correct below-resonance
  physics* or an *over-damped discrepancy vs PySONIC* cannot be decided without the
  external reference data — hence the validation plan below.
- *Pressure terms verified.* `P_M` (area-averaged quadrature), `P_E = −T_E/R` with
  `T_E = k_A(Z/a)²`, `P_elec = −(a²/(a²+Z²))·Q_m²/(2ε₀ε_R)`, `P_g`, and
  `P_V = −12·U·δ₀·μ_S/R² − 4·U·μ_L/|R|` match the cited Krasovitski/PySONIC forms and
  constants on inspection; no term is structurally wrong, so the gap (if a bug) is a
  coefficient/discretisation mismatch the differential audit must localise.

So the residual gap lives in the **per-step physics**, not the numerics: the
prime suspects are (a) the viscous pressure `P_V = −12·U·δ₀·μ_S/R² − 4·U·μ_L/|R|`
(over-damping suppresses resonance), (b) the inertial term
`dU/dt = P_tot/(ρ_L·|R|) − 3U²/(2R)` (effective mass / curvature factor), and
(c) the tabulated `P_M(Z)` interpolation vs the exact quadrature near the operating
deflection. PySONIC reaches ≈12 nm with the same equation set, so the deviation is
a fidelity gap against that reference implementation, recoverable by exact
matching — **not** a fundamental model-topology change (the BLS is a single
symmetric-leaflet deflection `Z`; there is no missing "second leaflet" degree of
freedom).

## Decision

Open a [major] reference-fidelity calibration of `BilayerSonophoreDynamic` whose
**acceptance criterion is digitised-Plaksin-Fig.1 agreement at multiple amplitudes
and on an independent observable**, so success cannot be a single-point fit.

Work items, in dependence order:

1. **Differential audit against PySONIC `bls.py`.** Term-by-term compare the
   Rust `derivatives`/pressures against the upstream reference at a fixed `(U, Z,
   n_g, P_ac)`: each of `P_M, P_g, P_E, P_V, P_elec`, the inertial `dU/dt`, and the
   gas flux must match to round-off. Any divergence is the bug. This is the
   highest-value, lowest-risk step and may resolve the gap outright.
2. **Exact `P_M(Z)` in the operating band.** If the lookup interpolation biases the
   peak, raise the table resolution and/or fall back to exact quadrature across the
   smooth band (not only the steric wall). Re-profile to keep construction fast.
3. **Inertia/damping verification.** Confirm `ρ_L·|R|` (effective mass per area)
   and the viscous coefficients match Krasovitski 2011 / PySONIC exactly; restore
   the resonant amplification (`peak/QS → ~2×`) if an error is found.
4. **Seed / settle / record protocol parity.** Match PySONIC's
   `computeInitialDeflection`, settle-to-steady detection, and steady-cycle
   sampling; verify the peak is read from the integrated trajectory, not only the
   interpolated phase grid.

The ADR explicitly forbids reaching the target by **tuning a damping or stiffness
coefficient to hit 12 nm** — that is fitting-to-target (integrity: empirical
hacks). The constants are fixed by the cited literature; only genuine
implementation/discretisation discrepancies may be corrected.

## Alternatives considered

- **A. Accept the reduced model (status quo).** Keep 10.45 nm, document the ~13 %
  gap (already done in the module doc + the new test). Zero risk, zero cost.
  Rejected as the *default* because the near-quasi-static behaviour means the
  transient source currently adds little over the cheaper quasi-static one — the
  capability is not delivering its intended value. Retained as the fallback if the
  audit shows the gap is irreducible without nightly/expensive machinery.
- **B. Per-step exact `P_M` quadrature (drop the table).** Simplest accuracy lever,
  but the table was introduced precisely to avoid a per-derivative integral
  (perf/hang history, see `[[project_neuromod_nice_hh_jun8]]`). Only adopt if the
  audit proves the table is the dominant error; otherwise keep the table and widen
  its band.
- **C. Literal line-by-line PySONIC port.** Maximum fidelity, but imports PySONIC's
  full solver structure (event detection, dense output) — large, and against the
  "reproduce the physics, not the framework" stance of the existing module.
  Rejected; item 1's differential audit captures the fidelity benefit without the
  framework.

## Validation plan (acceptance criteria — multi-point, no single-point fit)

1. **Digitised reference curve.** Extract `Z_peak(P_ac)` from Plaksin Fig. 1 at
   ≥ 3 amplitudes (e.g. 100, 300, 500 kPa, 0.5 MHz, RS rest) into a committed CSV
   with provenance, and assert `|model − reference| / reference < 5 %` at **every**
   point (current single-point is 13 % at 500 kPa).
2. **Independent observable.** Assert the leaflet **tension** (or membrane
   capacitance excursion `C_m,min/C_m0`) at 500 kPa also matches Fig. 1 (≈ 15 mN/m)
   to < 10 % — a second, physically-distinct quantity so the fit is not degenerate.
3. **Resonant amplification restored.** `peak/quasi-static ≥ 1.5×` at 500 kPa
   (currently 1.08×), confirming the inertial dynamics are active.
4. **No regressions.** All 41 neuromodulation tests, the NICE/SONIC differential
   test, and the post-stimulus-AP tests still pass; `kwavers-physics` clippy clean
   (default + `--all-features`); construction time stays sub-second (the perf
   budget from the hang-fix work).
5. **Update the prose.** The module-doc "Approximations → Deflection accuracy"
   paragraph and the test band are tightened to the achieved fidelity, citing the
   committed reference CSV (close the doc/evidence loop).

## Consequences

- **Behavioural change.** `BilayerSonophoreDynamic` (and its PyO3 surface
  `nice_dynamic_response`) returns a larger, reference-matched deflection; the
  pinned test band and the module doc move from "≈10.5 nm (~13 % low)" to
  "within 5 % of Plaksin Fig. 1". This is the [major] observable change that
  necessitates this ADR.
- **Downstream.** NICE/SONIC firing thresholds shift slightly (stronger drive at a
  given pressure); the threshold-finder (`analysis::ThresholdQuery`) results move
  accordingly. The differential SONIC↔NICE test is unaffected (both use the same
  source).
- **Risk.** The audit may reveal the gap is intrinsic to the reduced single-`Z`
  representation, in which case Alternative A stands and this ADR is re-statused
  *Rejected — gap documented as irreducible*; the multi-point validation harness
  (items 1–2) is still committed as the honest evidence backing whatever value the
  model reports.
- **Scope guard.** This ADR does **not** add new neuron classes, the second NICE
  pathway, or temperature-dependent gating (those remain out of scope per the
  module doc); it only raises the fidelity of the existing transient source.

## References

- Krasovitski, B. et al. (2011). *PNAS* 108(8), 3258–3263 (BLS model).
- Plaksin, M., Shoham, S. & Kimmel, E. (2014). *Phys. Rev. X* 4, 011004 (Eq. 2; Fig. 1).
- Lemaire, T. et al. (2019). *J. Neural Eng.* 16, 046007 (PySONIC reference implementation).

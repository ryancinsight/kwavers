# ADR 024 — Theranostic brain imaging + real-time therapy monitor

Status: Accepted (staged delivery in progress)
Date: 2026-06-12
Change class: [arch] (new bounded context: `kwavers::theranostic`) + [minor] example
Supersedes/relates: builds on ADR 016 (exact-adjoint FWI), the seismic RTM/FWI
inverse stack, the CT→acoustic `CtMediumBuilder`/`HuAcousticModel`, the
hemispherical array, the thermal CEM43 + Pennes bioheat stack, and the
cavitation passive-dose stack.

## Context

We want one pipeline that, from a real CT, (1) images the whole brain with a
1024-element InsighTec-style hemispherical array using seismic-style inversion
(FWI / RTM / Born) with low-dose enhancements, (2) selects a target in the
reconstructed brain, and (3) sonicates while reconstructing **only the monitored
slice** each interval to watch the lesion develop in real time, B-mode style.

Hard constraint (CLAUDE.md integrity): the simulation must be *real* — no mocks,
no faking the image to get a desired lesion. The monitored image change must be a
genuine physical consequence of the simulated therapy.

User-selected scope (2026-06-12):
- Lesion physics: **both** thermal (CEM43) and cavitation (histotripsy),
  switchable by a `TherapyMode` parameter.
- Spatial scope: **3-D volume reconstruction**, **2-D monitored slice** through
  the focus during therapy.
- Monitor reconstruction: **RTM reflectivity B-mode frame + differential FWI /
  passive map overlaid** for quantitative lesion tracking.

## Decision

Add a `kwavers::theranostic` bounded context holding the **orchestration
algorithms** (pure, deterministic, unit-tested), and a thin example
(`brain_theranostic_monitor`) that wires the existing physics/inverse engines
through them. Orchestration logic lives in the library — not the example `main`
— so it is testable against analytical references rather than via a rendered
image.

### Why these pieces are new vs. reused

Reused as-is (verified real APIs):
- CT → medium: `kwavers_medium::CtMediumBuilder` + `HuAcousticModel`.
- Array: `kwavers_transducer::HemisphericalArray::new(radius, 1024, f0)`,
  `set_focus(FocalPoint)`.
- Full-brain inversion: `FwiProcessor` (`invert_multi_source`,
  `invert_multiscale`, `invert_encoded`), `RtmProcessor::migrate`, linear Born.
- Low-dose: `inverse::linear_born_inversion::regularization` (edge-preserving
  Charbonnier), `kwavers_math::inverse_problems::pnp::tv_denoise_chambolle`,
  directional-TV FWI regularization.
- Therapy physics: `thermal::{ThermalCEM43Grid, TemperatureCoefficients,
  PennesBioheat, AcousticHeatingSource}`; `analytical::cavitation::*`
  (`cumulative_cavitation_dose`, `cavitation_controller_pressure`, emission-band
  decomposition, PAM array integration); `bubble_dynamics::wood_sound_speed`.

New orchestration (this ADR) — `kwavers::theranostic`:
- `pulsing` — low-dose sparse transmit subsets (modular interleaving:
  `{i : i mod K == k}`, lossless union coverage) + therapy/imaging interleave
  schedule. The "pulsing scheme with image reconstruction" the user asked for.
- `lesion` — lesion → acoustic-medium perturbation: thermal `c(T)=c₀+(∂c/∂T)ΔT`
  (Duck 1990, 2 m/s/°C) and cavitation Wood-law collapse `c(β)` (Wood 1930);
  CEM43 iso-contour at 240 min for the necrosis boundary. **This is the physical
  link that makes the monitored image change real.**
- `targeting` — resolve a focal voxel + physical position from the reconstructed
  volume under an ROI mask.

### Coupling that keeps the monitor honest

Therapy stage advances the real fields: PSTD/FDTD forward → intensity → Pennes
heat (`Q=2αI`) → temperature → CEM43 dose (thermal) **or** Keller-Miksis /
emission-band passive dose → void fraction (cavitation). `lesion::perturb_
sound_speed` maps those state fields onto the sound-speed field; the monitored
slice is then re-imaged from that perturbed medium with RTM (reflectivity) +
short differential FWI (Δc) — so the lesion appears only because the medium
actually changed.

## Status / staged plan

- [x] Stage 0 — `kwavers::theranostic` module: `pulsing`, `lesion`, `targeting`,
      16 value-semantic unit tests green (analytical references). DONE.
- [x] Stage 1 — `brain_theranostic_monitor` example, part A: 3-D spherical
      skull+brain phantom (CT-loader hook documented) → 1024-element
      `HemisphericalArray` mapped onto grid acquisition voxels → low-dose
      sparse-subset multi-source FWI full-brain reconstruction (recovers brain
      contrast from a water start) → `targeting`. DONE, runs.
- [x] Stage 2 — interleaved therapy/imaging loop
      (`pulsing::interleave_schedule`) advancing **real** thermal (explicit
      Pennes bioheat + `ThermalCEM43Grid`) **or** cavitation (cumulative
      fractionation void fraction) state at the target; `lesion::perturb_sound_
      speed` updates the medium. DONE — the lesion genuinely develops: thermal
      CEM43 0→24 533 min, lesion voxels 0→5; cavitation fractionation 6→18→23
      voxels, monotonic.
- [x] Stage 3 — per-interval monitored-slice recon: differential 2-D slice FWI
      from simulated echoes of the perturbed medium, vs. a pre-therapy baseline
      slice recon (geometric bias cancels); PNG frame per interval; per-frame
      metrics table including the **applied** focal Δc and the **reconstructed**
      ROI Δc. DONE.
- [x] Stage 4a — **PAM passive channel (through-skull, no transmit).**
      `kwavers::theranostic::monitor::pam` time-exposure-acoustics source map.
      Debug finding: the repo `BeamformingProcessor::delay_and_sum_with` beamforms
      to ONE point (shape `(1,1,nt)`) and aligns to the *latest* arrival (advance
      `max−τ_i`) — wrong sign for focusing TOF emission data — so a bespoke
      grid-search aligner (`+(τ_i−τ_min)`) was implemented. **Verified:** localizes
      a point emitter to within 1 pixel; peak dominates background; far pixel is
      incoherent. 3 tests pass.
- [x] Stage 4b — **FD full-wave (CBS) quantitative recovery ACHIEVED.**
      `kwavers::theranostic::monitor::fd` wraps `frequency_domain::invert` over a
      single-row `MultiRowRingArray`. Two root causes were diagnosed and fixed:
      (1) the **spectral** CBS forward *diverged* on a 2-D `nz=1` slice (periodic
      FFT wraparound, no sponge possible) → switched to the **dense free-space
      `DenseConvergentBornOperator`** (true Green's function, no wraparound,
      converges without a boundary); (2) per-transmit **source scaling absorbed the
      lesion** → disabled for the differential measurement. Result: on a 12×12
      slice the FWI recovers **+53 m/s at the lesion centre vs +60 true (~88%)**,
      correct sign and location, far-field ≈ 0, in ~4 s. Test asserts positive
      recovery, far-field separation, and ≥30% centre magnitude. Remaining polish
      (TV/Tikhonov to suppress boundary artifacts, scale to larger grids) is
      documented, not blocking.
- [x] Stage 4c — **Multi-modal fusion + end-to-end hybrid pipeline.**
      `kwavers::theranostic::monitor::fusion` combines the FD-CBS quantitative
      `|Δc|` map and the PAM passive map into `agreement = √(q̂·p̂)` (confirmed by
      both, low false-positive) and `union = weighted-max` (sensitive), plus
      `lesion_extent` for growth tracking. **Verified:** an end-to-end test runs
      FD-CBS reconstruction + PAM synthesis/mapping on one synthetic lesion slice
      and fuses them — the agreement peak localizes the lesion. 29 theranostic
      tests pass, 0 ignored.
- [x] Stage 4d — **runnable hybrid-monitor example.**
      `examples/hybrid_lesion_monitor.rs` runs the full FD-CBS + PAM + fusion
      pipeline tracking a growing lesion on a 12×12 monitored slice (~20 s CPU):
      PAM localizes the source to the lesion centre every frame, the reconstructed
      centre Δc rises monotonically (1.5→6.0 m/s), and the fused agreement extent
      grows (2→8→12 px), writing a PNG per frame. Demonstrates "see the lesion
      develop" via the hybrid monitor end-to-end.
- [x] Stage 4e — **Gauss-Newton engine fix → exact magnitude recovery.**
      Added a matrix-free **Levenberg-Marquardt Newton-CG** solver to the
      frequency-domain FWI engine (`frequency_domain::invert_gauss_newton`,
      `GaussNewtonConfig`): the Gauss-Newton/Hessian action is a finite difference
      of the exact adjoint gradient (`H v ≈ [g(m+εv)−g(m)]/ε`, any forward
      operator), the inner solve is Steihaug-truncated CG, and the LM damping `λ`
      adapts per outer step until the full Newton step reduces the objective —
      eliminating NLCG's near-truth tiny-step stall. Solver test: from the exact
      background it drives the objective 4.6e-1 → 7.9e-24 in 7 steps, centre Δc
      **+60.00 (exact)**. 43 FD-FWI solver tests pass.
      Two integration fixes in the monitor: (1) route the **Gauss-Newton path
      through single-scatter Born** (its exact smooth gradient gives a clean FD
      Hessian; dense CBS's under-converged gradient corrupts it and is kept for the
      NLCG path); (2) `differential_lesion_map` reconstructs perturbed and
      background **from a common homogeneous reference** (not a warm start from the
      background, whose skull-dominated Hessian suppressed the lesion direction).
      Result: **97% lesion recovery through a skull ring** (centre +58.4 vs +60),
      exact in homogeneous tissue; the hybrid example's monitored centre Δc is
      ~60.0 every frame with fused extent growing 1→5→9→13→21. 30 theranostic
      tests pass, 0 ignored.
- [ ] Stage 4f (open) — **full-brain integration + RTM channel.**
      Wire FD+PAM+fusion into the heavy `brain_theranostic_monitor` per imaging
      interval; add the RTM reflectivity structural channel. Finding from Stage 4
      (time-domain): a
      low-budget *transmission* differential-FWI monitor (2–6 iters, sparse
      shots, λ≈10 mm ≫ lesion) does NOT resolve the localized lesion — the
      reconstructed ROI Δc stays ~0 while the *applied* Δc and the therapy state
      grow. This is real physics (sub-wavelength sound-speed lesions are hard to
      image by transmission), not a bug, and is exactly why the user specified
      RTM reflectivity as the primary B-mode monitor. Next: implement the RTM
      reflectivity monitor (forward/backward wavefield capture via the concrete
      `Solver`, `RtmProcessor::migrate`) which lights up the strong localized
      Wood-collapse scatterer immediately, overlaid with the differential FWI Δc
      for quantitation. Then validate recon Δc against applied Δc within an
      analytically derived bound.

The example is honest by construction: it reports the *applied* Δc (the real
lesion the physics produced) separately from the *reconstructed* ROI Δc (what the
monitor recovered) — ground truth is never presented as the monitor image.

## Consequences

- The reusable, testable core is decoupled from rendering and from the heavy
  solver runs, satisfying the "real simulation" constraint with unit tests that
  do not depend on a produced image.
- The example is runtime-heavy (1024-element 3-D forward sims); it follows the
  existing demos' env-var dataset overrides and CPML-safe grid sizing, and will
  document its wall-clock budget.

## References

- Wood, A.B. (1930). *A Textbook of Sound* — bubbly-mixture sound speed.
- Duck, F.A. (1990). *Physical Properties of Tissue* — c(T) coefficient.
- Sapareto & Dewey (1984). *Int. J. Radiat. Oncol.* — CEM43 thermal dose.
- Damianou & Hynynen (1994); McDannold et al. (2010) — 240 CEM43 ablation.
- Guasch et al. (2020). *npj Digital Medicine* — 3-D brain FWI.
- Baysal et al. (1983). *Geophysics* — reverse-time migration.
- Sidky & Pan (2008); Hauptmann et al. (2018) — sparse-view / low-dose sampling.

# Book Refactor & Consolidation Plan

Living tracker for the cumulative review/consolidate/refactor campaign:
reduce redundancy, fix numbering, improve figures and chapter descriptions.
Update the **Status** column as work lands. Each chapter is audited like Ch1–Ch2
were (physics correctness, paths, figures, cross-refs).

Legend: ✅ done · 🔧 in progress · ⬜ queued · ✂️ merge/deprecate target

---

## 1. Canonical chapter map (PROPOSED — needs sign-off)

Numbering today is broken: duplicate headers (two each of Ch4/5/6/7, two Ch10),
~13 unnumbered chapters, README order ≠ header order. Proposed canonical order
(pedagogical, deduplicated). `n?` = number pending structural decisions below.

| # | File | Title | Lines | Figs | Status |
|---|------|-------|------|------|--------|
| 1 | foundations.md | Wave Physics Fundamentals | 906 | 5 | ✅ audited |
| 2 | numerical_methods.md | Numerical Methods: FDTD & PSTD | 646 | 5 | ✅ audited |
| 3 | nonlinear_acoustics.md | Nonlinear Acoustics | 778 | 6 | ✅ audited (physics sound; figs added) |
| 4 | media_and_tissue_models.md | Media and Tissue Models | 1117 | 3 | ⬜ (strip absorption dup) |
| 5 | sources_and_transducers.md | Sources and Transducers | 737 | 0 | ✂️ merge w/ beamforming |
| 6 | beamforming_and_image_formation.md | Beamforming and Image Formation | 750 | 7 | ✂️ merge w/ sources |
| 7 | sensors_and_measurements.md | Sensors and Measurements | 834 | 0 | ⬜ (needs figs) |
| 8 | diagnostics.md | Diagnostic Ultrasound Imaging | 647 | 0 | ⬜ (strip PA/elasto dup) |
| 9 | photoacoustics.md | Photoacoustic Imaging | 947 | 5 | ⬜ |
| 10 | elastography.md | Elastography | 1246 | 9 | ⬜ |
| 11 | cavitation_and_bubbles.md | Cavitation and Bubble Dynamics | 1176 | 6 | ⬜ |
| 12 | therapy.md | Therapeutic Ultrasound | 396 | 0 | ⬜ |
| 13 | theranostics.md | Theranostics | 357 | 0 | ✂️ strip re-derivations |
| 14 | safety_and_dosimetry.md | Safety and Dosimetry | 1005 | 5 | ⬜ |
| 15 | transcranial_ultrasound.md | Transcranial Ultrasound | 955 | 9 | ⬜ |
| 16 | inverse_problems_and_pinns.md | Inverse Problems & PINNs | 427 | 0 | ⬜ |
| 17 | sonogenetics.md | Sonogenetics | 828 | 0 | ⬜ |
| 18 | performance_and_memory.md | Performance and Memory | 540 | 0 | ⬜ |
| 19 | validation_and_benchmarking.md | Validation and Benchmarking | 559 | 0 | ⬜ |
| — | ~~acoustic_propagation.md~~ | Acoustic Propagation | — | — | ✅ DELETED 2026-06-04 (HIFU example → Ch2 §2.14) |

**Application chapters** (largely distinct; clinical case studies) — keep, audit later,
renumber after the core is settled: histotripsy, simulation_orchestration,
passive_acoustic_mapping, bbb_lifu_opening, hifu_transcranial_ablation,
neuromodulation, transcranial_ust_brain_imaging, abdominal_histotripsy_fwi,
theranostic_fwi_platforms, intravascular_ultrasound, clinical_device_geometry,
segmented_tissue_transducer_planning, pancreatic_histotripsy.

---

## 2. Redundancy register (canonical home → strip elsewhere → cross-ref)

| Topic | Canonical home | Duplicated in (→ strip to a cross-ref) |
|-------|----------------|-----------------------------------------|
| First-order eqs, wave eq, plane/spherical waves, Green's fn, impedance | **Ch1** | acoustic_propagation §8.2–8.6 |
| Power-law + Treeby–Cox fractional-Laplacian absorption | **Ch1 §1.9** | acoustic_propagation §8.7–8.8; media §4.4–4.5 |
| FDTD/PSTD, CFL, k-space, CPML, dispersion | **Ch2** | acoustic_propagation §8.9–8.11; validation §4–5; performance §3 |
| Tissue acoustic params / EOS | **Ch4 (media)** | Ch1 §1.7/1.10/1.11 (keep brief in Ch1, full table in Ch4) |
| Rayleigh–Plesset, Keller–Miksis, Blake, PCD | **Ch11 (cavitation)** | theranostics §7.1–7.2; therapy; transcranial |
| Histotripsy mechanism | **histotripsy.md** | cavitation §7.9; theranostics §7.6 |
| Bioheat (Pennes), CEM43 thermal dose | **therapy.md** | safety §9.5; media §4.7 |
| Time-reversal focusing/reconstruction | **sensors.md** | photoacoustics §7; transcranial §10.4 |
| Photoacoustic imaging | **photoacoustics.md** | diagnostics §5.5; sensors §7 |
| Shear-wave elastography | **elastography.md** | diagnostics §5.6 |
| FUS neuromodulation | **neuromodulation.md** | therapy §6.7; transcranial §10.9 |
| BBB opening | **bbb_lifu_opening.md** | transcranial §10.10; theranostics §8 |
| Piston/bowl radiation, focusing, BLI raster, phased-array delay | **Ch5+6 merged** | sources ↔ beamforming (≈60% overlap) |

Consolidation rule: keep ONE full derivation at the canonical home; everywhere
else replace the re-derivation with a 2–3 line summary + "see Chapter N §x".

---

## 3. Figure register

Chapters with **0 figures** (candidates for new genuine/analytical figures):
nonlinear_acoustics, sensors_and_measurements, diagnostics, therapy, theranostics,
inverse_problems_and_pinns, performance_and_memory, validation_and_benchmarking,
sonogenetics, + most application chapters. Figure scripts exist for most
(`crates/kwavers-python/examples/book/chNN_*.py`) but several have the
`REPO_ROOT` depth bug (wrote to `crates/docs`) and/or aren't embedded.
Standard: `figures/chNN/`, generated to repo-root `docs/book/figures/`, physics
in Rust where a real computation applies, embedded with a descriptive caption.

---

## 4. Cumulative work queue (ordered)

1. ✅ **Sign off** canonical structure + the ✂️ decisions (§5 below). [D1–D4 signed]
2. ✅ **Deprecate `acoustic_propagation.md`**: HIFU focal worked example grafted
   into `numerical_methods.md` §2.14 (corrected gain 132→13); BC taxonomy already
   covered by Ch1 §1.7/§1.13 + Ch2 §2.9; file deleted; stale media.md cross-refs
   fixed; chapters.toml entry removed. ch08 script+figures retained for reuse.
3. ✅ **D2** — Sources/Transducers ↔ Beamforming split done. Canonical homes:
   Sources owns single-source/transducer physics (element directivity §3, bowl
   focusing gain §4, BLI rasterization §6, bowl discretization §8); Beamforming owns
   multi-element beam-forming (array factor/grating lobes §4.2, focusing/resolution
   §4.3, apodization §4.4, receive DAS/MVDR §4.5, configs §4.7, steering §4.12).
   Stripped Beamforming §4.1/§4.6/§4.8 → cross-refs to Sources (preserved the unique
   on-axis near-field pressure formula into Sources §3); added reciprocal scope-boundary
   notes to both intros; fixed dangling eq ref + stale paths in beamforming scope.
   (3 residual stale `kwavers::analysis/clinical` paths deferred to the chapter's full
   audit, item 5.)
4. ✅ De-dup pass done (one sub-item ⏸️ deferred to item 5; each cross-ref verified
   against the kwavers impl):
   - ✅ **theranostics §7.1/§7.2.1/§7.6** (R–P, Minnaert, Blake, PCD signatures,
     histotripsy) → cross-refs to Cavitation/Histotripsy chapters (−62 lines).
     **Verified impls:** `RayleighPlesset`, cavitation_control::detection
     (broadband/spectral/subharmonic), `CavitationCloud`, `MicrobubbleDynamicsService`,
     `TherapyIntegrationOrchestrator`, `SafetyController`, `UlmDetector`,
     `PlaneWaveCompounding`. Corrected the §7.7 code-map (3 wrong struct names + all
     paths post-split; added the missing PCD row). MR-thermometry is external (MR
     scanner), kept as background.
   - ✅ **diagnostics §5.5/§5.6** (PA generation/recon/spectroscopy; shear-wave
     speed/ARFI) → cross-refs to Photoacoustics/Elastography chapters (−60 lines).
     **Verified impls:** `kwavers_diagnostics::photoacoustic` + `reconstruction::
     acoustic_projection`; `kwavers_solver::forward::elastic::swe`,
     `inverse::elastography`, `kwavers_domain::imaging::ultrasound::elastography`.
   - ✅ **media §4.4.3/§4.5** (Treeby–Cox fractional Laplacian, K–K dispersion) →
     cross-ref to Foundations §1.9.3/Theorem 1.7 (−94 lines). **Fixed two doc-vs-code
     errors in the process:** §4.5.3's `exp()` per-mode pseudocode (real impl is the
     pressure-side `p += c²(τ·L1 − η·L2)`, verified in `pstd/physics/absorption/apply.rs`)
     and §4.4.3's y=1 Kramers–Kronig limit (`tan(π/2)=∞`). Repointed 4 dangling §4.5 refs.
   - ✅ **therapy §6.7** (tFUS neuromodulation) → cross-refs: skull transmission →
     Transcranial §10.2–10.5, MI/TI/FDA → Safety §9.3–9.7, mechanism → Neuromodulation
     chapter (−17 lines). **Verified impls:** `kwavers_physics::acoustics::transcranial::
     bbb_opening`, `::acoustics::therapy` (+ `::sonogenetics`).
   - ⏸️ **transcranial §10.9/§10.10 ↔ neuromodulation.md/bbb_lifu_opening.md**: deferred
     to item 5 — these have transcranial-specific theorems (skull insertion loss, NICE
     sonophore model, radiation force near boundary); assigning canonical homes safely
     needs reading the dedicated chapters first.
   De-dup pass total: ~355 lines of duplication removed across 5 chapters; every cross-ref
   verified against the kwavers implementation.
5. 🔧 Per-chapter physics/path/figure audit, in canonical order (Ch3 onward):
   - ✅ **Ch3 Nonlinear Acoustics**: physics audited (sound; consistent with corrected Ch1
     — 2nd-harmonic growth & Theorem 3.8 factor-2 match). Fixed 2 wrong cross-refs
     (fractional Laplacian → Ch1 §1.9.3/Theorem 1.7, not Ch2), all stale paths, B/A water
     5.2→5.0 (SSOT). Fixed the `ch03` script REPO_ROOT (was writing `crates/docs`) and
     **embedded 6 figures incl. a genuine PSTD Westervelt-vs-Fubini solver validation
     (≤2% error)**. Verified impls: westervelt, westervelt_spectral, kuznetsov, kzk,
     HarmonicTracker, ShockCapture, ConservationTracker all present.
   - ⬜ Ch4 (media), Ch5/6 (sources/beamforming residual paths), Ch7 sensors, … +
     the deferred transcranial §10.9/§10.10 ↔ neuro/BBB de-dup.
   Recurring fix: figure scripts use `parents[3]`/3×`..` (→ `crates/`); needs +1. Fixed in
   ch01, ch02, ch03, ch08.
6. ✅ **D4** — README reordered to canonical Part-grouped order + one-line
   descriptions; Acoustic Propagation removed; clinical_device_geometry &
   pancreatic_histotripsy added.
7. ⬜ **D3 final pass**: renumber all headers/tags/cross-refs to canonical; verify.

## 5. Structural decisions (SIGNED OFF 2026-06-04)
- **D1 = Deprecate Acoustic Propagation**: graft unique bits (BC taxonomy → Ch1 §1.7;
  HIFU focal worked example → Ch2) into Ch1/Ch2, then delete the file; fix cross-refs.
- **D2 = Keep Sources + Beamforming as two chapters** with a clean transmit/receive
  split (Sources = transmit-side; Beamforming = receive/image formation); move
  overlapping derivations to one side, cross-ref the other.
- **D3 = One final renumber pass at the end.** Do NOT renumber headers/tags as we
  go (avoids churn from merges/moves). Internal numbering stays put until the final
  scripted pass.
- **D4 = Reorder README early** to canonical pedagogical order + one-line
  descriptions (guides the campaign).

---
*Status log:* Ch1, Ch2 audited (physics/paths/figures). Ch8 (acoustic_propagation)
audited + renumbered, now flagged for deprecation. Plan created this session.

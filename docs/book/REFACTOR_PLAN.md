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
| 4 | media_and_tissue_models.md | Media and Tissue Models | 1045 | 5 | ✅ audited (absorption de-duped; figs fixed) |
| 5 | sources_and_transducers.md | Sources and Transducers (hdr "Ch5"⚠collision) | ~770 | 5 | ✅ audited (kwavers_source/transducer split; piezo/CMUT=theory) |
| 6 | beamforming_and_image_formation.md | Beamforming and Image Formation (Ch4) | ~780 | 11 | ✅ audited (kwavers_transducer paths; figs 4.1–4.11) |
| 7 | sensors_and_measurements.md | Sensors and Measurements | 853 | 5 | ✅ audited (physics sound; figs embedded) |
| 8 | diagnostics.md | Diagnostic Ultrasound Imaging (internally "Ch5") | ~660 | 6 | ✅ audited (code-map verified; 6 ch05 figs embedded) |
| 9 | photoacoustics.md | Photoacoustic Imaging | 948 | 5 | ✅ audited (paths fixed; figs re-wired ch13; Γ eq fixed) |
| 10 | elastography.md | Elastography | ~1230 | 5 | ✅ audited (de-fictioned impl claims; figs→ch10; design/order) |
| 11 | cavitation_and_bubbles.md | Cavitation and Bubble Dynamics | 1172 | 4 | ✅ audited (physics sound; figs re-wired) |
| 12 | therapy.md | Therapeutic Ultrasound | 401 | 5 | ✅ audited (code-map 19/19 verified; figs ch06; §12.10 10× units fix) |
| 13 | theranostics.md | Theranostics | 358 | 0 | ✅ audited (already recap-cross-ref'd; 3 struct names + theorem/eq renumber + intensity fix; registration marked not-impl) |
| 14 | safety_and_dosimetry.md | Safety and Dosimetry | 1012 | 5 | ✅ audited (paths+structs fixed; figs re-wired ch15) |
| 15 | transcranial_ultrasound.md | Transcranial Ultrasound | ~930 | 5 | ✅ audited (flat paths→split; 28 eq tags + table 10.x→15.x; 5 fictional APIs→real; 9 broken ch_tc figs→5 real ch16; HU-map note) |
| 16 | inverse_problems_and_pinns.md | Inverse Problems & PINNs | ~470 | 5 | ✅ audited (§8/§9 de-fictioned; figs→ch17) |
| 17 | sonogenetics.md | Sonogenetics | ~860 | 5 | ✅ audited (flat paths→split; eq refs 11.x→17.x, 10.15→15.15, Ch10→15; PSTDSolver API + consts fixed; 5 ch18 figs embedded) |
| 18 | performance_and_memory.md | Performance and Memory | ~560 | 5 | ✅ audited (title 19→18; §N→18.N renumber; flat→split paths; Scalar/AbsorptionKernel/CpmlUpdater/BATCH_SIZE fictions corrected; 5 ch19 figs embedded) |
| 19 | validation_and_benchmarking.md | Validation and Benchmarking | ~580 | 5 | ✅ audited (title 20→19; §N→19.N; flat→split paths; RITK registration verified REAL; 5 ch20 figs embedded) |
| — | ~~acoustic_propagation.md~~ | Acoustic Propagation | — | — | ✅ DELETED 2026-06-04 (HIFU example → Ch2 §2.14) |

**Application chapters** (clinical case studies) — auditing in turn:
✅ **21 simulation_orchestration** (#30), ✅ **22 passive_acoustic_mapping** (#31),
✅ **14 histotripsy** (#34), ✅ **23 bbb_lifu_opening** (#32),
✅ **24 hifu_transcranial_ablation** (#33), ✅ **25 neuromodulation** (#35),
✅ **26 transcranial_ust_brain_imaging** (#36), ✅ **27 abdominal_histotripsy_fwi** (#37),
✅ **28 theranostic_fwi_platforms** (#38), ✅ **29 intravascular_ultrasound** (#39),
✅ **30 clinical_device_geometry** (#40), ✅ **31 segmented_tissue_transducer_planning** (#41),
✅ **32 pancreatic_histotripsy** (#42), ✅ **33 cmut_vs_pmut** (#43).
**🎉 Full Ch1–33 book sweep complete — every chapter audited.**
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
   - ✅ **Ch4 Media and Tissue Models**: physics audited (EOS/B-A/Voigt/Biot/skull sound).
     Fixed all stale paths; **Liver B/A 3-way inconsistency** (table 6.5 / text 6.5 / code
     7.0 → SSOT `B_OVER_A_LIVER = 6.75`) + Liver α₀ 0.40→0.50; fixed the **Voigt dispersion
     coefficient** ((ωτ)²/4 → 3(ωτ)²/8, binomial 3/8). Re-wired the 2 broken `ch_media`
     embeds → existing `ch12` figures, removed the orphan skull embed, fixed the `ch12`
     script REPO_ROOT, and embedded **5 figures** (was 0 working). Verified impls:
     `TissueProperties`/`LIVER`, `solver/forward/thermal/pennes.rs`, `medium/elastic.rs`,
     transcranial aberration (skull), anisotropic stiffness.
   - ✅ **Ch7 Sensors and Measurements**: physics audited (hydrophone directivity,
     spatial Nyquist, pressure–velocity, time-reversal — all sound; directivity correctly
     cross-refs Sources/Ch5). Embedded **5 figures** (ch14 script; was 0) at their sections,
     replaced the placeholder §11 "Figure References" table (which cited fictional scripts)
     with a correct generation note, fixed stale paths. Verified impls: `kwavers_domain::
     sensor::recorder`, `kwavers_solver::inverse::reconstruction` (back-projection/TR).
     NOTE: ch14 regen currently blocked by the in-flight pykwavers refactor; used the
     existing root figures.
   - ✅ **transcranial §10.9/§10.10 ↔ neuromodulation (Ch26)/BBB (Ch24)**: the
     implementation check showed these are **complementary, not duplicates** — so
     cross-referenced rather than stripped. **Caught a fictional impl claim:** §10.9.2
     said the NICE model is "implemented in `…transcranial::NICEModel`" — *no such struct
     exists anywhere*. Corrected to state NICE is a theoretical mechanism and pointed to
     the actually-implemented neuromodulation path (`…therapy::sonogenetics`: channels/
     membrane/neuron/arf_field) + Ch26. Added a verified BBB cross-ref/impl pointer
     (`…transcranial::bbb_opening`: models/safety/simulator) + Ch24.
   - ✅ **Ch11 Cavitation and Bubble Dynamics** (canonical home for the bubble physics
     cross-referenced from theranostics/transcranial): physics audited — Keller–Miksis
     (7.2), R–P, Blake, Minnaert all canonical-correct; doc path `…bubble_dynamics::
     keller_miksis::equation` is exact. **Verified impls:** Keller–Miksis (+ shape_instability),
     Marmottant/encapsulated shell, Bjerknes (`bjerknes_forces`), Epstein–Plesset
     (`dissolution`), sonoluminescence — all present. Fixed 30 stale paths; re-wired 4
     broken `ch_cav` embeds → existing `ch09` figures (RP, Blake, Minnaert, collapse);
     removed 2 broken embeds (Bjerknes, Marmottant) with no `ch09` figure.
   - ✅ **Ch9/14 Safety and Dosimetry** (canonical home for MI/TI/CEM43/Arrhenius/FDA
     cross-referenced from therapy/Ch3/transcranial): physics audited — MI = P_r.3/√f_c,
     TI, CEM43, Arrhenius Ω, FDA Track-3 limits all canonical-correct. **Verified impls:**
     CEM43 → `kwavers-core::constants::medical`; Arrhenius → `…constants::chemistry`;
     safety module → `kwavers_therapy::safety` (MechanicalIndexCalculator,
     ThermalIndexCalculator, InterlockSystem, DoseController, EnhancedComplianceValidator,
     SafetyAuditLogger, ClinicalSafetyMonitor, ClinicalSafetyLimits); 3-D dose grid →
     `kwavers_physics::thermal::thermal_dose::ThermalCEM43Grid`. Fixed §9.11 struct drift
     (SafetyMonitor→ClinicalSafetyMonitor, SafetyLimits→ClinicalSafetyLimits,
     ThermalDose→ThermalCEM43Grid); batch-fixed all stale `kwavers::clinical`/`kwavers::physics`
     paths (now 0 `kwavers::` left); re-wired 3 broken `ch_safe` embeds → `ch15` (MI/TI/CEM43),
     removed 2 with no `ch15` equivalent (derating, MR thermometry), added 2 unused `ch15`
     figs (Arrhenius §9.6, FDA §9.7). 5 `ch15` figs now resolve.
   - ✅ **Ch9-doc Photoacoustic Imaging**: physics audited (thermoelastic p₀=ΓH, PA wave
     eq, Green's/SRT, universal back-projection, time-reversal convergence, Beer–Lambert,
     diffusion μ_eff, two-λ unmixing, PAM resolution — all first-principles correct).
     **Fixed** the muddled Grüneisen alt-form in Theorem 2.1/Eq(2.1),(2.4): old
     `αβc²/(κ_T C_p)` with the proof's own `α=ρκ_T` gave an extra factor ρ; replaced with
     the correct, already-derived `β/(ρκ_T C_p)` (Eq 2.3). **Verified impls (paths updated
     monolith→split):** GrueneisenModel→`kwavers_physics::photoacoustics::thermoelasticity`
     (soft_tissue()); thermoelastic source / optical diffusion+MC / Planar+Line+TimeReversal
     reconstruction + PhotoacousticReconstructionModel::reconstruct →
     `kwavers_simulation::photoacoustics::vertical::{source,optical,reconstruction}`;
     HemoglobinDatabase→`kwavers_optics::chromophores`; SpectralUnmixer→
     `kwavers_analysis::signal_processing::spectroscopy`; signal primitives→`kwavers_signal`.
     PyO3 book bindings confirmed (hbo2/hb_molar_absorption, gruneisen_parameter_water,
     pa_sphere_pressure_signal, spectroscopic_unmixing_lstsq). All `kwavers::` flat paths
     gone (0 left); §11.1 module tree rewritten. Re-wired 5 broken `ch_pa` embeds → 5
     existing `ch13` figs (PA N-wave, Γ(T), HbO₂/Hb spectra, unmixing, bandwidth) placed at
     natural homes; removed 3 schematics with no ch13 fig (Green's geometry, back-projection,
     PAM modes). Fixed recurring `ch13` script REPO_ROOT depth bug (3→4 `..`).
   - ✅ **Ch10 Elastography**: physics audited (strain/stress, Helmholtz P/S decomposition,
     μ=ρc_S², quasi-incompressibility, RF cross-corr strain, ARFI/ToF/phase-gradient SWE,
     local-Helmholtz MRE, Voigt complex modulus, Murnaghan acousto-elasticity, CRLB,
     resolution — all derivations sound). **De-fictioned implementation claims** (high-value):
     the chapter asserted ~15 nonexistent types — `StrainElastographer`, `SweInverter`,
     `HelmholtzInverter`, `DispersionFitter`, `TissueClassifier`, `AcoustoElasticInverter`,
     `UncertaintyEstimator`, `VoigtKernel`, `ShearModulusMap`, `ShearWavelengthEstimator`,
     `LocalFrequencyEstimator`, `ElasticPropagator`, `phase_unwrap::unwrap_2d`,
     `crosscorr::rf_displacement`, `TissueMechanics`. **Verified real surface:**
     `kwavers_solver::inverse::elastography::linear_methods::ShearWaveInversion` (ToF /
     phase-gradient / `direct` Gauss-Seidel algebraic-Helmholtz / volumetric / directional);
     `kwavers_solver::forward::elastic::{swe::ElasticWaveSolver, nonlinear::NonlinearElasticWaveSolver
     (hyperelastic, NOT Murnaghan)}`; `kwavers_physics::acoustics::imaging::modalities::elastography::
     {displacement::DisplacementEstimator, harmonic_detection::HarmonicDetector,
     radiation_force::*(ARFI)}`; `kwavers_medium::{elastic,viscous}` (generic Stokes);
     `kwavers_physics::analytical::elastography` (shear_wave_speed, voigt_complex_modulus,
     voigt_shear_wave_dispersion — PyO3-exposed, generate the figures). Marked Murnaghan/
     acousto-elasticity, dedicated LFE/Goldstein unwrap, Voigt/Zener medium kernel, CRLB
     estimator, organ-staging classifier as **theory-only / not yet implemented**. Rewrote
     §10.13 module topology + data flow; corrected 10 inline impl notes; all flat `kwavers::`
     gone (0 left). **Design/order:** added "Chapter 10" to header + consistent dash; fixed
     prereq cross-refs (Ch9→Ch16 inverse, Ch4-SP→Ch7 sensors, Ch2-hetero→Ch4 media).
     **Figures:** 9 broken `ch_elasto` embeds → 5 computed `ch10` figs (shear-speed, c_P/c_S
     ratio, MRE displacement, Voigt G'/G'', dispersion) placed at natural homes, ascending
     captions 10.1–10.5; removed 4 schematics (modality overview, stress tensor, P/S motion,
     strain pipeline, resolution-depth) with no computed equivalent. Fixed ch10 script
     REPO_ROOT depth bug (3→4 `..`).
   - ✅ **Ch5 Diagnostic Ultrasound Imaging** (internally "Chapter 5": ch05 script + 5.x
     sections + README Part III — self-consistent; REFACTOR_PLAN row-8 is a table index, not
     the chapter number). Physics audited (B-mode signal model, TGC, Hilbert envelope, PW
     compounding √N_c SNR, Doppler, CEUS microbubble RP linearization, ULM σ_loc, gCNR
     invariance, speed-of-sound shift tomography — all sound). **Code-map verified & corrected**
     (renamed/relocated types): `PlaneWaveCompound` (`kwavers_diagnostics::workflows::
     plane_wave_compounding`); DAS `delay_and_sum()` (`kwavers_analysis::…::beamforming::
     time_domain::das`); envelope `instantaneous_envelope()` (`kwavers_signal::analytic`);
     Doppler `AutocorrelationEstimator`/`WallFilter`/`ColorFlowImaging`
     (`kwavers_analysis::…::doppler`); PA gen `kwavers_diagnostics::photoacoustic`, recon
     `AcousticProjectionGeometry`, spectroscopy `SpectralUnmixer`
     (`kwavers_analysis::…::spectroscopy`), `HemoglobinDatabase` (`kwavers_optics::chromophores`);
     ULM `UlmDetector`/`HungarianTracker`/`SuperResReconstructor`/`VelocityMapper`
     (`kwavers_analysis::…::ulm`); all SoS-shift types confirmed in
     `kwavers_diagnostics::reconstruction::sound_speed_shift::*`. **Marked NOT implemented:**
     f-k/Stolt migration (§5.2.2), Frangi vesselness, CEUS pulse-inversion/AM contrast
     sequences. All flat `kwavers::` gone (0 left). Cross-refs to "Chapter 4" (beamforming
     Thm 4.4 lateral res, Eq 4.15 Stolt) verified correct against
     beamforming_and_image_formation.md (internally Ch4). **Design:** scope sentence clarified
     (PA/elasto summarized + cross-ref; SoS-shift added); §5.10 worked-example dB notation
     fixed. **Figures:** chapter had ZERO embeds despite 6 computed ch05 figs on disk —
     embedded all 6 (PSF, PW compounding, Doppler spectrum, PA signal, Hb spectra, SWE) at
     natural homes, captions 5.1–5.6. Fixed ch05 script REPO_ROOT depth (3→4 `..`).
   - ✅ **Ch4 Beamforming and Image Formation** (internally Ch4; verified as the home of the
     "Chapter 4" cross-refs diagnostics relies on — Thm 4.4, Eq 4.15). Physics audited (array
     factor, grating-lobe theorem + λ/2 rule, focusing delay law, DAS SNR ∝ N, MVDR, BLI
     rasterization, electronic steering, eikonal aberration correction — all sound).
     **Key finding: transducer code lives in a dedicated `kwavers_transducer` crate**, not
     `kwavers_domain::source`. **Verified + corrected paths/types:** `PhasedArrayTransducer`,
     `BeamformingMode::{Focus,Steer,Custom,PlaneWave}` (NOT DAS/Fourier/Hadamard),
     `KwaveApodizationWindow`, `BowlTransducer` (was `FocusedBowl`), `ArcSource` (was
     `FocusedArc`), `KWaveArray`, `KwaveBli::map_surface_sample`,
     `get_focus_delays`/`get_element_delays` (was `compute_delays`), `LinearArray`/`MatrixArray`
     all under `kwavers_transducer`; single-element directivity is
     `kwavers_physics::analytical::transducer::circular_piston_directivity` (was
     `directivity_factor` in transducers); steering `SteeringController`/`SteeringMode`/
     `FocalPoint` (`kwavers_transducer::hemispherical::steering`); eikonal aberration delays
     `kwavers_therapy::therapy::theranostic_guidance::waveform::eikonal::eikonal_travel_time`;
     MVDR/Capon **confirmed real** at `kwavers_analysis::signal_processing::beamforming::
     adaptive::mvdr` (added pointer to §4.5.3, was theory-only). The §4.12.8 analytical funcs
     + unit tests all confirmed in `kwavers_physics::analytical::transducer`. f-k/Stolt (§4.5.2)
     left as theory (not implemented, consistent with Ch5). All flat `kwavers::` gone (0 left);
     Appendix 4A paths fixed. **Design/figures:** chapter embedded only its 5 steering figs +
     2 anims — embedded the 6 missing computed figs (fig01–06: directivity, array factor,
     2-D beam pattern, lateral resolution, apodization, BLI) at natural §4.1–4.6 homes and
     **renumbered all figures + in-text cross-refs to a clean sequential 4.1–4.11** (steering
     figs 4.12–4.16 → 4.7–4.11; 6 prose refs + 3 validation refs updated). Fixed REPO_ROOT
     depth (3→4 `..`) in BOTH ch04 scripts.
   - ✅ **Ch (Sources and Transducers)** (transmit-side, D2 kept separate). Physics audited
     (piezoelectric constitutive + Mason circuit, piston directivity, bowl focusing gain,
     phased-array delay law, BLI rasterization, source contract — all sound). **Key finding:
     source code is split across two crates** — `kwavers_source` (Source trait, config,
     structs, grid_source, injection, wavefront family) and `kwavers_transducer` (factory,
     basic/piston/arrays, kwave_array, flexible, hemispherical, array_2d). Rewrote the §10
     module tree to the split layout. **Corrected wrong/fictional details:** BLI constants
     `DISC_AXIS_EPSILON` 1e-10→1e-12 and `DISC_PACKING_NUMBER` 6.28→7.0 (verified against
     `kwave_array/math.rs`); Euler convention ZYZ→**XYZ** (`euler_xyz_rotation_matrix`,
     R=Rz·Ry·Rx — the chapter claimed ZYZ R_z·R_y·R_z); removed nonexistent
     `ArrayGridWeights`/`KWaveArrayError`/`WavefrontSource`/`get_array_grid_weights`
     (→ `get_array_binary_mask`, builder `&mut Self`, wavefront source family); `SourceMode`
     = Additive|AdditiveNoCorrection|Dirichlet; `SourceInjectionMode` = Boundary|Additive{scale};
     annular via `KWaveArray::add_annular_element` (not PistonSource/add_annular_array).
     **Marked NOT implemented:** piezoelectric/Mason material model and CMUT (theory only —
     kwavers injects a prescribed kinematic source, not an electromechanical solve); scope +
     §2 status note added. PyO3 figure bindings (`circular_piston_directivity`,
     `focused_bowl_onaxis`, `linear_array_factor` → `kwavers_physics::analytical::transducer`)
     confirmed. All flat `kwavers::` gone (0). **Design/figures:** chapter had ZERO embeds +
     a stale §12 "Figure References" table listing 8 fictional scripts (Fig 5.1–5.8) — embedded
     the 5 real ch11 figs (Figures 1–5: directivity, bowl on-axis, delay law, BLI accuracy,
     beam pattern) at §3–§9 homes and rewrote §12 to an accurate note. ch11 script REPO_ROOT
     depth already correct (4 `..`).
   - 📌 **D3 numbering collisions found** (for the final renumber pass): internal chapter
     numbers are inconsistent with README pedagogical order and collide — **"Chapter 4"** is
     claimed by BOTH Media (4.x) and Beamforming (4.x); **"Chapter 5"** by BOTH Sources and
     Diagnostics (5.x). Cross-refs currently resolve by the de-facto internal numbers
     (diagnostics→beamforming "Ch4" works), so D3 must renumber atomically across all chapters
     + cross-refs. Canonical target = README Part order: 1 Foundations · 2 Numerical · 3
     Nonlinear · 4 Media · 5 Cavitation · 6 Sources · 7 Beamforming · 8 Sensors · 9 Diagnostics
     · 10 Photoacoustics · 11 Elastography · 12 Therapy · 13 Theranostics · 14 Histotripsy ·
     15 Transcranial · 16 Safety · 17 Sonogenetics · 18 Inverse · 19 Performance · 20 Validation.
   - ✅ **Ch (Inverse Problems and PINNs)**. Physics audited (Tikhonov–Morozov, adjoint-state
     gradient, Born, PINN universal approximation + loss/AD, acoustic CT Radon/FBP,
     regularization — all derivations sound). **PINN is REAL and Burn-backed** (Burn 0.19,
     autodiff/wgpu, `pinn` feature) — big positive verification. **De-fictioned §8/§9** (many
     idealized type names didn't exist): real surface is `kwavers_solver::inverse::fwi::
     time_domain::FwiProcessor` (+ `adjoint_state` primitives, `frequency_continuation`,
     `encoded_source` Hadamard, `search` **Armijo** line search — NOT Wolfe/L-BFGS); misfit
     `MisfitType` enum {L2,L1,Envelope,Phase,Correlation,Wasserstein(OT)}; CBS frequency-domain
     FWI (PyO3 `invert_breast_fwi`); `linear_born_inversion` (pcg_invert); PINN
     `elastic_2d::{ElasticPINN2D<B>, LossComputer, PINNOptimizer<B> (SGD/Adam/AdamW)}` +
     `geometry::CollocationSampler` (Sobol); regularization in
     `kwavers_math::inverse_problems::regularization::ModelRegularizer3D` (Tikhonov/TV-Huber/
     smoothness/L1); elastography `ShearWaveInversion` + nonlinear_methods. **Marked NOT
     implemented / theory-only:** generic `AdjointState<S>`/`GradientComputer<M>` wrappers,
     Wolfe line search, L-BFGS quasi-Newton, `AdamLBFGS`, `RadonTransform`/FBP **acoustic CT**,
     `TikhonovSolver`-as-struct, `LocalFrequencyEstimation`, `ConstrainedInversion`
     projected-gradient, and the scalar-wave/1-D/3-D PINN (real PINN is 2-D elastic).
     Corrected §9 validation claims (L-BFGS→Armijo; 1-D-wave PINN→2-D elastic Adam/AdamW). All
     flat `kwavers::` gone. **Figures:** chapter had ZERO embeds — embedded 5 real ch17 figs
     (Figures 1–5: SVD spectrum, PINN loss, Tikhonov L-curve, CBS-vs-Born convergence, c-map
     reconstruction); ch17 script REPO_ROOT already correct (4 `..`).
   - ✅ **Ch6 Therapeutic Ultrasound** (the "Chapter 6" flagged as skipped/consolidated —
     completed). Physics audited (acoustic power deposition, Pennes bioheat, CEM43, ARFI,
     sonoporation/BBB, lithotripsy, tFUS neuromodulation — all sound). **Code-map fully
     verified** (all FOUND, renames applied): `HIFUPlanner`, `PennesBioheat`
     (kwavers_physics::thermal::diffusion) + `ThermalDiffusionSolver`
     (kwavers_solver::forward::thermal_diffusion), `ThermalDoseCalculator`/`ThermalCEM43Grid`,
     `IntensityTracker`, lithotripsy `ShockWaveGenerator`/`StoneFractureModel`/
     `CavitationCloudDynamics` (was CavitationCloud), `HistotripsyScenario`/`PulsePattern`,
     `MicrobubbleDynamicsService` (was MicrobubbleService), `SafetyController`,
     `TherapyIntegrationOrchestrator` (was TherapyOrchestrator), sonogenetics stack. PyO3
     bindings confirmed (ThermalSimulation, cem43_at_temperatures, hifu_focal_pressure_gain,
     absorption_power_law_db_cm). All flat `kwavers::clinical::therapy` → `kwavers_therapy`
     (0 left). **Figures:** chapter had ZERO embeds — embedded the 5 real ch06 figs
     (Figures 6.1–6.5: power deposition, HIFU focal gain, Pennes temp rise, CEM43 accumulation,
     ablation zone); ch06 REPO_ROOT already correct.
   - ✅ **IMPLEMENTED: Local Frequency Estimation (LFE)** (user directive — implement
     not-yet-implemented components). New `lfe.rs` in
     `kwavers_solver::inverse::elastography::linear_methods` + `InversionMethod::
     LocalFrequencyEstimation`; windowed energy-ratio `|k|²≈⟨|∇u|²⟩/⟨u²⟩` (Oliphant/Manduca
     2001). Value-semantic test recovers a known cs=1.0 m/s plane wave (±0.4) and checks
     μ=ρcs²; all 10 linear-methods tests green; workspace compiles (kwavers-imaging enum +
     kwavers-simulation checked). Elastography chapter updated (LFE no longer theory-only).
     Remaining not-implemented components tracked in **backlog.md** (Frangi, Goldstein unwrap,
     CRLB, ConstrainedInversion, f-k migration, Voigt kernel, L-BFGS, acoustic-CT Radon/FBP,
     CEUS sequences, Murnaghan, piezo/CMUT).
   - ✅ **Ch6 re-verified complete**: both "Chapter 6" files are done — Therapy (last cycle)
     and **Sensors and Measurements** (0 flat paths, 5 ch14 figs embedded, structure sound).
   - ✅ **IMPLEMENTED: CRLB estimation bounds** (user directive). New
     `kwavers_analysis::signal_processing::estimation_bounds`: `time_delay_crlb_variance`
     (Walker–Trahey), `time_delay_crlb_std`, `strain_crlb_std`, `shear_wave_speed_crlb_std`.
     5 value-semantic tests green (closed-form equality, monotonicity, degenerate→∞).
     Elastography §10.12/§10.13 updated (no longer theory-only).
   - ✅ **CORRECTED false-negative: Frangi vesselness ALREADY EXISTS** (verification mandate in
     reverse). `kwavers_analysis::signal_processing::vasculature::{compute_frangi_response,
     VesselSegmentation::segment}` — the audit's "NOT FOUND" was a name-only miss (searched
     `FrangiFilter`). Diagnostics §5.9 corrected (real vasculature row added; wrong
     not-implemented marker removed). Lesson recorded in backlog.md: re-verify by algorithm,
     not type name.
   - ⬜ Remaining chapters (Theranostics/Histotripsy, Transcranial figures (needs pykwavers),
     Sonogenetics, Performance, Validation, application chapters) + D3 atomic renumber
     (collisions at 4/5/6/7 logged) + backlog.md implementation queue (next: Goldstein 2-D
     unwrap, ConstrainedInversion).
   - 📌 Backlog (needs pykwavers): add Bjerknes-force + Marmottant-shell figures to the
     `ch09` script and re-embed in Cavitation §7.8/§7.10.
   - ⚠️ **transcranial figures all broken**: 9 embeds point to empty `figures/ch_tc/`;
     the `ch16` script makes 5 *different* figures (skull insertion loss, phase aberration,
     CT conversion, Strehl, skull temp) and has no NICE/BBB figure. Full re-wire deferred to
     Ch15's audit (needs pykwavers, currently down mid-refactor).
   - ⚠️ NOTE: a code refactor is in flight (CPML moved to a new `kwavers_boundary` crate,
     seen in `dispatch/fdtd.rs`). The final path-verification pass (item 7) must re-check
     all `kwavers_*::` module paths against the then-current crate layout.
   Recurring fix: figure scripts use `parents[3]`/3×`..` (→ `crates/`); needs +1. Fixed in
   ch01, ch02, ch03, ch08.
6. ✅ **D4** — README reordered to canonical Part-grouped order + one-line
   descriptions; Acoustic Propagation removed; clinical_device_geometry &
   pancreatic_histotripsy added.
7. ✅ **D3 renumber — CORE Chapters 1–20 DONE** (resolves the duplicate "Chapter 4/5/6/7"
   collisions; "Chapter 6 appears twice" fixed). Canonical README pedagogical order applied:
   1 Foundations · 2 Numerical · 3 Nonlinear · 4 Media · 5 Cavitation · 6 Sources ·
   7 Beamforming · 8 Sensors · 9 Diagnostics · 10 Photoacoustics · 11 Elastography ·
   12 Therapy · 13 Theranostics · 14 Histotripsy · 15 Transcranial · 16 Safety ·
   17 Sonogenetics · 18 Inverse · 19 Performance · 20 Validation.
   - **Headers**: all 20 unique + numbered.
   - **Prefixed chapters** (cavitation 7→5, beamforming 4→7, diagnostics 5→9, elastography
     10→11, therapy 6→12, theranostics 7→13, histotripsy 21→14, transcranial 10→15, safety
     9→16, sonogenetics 11→17): headers + own §/Theorem/Figure/Eq/§-tags renumbered with
     structurally-safe patterns (physical values like "(7.5 mm)" untouched; verified).
   - **Bare-section chapters** (sources→6, sensors→8, photoacoustics→10, inverse→18,
     performance→19, validation→20): header-numbered; their per-section eq numbering left
     intact (self-consistent).
   - **Cross-references** resolved per-occurrence (Chapter N, Theorem N.M, Eq N.M, §N.M,
     Figure N.M) incl. collision-twin damage (theranostics→cavitation §7→5; therapy focal-gain
     "Theorem 4.9"→Ch.6 §4; diagnostics→beamforming Thm 7.4/Eq 7.15). Trefethen-book citation
     correctly preserved (external ref, not a kwavers chapter).
   - **README**: TOC numbered 1–20 + note updated.
   - ✅ **Part VII (application/case-study chapters) DONE** — the whole book is now a unique,
     contiguous sequence **1–32**. Prefixed app chapters self-renumbered (sim_orch 22→21,
     PAM 23→22, bbb 24→23, hifu 25→24, neuromod 26→25, transcranial_ust 27→26); 6 unnumbered
     case studies header-numbered (27–32). Inline cross-refs remapped (simultaneous single-pass
     21→14, 22→21, 23→22, 24→23, 25→24, 26→25, 27→26). Fixed the app-scheme→core prerequisite
     mismaps (the case studies used Transcranial=16/Safety=15/Sonogenetics=18/Inverse=17/
     Sensors=14 → corrected to canonical 15/16/17/18/8), the comma-list
     transcranial_ust "Chapter 14,22,24,26", and the bbb "Theorem 22.1→21.1" cross-ref. README
     TOC numbered 1–32. Verified: all 32 headers unique/contiguous, residual §22–27 refs are
     legit self-refs.

43. ✅ **AUDITED + FIXED: Chapter 33 (CMUT vs PMUT).** The chapter I authored this campaign;
   content verified accurate — §33.10 code-map matches `kwavers_transducer::mems::{plate, cmut,
   pmut, comparison}` + `flexible::FlexibleTransducerArray`, cross-refs (Ch6/16/29) correct,
   verdict physics (gap-limited CMUT ceiling, drive-scaling PMUT, flex penalty) matches the
   implemented `evaluate_ivus`/`evaluate_therapy`. Fixed a stale §33.8 critique ("squeeze-film
   out of scope" — it's since implemented). **The 6 figures were broken links** (never generated;
   the Done entry had flagged a needed rebuild). Generating them surfaced **two real bugs:** (1)
   pykwavers was failing to import entirely — a stale-DLL mismatch (`hodgkin_huxley_response` added
   by the other sessions' neuromodulation work post-dated my Ch29 build); rebuilt to fix it,
   unblocking all book scripts. (2) the 15 mems `#[pyfunction]`s were registered in `register_book`
   but **omitted from the hand-maintained `__init__.py` re-export list**, so the public `kw.cmut_*`/
   `kw.ivus_figure_of_merit` API raised AttributeError — added all 15 to the import block + `__all__`.
   Then ran `ch33_cmut_vs_pmut.py` → all **6 `figures/ch33/*`** generated and resolve.

42. ✅ **AUDITED + RECONCILED: Chapter 32 (Pancreatic Cancer Histotripsy).** Two findings. **(1)
   54 MB base64 cleanup:** the chapter referenced `embedded_figures.md` (a base64-inlined PNG
   bundle); these existed for ch21d (6 MB), ch21e (**46 MB**), ch21f (2 MB), all git-tracked, only
   ch21f referenced by any chapter, and all duplicating the committed PNGs — deleted all three (~54
   MB freed) and removed the chapter reference. **(2) Chapter↔script drift** (same shape as Ch31):
   the chapter describes the analytic histotripsy planner (Rayleigh–Sommerfeld forward + intrinsic-
   threshold cavitation + Pennes + raster, 3 regimes) and its figures `fig01_pdac_histotripsy_overview`/
   `fig02_pdac_thermal`, but the current `ch21f` script was migrated to `run_theranostic_inverse_from_ritk`
   (FWI, writing `fig00`–`fig03`). **Key difference from Ch31:** the analytic model **still exists in
   Rust** (`kwavers_physics::analytical::cavitation::sonication`: `build_sonication_schedule`,
   `histotripsy_pulses_for_lesion_radius`, `histotripsy_kill_fraction`, …), so I kept the valid
   Mathematical Contract, embedded the 2 existing analytic figures (which depict it), and added an
   honest **Executable note** that the script now demonstrates the same-aperture FWI instead. No
   re-run needed. **Env note:** pykwavers currently fails to import (`hodgkin_huxley_response`
   missing) — a stale-DLL mismatch: other sessions added the neuromodulation HH binding + updated
   `__init__.py` after my Ch29-era `maturin develop`; a rebuild fixes it (the binding is real in
   `analytical_bindings::neuromodulation`). Book scripts can't run until pykwavers is rebuilt.

41. ✅ **AUDITED + RECONCILED: Chapter 31 (Segmented Tissue Transducer Planning).** **Significant
   drift found and fixed.** The chapter described a Python-side ray-trace optimiser (per-aperture
   hazard-path scoring, an angular crossfire plan at −90/−170/−10°, complex ridge least-squares,
   dense-field hotspot refinement, coverage 0.7838 metrics) with figures `segmentation_candidate_scores`/
   `optimized_spot_and_avoidance`/`hybrid_solver_tradeoffs`. Git showed that optimiser was
   **deliberately removed** (commit *"ch32: replace Python ray-trace physics with pykwavers FWI
   bindings"*) for PyO3-only compliance and replaced by `plan_abdominal_array_placement_from_ritk_ct`
   + `run_theranostic_inverse_from_ritk` — but the chapter and its (orphaned) figures were never
   updated, so it documented non-existent functionality. **Reconciled to the current
   implementation:** rebuilt pykwavers (from the Ch29 build) and ran the current script (exit 0,
   65-iter FWI, objective 0.95→0.14) to generate the **4 real figures** (`fig01_bowl_placement_3d`,
   `fig02_exposure_and_segmentation`, `fig03_multimodal_reconstruction`, `fig04_fwi_convergence`);
   rewrote §Planning Contract as the real two-phase Rust pipeline (256-element bowl placement +
   same-aperture finite-frequency inverse / RTM / elastic-shear FWI / cavitation channels / fusion)
   with the **real metrics from `metrics.json`** (fusion pearson 0.978 / dice 1.0 / NRMSE 0.015) and
   the honest model flags (`is_full_wave_inversion=false`, `iterative_elastic_fwi=true`); added a
   transparent History note; removed the 3 orphaned figures (PNG+PDF); reframed the label-semantics
   away from the removed optimiser's penalty/null language. Filed backlog #15 for an optional
   Rust-native re-implementation of the segmentation-driven crossfire/hazard-path optimiser.

40. ✅ **AUDITED: Chapter 30 (Clinical Theranostic Device Geometries).** Well-grounded — already
   embeds its 5 figures (`figures/ch31/*`), and the §282 Rust-core code-map is **fully accurate**:
   verified `plan_abdominal_array_placement` (`…::abdominal3d::placement`),
   `plan_transcranial_focused_bowl_placement` (`…::transcranial_focused_bowl3d`), both PyO3
   `…_from_ritk_ct` bindings, and all **3 property tests**
   (`placement_returns_skin_point_outside_body`, `bowl_vertex_matches_skin_contact`,
   `all_elements_on_sphere_of_correct_radius` in `abdominal3d/tests.rs`). Fixes: (1) flat code-map
   paths `kwavers::clinical::therapy::theranostic_guidance::{abdominal3d, transcranial_focused_bowl3d}`
   → split crate `kwavers_therapy::…`; §Validation `kwavers/src/…/abdominal3d.rs` →
   `crates/kwavers-therapy/…/abdominal3d/tests.rs`. (2) cross-refs: **4× "Chapter 29"** (the
   `run_theranostic_inverse_from_ritk` / same-aperture inversion + brain focused-bowl) → **Chapter
   28** (Ch29 is IVUS, not the same-device chapter); "Chapter 14 for transcranial skull aberration"
   → **Chapter 15** (Ch14 is Histotripsy; skull aberration is the Transcranial chapter). No missing
   capability.

39. ✅ **AUDITED: Chapter 29 (Intravascular Ultrasound Imaging and Therapy).** Concise, honest
   chapter (analytic vessel phantom, "not redistributed patient IVUS", "next increment is a real
   external-dataset loader"). No flat paths. Fixes: (1) §Figures listed 5 figure names without
   embedding — embedded the **5 real `figures/ch30/*`** (dataset_and_anatomy, transducer_design,
   ivus_bmode_simulation, microbubble_therapy_map, usage_sequence) as Figs 29.1–29.5; (2) added a
   cross-ref to **Chapter 33 §33.8/§33.9** (the kwavers CMUT/PMUT IVUS verdict — CMUT wins imaging,
   bulk-piezo/PMUT wins therapy), connecting the IVUS chapter to its transducer-physics home.
   **Delegation gap — RESOLVED.** The example script `ch30_intravascular_ultrasound.py` computed
   B-mode log-compression + therapy intensity/ΔT in pure Python (only MI used a Rust kernel).
   Refactored it to delegate to existing Rust PyO3 kernels: intensity →
   `kw.acoustic_intensity_from_amplitude`, ΔT → `kw.adiabatic_temperature_rise_kelvin`, B-mode
   log-compression → `kw.bmode_db_fixed_reference` (each with a bit-identical Python fallback for
   when the extension is absent). The spatially-varying-α heat source `Q = 2α(x)·I` is applied over
   the Rust intensity (the scalar-α `acoustic_heat_source_density` kernel can't take a field).
   **Verified end-to-end**: rebuilt pykwavers (`maturin develop --release`, exit 0), re-ran the
   script (5 figures regenerated), and confirmed **all 24 metrics bit-identical** (max rel diff
   0.0e0) between the pre-refactor Python run and the post-refactor Rust-kernel run. The B-mode
   *reflectivity* phantom forward model has no single Rust kernel and stays in the script as the
   visualization layer.

38. ✅ **AUDITED: Chapter 28 (Same-Device Therapeutic Ultrasound / Finite-Frequency Inverse / RTM).**
   Large theorem-heavy chapter; **exemplary honesty** — explicit model-fidelity flags
   (`is_full_wave_inversion=false` for the reduced-Born/Tikhonov channels, `iterative_elastic_fwi=true`
   for the elastic-shear channel, "not yet a joint c/α/ρ/β/bubble-density coupled KKT/Gauss-Newton
   solve"). Code verified: bindings `run_theranostic_inverse_from_ritk` /
   `run_theranostic_nonlinear_3d_from_ritk` exist; the cited modules
   `kwavers_therapy::therapy::theranostic_guidance`, `kwavers_solver::inverse::{same_aperture,
   seismic}`, and `TheranosticInverseConfig::passive_reconstruction` all exist. Fixes: (1) 3 flat
   `kwavers::clinical/solver::…` code-map paths → split crates; (2) **8 stale prose "Chapter 29"
   self-references** → Chapter 28 (was Ch29 under old numbering; the `KWAVERS_CH29_*` env-var
   constants left intact); (3) Figures: §Figures listed 6 outputs without embedding — embedded the
   **6 real `figures/ch29/*`** (device_placement, exposure_and_reconstruction, focused_bowl_3d,
   dynamic_range_diagnostics, nonlinear_3d_westervelt_rp, controlled_comparison) as Figs 28.1–28.6.
   No missing capability.

37. ✅ **AUDITED: Chapter 27 (Abdominal Histotripsy FWI).** Concise, **already honest** (explicit
   Scope Limits: 2-D, straight-ray Born, "not a full 3-D adjoint Westervelt/RP FWI"). Code verified:
   the script `ch28_abdominal_histotripsy_fwi.py` calls `pykwavers.run_theranostic_inverse_from_ritk`
   (binding exists in `theranostic_bindings::inverse::run`; abdominal modules in
   `kwavers_therapy::…::abdominal3d`) — all physics Rust-side. Fixes: (1) stale self-reference
   "**Chapter 28** implements…" → "This chapter…" (was Ch28 under old numbering; clarified the
   "Chapter 14 (Histotripsy)" CT-loader ref); (2) §Figures listed 4 figure names without embedding —
   embedded the **4 real `figures/ch28/*`** (kidney/liver abdominal FWI + kidney/liver
   subharmonic-nonlinear) as Figs 27.1–27.4. No flat paths, no missing capability.

36. ✅ **AUDITED: Chapter 26 (Transcranial UST Brain Imaging).** Well-grounded Born-FWI / RTM
   chapter; already **honest** about gaps (§26.6 table marks "Elastic/shear FWI: Not implemented;
   deferred" and "bubble-state FWI pending"). Code verified: `kwavers_diagnostics::reconstruction::
   transcranial_ust` (+`volume_born`) exists, and both PyO3 bindings
   (`run_transcranial_ust_volume_inversion_from_ritk_ct`, `run_theranostic_nonlinear_3d_from_ritk`)
   are real. Fixes: (1) prereq "Chapter 17 (Inverse Problems and PINNs)" → **Chapter 18** (Ch17 is
   Sonogenetics); (2) flat path `kwavers::clinical::imaging::reconstruction::transcranial_ust` →
   `kwavers_diagnostics::reconstruction::transcranial_ust`; (3) `pykwavers::`→`pykwavers.` call form;
   (4) "same-device aperture contract (**§29.2**)" → **§28.2** (Ch28 is the same-device theranostic
   chapter; Ch29 is IVUS); (5) Figures: §26.8 listed 10 figures as path lines but embedded none —
   embedded all **10 real `figures/ch27/*`** (ct_geometry, acoustic_model, brain_reconstruction,
   optimization_and_data, simulated_ultrasound, multislice_stack, centroid_roi, + histotripsy
   scenarios/metrics/passive-RTM) as Figs 26.1–26.10 at their sections. No missing capability.

35. ✅ **AUDITED: Chapter 25 (Low-Intensity Ultrasound Neuromodulation).** **Exemplary chapter** —
   the companion `ch26_neuromodulation.py` delegates *all* physics to the kwavers Rust core via
   PyO3 (verified all 9 bindings exist: `kw.{mechanical_index_field, acoustic_intensity_from_amplitude,
   compute_acoustic_membrane_tension_py, boltzmann_open_probability_py, coupled_channel_drive_py,
   gaussian_beam_pressure_field_py, compute_cem43, simulate_lif_neuron_py, acoustic_heat_source_density}`),
   honoring the pykwavers-is-PyO3-only principle (Python only orchestrates/plots). No missing
   capability. Fixes: (1) prereq had a **duplicate "Chapter 16"** — the second ("Transcranial
   Ultrasound") corrected to **Chapter 15**; (2) "CEM43 follows the **Chapter 15** convention" →
   **Chapter 16** (Safety §16.5); (3) §25.8 listed fig01–06 without embedding — embedded the **6 real
   `figures/ch26/*`** (acoustic_focus, mechanochemical_response, channel_activation, thermal_safety,
   clinical_guidance_space, guidance_map) as Figs 25.1–25.6 at their sections and rewrote §25.8 as
   an index that also documents the Rust-kernel delegation. 0 flat paths.

34. ✅ **AUDITED: Chapter 14 (Histotripsy — Classical vs Millisecond).** **Major cleanup win:** the
   file was **960 KB** because §14.11 embedded 4 figures as inline **base64 data-URIs** (Figs
   14.18–14.21, KiTS19 real-CT run). The PNGs already exist on disk in `figures/ch21d/`, so the
   inline copies were pure redundant bloat — replaced all 4 with file references → **39 KB (96 %
   smaller)** and fixed the "embedded directly as base64 … no external links" intro. Also embedded
   the **17 other figures** (14.1–14.17), which were standalone `` `figures/chXX/…{png,pdf}` ``
   path-spans with no rendered image → proper `![]()` embeds (ch21=comparison, ch21b=HCC,
   ch21c=diagnostics). **All 21 figures verified to resolve on disk.** Fixes: (1) module-ownership
   header + 2 source links used flat `kwavers::clinical/physics::…` paths → split crates
   (`kwavers_therapy::therapy::{clinical_scenarios, lithotripsy, domain_types}`,
   `kwavers_physics::{acoustics::bubble_dynamics, analytical::cavitation::histotripsy,
   thermal::diffusion}` — all verified present). (2) cross-ref error "**Chapter 12** (Media and
   Tissue Models)" → **Chapter 4** (Media is Ch4; Ch12 is Therapeutic Ultrasound, separately and
   correctly cited). 0 flat paths, 0 base64 remain. (Note: the PROPOSED chapter table row 14 is
   stale vs the final README numbering — table predates the last renumber.)

33. ✅ **AUDITED: Chapter 24 (Transcranial HIFU and BBB Treatment Planning).** A Python-pipeline
   planning chapter (no flat `kwavers::` Rust paths). Notably **well-grounded** — a search agent
   confirmed every cited capability exists: the Rust benchmark
   `kwavers_therapy::…::transcranial_fus::benchmark::run_skull_adaptive_transcranial_benchmark`, the
   PyO3 binding `run_transcranial_skull_adaptive_benchmark_from_ritk_ct`, the `ritk` image/registration
   bindings, `transcranial_planning/scene.py::CANONICAL_BRAIN_SCENE`, and both
   `test_transcranial_planning.py` / `test_book_therapy_chapters.py`. Fixes: (1) stale cross-ref
   "CEM43 dose uses the **Chapter 15** convention" → **Chapter 16** (Safety §16.5; Ch15 is
   Transcranial, which does not define CEM43); (2) "the **Chapter 29** brain Figure 5 nonlinear
   branch" (Ch29 is Intravascular Ultrasound — wrong) → generalized to "the transcranial
   brain-imaging nonlinear-reconstruction figure"; (3) Figures: the §24.4/§24.5 plain figure-name
   lists were not embedded — embedded the **5 real `figures/ch25/*`** (registered_ct_mri_mni,
   bowl_phase_correction, essential_tremor_ablation, gbm_subspot_plan, gbm_bbb_opening) as
   Figs 24.1–24.5 with captions, and kept the two conditional outputs (fig06 / manifest) as a note.
   Verified `Chapter 14` (histotripsy) prereq is correct. No missing capability.

32. ✅ **AUDITED: Chapter 23 (LIFU-Mediated BBB Opening).** Physics (Keller–Miksis, Blake
   threshold, Hill dose–response, Pennes/CEM43, closure kinetics, multi-spot delay laws, sparse-array
   grating-lobe envelope, passive-cavitation dose) sound. Code verified: a search agent confirmed
   **15/15** cited Rust-core functions exist in `kwavers_physics::analytical::{transducer::beam,
   cavitation::passive_dose}` (one rename: `cavitation_emission_bands`→`decompose_emission_spectrum`).
   **Reversed a stale "not-implemented" claim**: §23.10 said `BubbleDynamics` is "a near-term roadmap
   item" whose `build_plugin` arm "returns a structured `ConfigError::InvalidValue`" — but the catalog
   arm **actually constructs a real `BubbleDynamicsPlugin`**. Corrected the prose and added the
   `BubbleDynamics{KellerMiksis}` capability to the worked-example config. Other fixes: (1) prereq
   cross-ref "Chapter 15 (Safety and Dosimetry)" → **16** (Ch15 is Transcranial); (2) worked-example
   flat paths + `BoundaryType::Absorbing`→`PhysicsBoundaryCondition::Absorbing` + real crate imports;
   (3) **duplicate `## 23.11`** (Figure sources + References) → References = 23.12; (4) stale
   "Figure 4.16" → "Figure 23.7"; "Chapter 21 §4" → "§21.4". (5) Figures: only fig07–10 were
   embedded; embedded the remaining **fig01–06** (`figures/ch24/*`) as Figs 23.1–23.6 and rewrote
   §23.11 as a 10-figure index. No missing capability.

31. ✅ **AUDITED: Chapter 22 (Passive Acoustic Mapping).** Physics (cavitation emission model,
   DAS resolution, eigenspace noise rejection, transcranial SNR budget) sound. Code verified: PAM
   is **real and well-implemented** — `kwavers_analysis::signal_processing::pam::{DelayAndSumPAM,
   PamBeamformingMethod::EigenspaceMinVariance, …}`, and the PyO3 `passive_acoustic_map_das` binding
   + `passive_acoustic_mapping_compare.py` harness both exist as the chapter claims. Fixes: (1)
   prereq cross-ref "Chapter 16" (transcranial) → 15. (2) flat paths → split crates (DAS path,
   post-processing path). (3) §22.6 worked example: `BoundaryType::Absorbing`→
   `PhysicsBoundaryCondition::Absorbing`, import block to the real crate split, and **removed the
   fictional `HeterogeneousMedium::from_ct_scan`** (no such constructor — replaced with a note
   pointing at the real `CTImageLoader::load` + `hu_to_*` path from Ch15 §15.5). (4) Figures: the §22.7
   list of 6 described figures pointed at no embedded images; embedded the **6 real `figures/ch23/*`**
   (cavitation_spectra, dose_accumulation, das_sensitivity, vcz_coherence, eigenspace_svd,
   snr_budget) inline as Figs 22.1–22.6 and rewrote §22.7 as an index. No missing capability.

30. ✅ **AUDITED: Chapter 21 (Simulation Orchestration).** First application chapter. Verified the
   plugin/catalog/scheduling architecture is **real and well-implemented** (a search agent confirmed
   `Plugin`, `PluginManager` + DFS topo-sort/cycle-detection, `PhysicsCatalog::build`,
   `PhysicsConfig`/`PhysicsModelConfig`, `BubbleDynamicsPlugin`, `BubbleModel{RP,KM,Gilmore}`,
   `GilmoreSolver::{step_rk4, should_use_gilmore}`, PAM). Fixes: (1) flat paths → split crates
   (`kwavers_solver::plugin::{Plugin,PluginManager,PhysicsCatalog,catalog}`,
   `kwavers_physics::factory::{config,models}`, PAM → `kwavers_analysis::signal_processing::pam`);
   stale manager line-number citation generalized. (2) **`MechanicalStress` capability does not
   exist** — `PhysicsModelType` has 5 variants, not 6; removed the row and documented the gap
   (elastic-wave solver exists but isn't a catalog variant — backlog). (3) worked example used
   `BoundaryType::Absorbing` → real `PhysicsBoundaryCondition::Absorbing`, and fixed the import
   block to the real crate split. (4) **duplicate `## 21.7`** (Visualisation + BubbleDynamics) →
   renumbered the second cluster to 21.8.x and References to 21.9; figure labels 22-B→21-B. (5)
   stale cross-ref "Chapter 16" (transcranial) → 15. (6) **Theorem 21.3 was physically wrong**: it
   claimed `BubbleState::new` sets `p_gas = p₀` (underpressured → contracts under zero forcing), but
   the constructor sets `p_gas = p₀ + 2σ/R₀` (mechanical equilibrium), and the cited test
   `step_rk4_surface_tension_drives_contraction_from_underpressured_state` **does not exist**.
   Rewrote it to the real, already-tested invariants (positivity, equilibrium stability,
   forced contraction) citing the two genuine tests `step_rk4_bubble_stable_at_equilibrium` and
   `step_rk4_compressive_forcing_contracts_bubble`. (7) embedded the 2 real `figures/ch22/*` SVGs
   (capability_fanout, field_dependency_dag) as Figs 21.1–21.2.

29. ✅ **AUDITED: Chapter 19 (Validation and Benchmarking).** Methodology/results chapter; physics
   (Pearson/PSNR/convergence/dispersion theorems) sound and the §19.9 parity numbers match the
   project memory records. Fixes: (1) title "Chapter 20"→19; (2) bare `## 1.`–`## 13.` → `## 19.N`
   + subsections + cross-refs (caught and fixed a double-prefix `§19.19.x` regex artefact); (3)
   flat paths → split crates: footer `kwavers_solver::validation` (exists), §19.7.1 test-location
   table remapped to the real `crates/kwavers-*` paths, arch test at `crates/kwavers/tests/`. (4)
   **Reversed a Ch13 false negative**: §19.10.2 "RITK registration" is **actually implemented** —
   `RitkRegistrationEngine` (`kwavers_physics::…::fusion::registration`, `ritk-registration` crate)
   does rigid/affine mutual-info + Demons non-rigid registration. Corrected §19.10.2, **and went
   back to fix Chapter 13** (§13.4.1 loop note + §13.7 code-map) and backlog #9 (⬜→✅). (5) Figures:
   embedded the 5 real `figures/ch20/*` (pearson_phase, psnr_amplitude, pstd_convergence,
   side_by_side_parity, validation_scatter) as Figs 19.1–19.5; rewrote §19.12 as an index. No
   missing capability. (Also spotted a stale `kwavers-math/.../staggered_grid_draft_<timestamp>`
   dir — flagged for cleanup.)

28. ✅ **AUDITED: Chapter 18 (Performance and Memory).** Heaviest doc-vs-code drift after Ch15.
   Fixes: (1) **title said "Chapter 19"** → 18. (2) bare section numbering `## 1.`–`## 14.` →
   `## 18.N` + subsections + cross-refs (`Section N`→`Section 18.N`, `§11.x`→`§18.11.x`), matching
   the book convention. (3) flat `kwavers::` paths → split crates; **GPU PSTD is a separate
   `kwavers_gpu` crate** (not `solver::…::gpu_pstd`), CPML→`kwavers_boundary`,
   sensor→`kwavers_receiver`, progress→`kwavers_solver::interface::ProgressReporter`. (4)
   **Fundamental falsehood corrected**: chapter claimed fields are `Vec<S> where S: Scalar` with
   "const generics for compile-time precision" — **no `Scalar` trait exists**; the CPU solver is
   monomorphic `f64`, GPU is `f32`. Rewrote §18.10.3, the §18.5 kernel, and §18.7.4 to the real
   types. (5) `AbsorptionKernel<S>`/`AbsorptionKernelInner` with `absorb_*` fields → real
   non-generic `AbsorptionKernel { tau, eta, nabla1, nabla2, alpha_si: Array3<f64> }`, held as
   `Option` at the orchestrator (lossless elides all 5 → ~670 MB, was mis-stated 536+134). (6)
   `CpmlUpdater` `TILE_SIZE` const-generic tiling → marked **not shipped** (real `CPMLUpdater` is a
   stateless axis-wise updater). (7) GPU TDR loop: `BATCH_SIZE=64`/`Maintain::Poll`/wrong file →
   real `STEP_BATCH=32`/`PollType::Wait` in `kwavers_gpu::pstd_gpu::time_loop::run`. (8) Figures:
   embedded the **5 real `figures/ch19/*`** (roofline, fft_scaling, memory_budget,
   gpu_cpu_throughput, checkpoint_overhead) as Figs 18.1–18.5; rewrote §18.13 as an index. Verified
   present: PSTD stepper, checkpoint (KWCP magic, run_to/from_checkpoint), 3-D beamforming. Filed a
   `[arch]` backlog item for genuine `Scalar`-trait genericization (aligns with project standards).

27. ✅ **AUDITED: Chapter 17 (Sonogenetics).** Core physics module is **fully and correctly
   implemented** — a search agent confirmed all 14 capabilities with exact field/fn names
   (`VolumetricArfField`{new,accumulate,finalize,intensity}, `compute_membrane_tension`,
   `compute_radiation_pressure`, `CellMembraneParams`, `boltzmann_p_open`,
   `pressure_threshold_p_open`, `ion_current`, `MechanoChannel`{MscLG22S…HsTrpa1},
   `BoltzmannGatingParams` (4 fields exact), `PressureThresholdParams`, `LifNeuron`{new,step,
   mean_firing_rate}, `LifParams`). Doc fixes: (1) flat `kwavers::` paths (all) → split crates,
   incl. `clinical::therapy`→`kwavers_therapy::therapy::therapy_integration`,
   `domain::therapy::microbubble`→`…::microbubble_dynamics`. (2) stale cross-refs: equations
   `11.x`→`17.x` (chapter was Ch11), `10.15`→`15.15` (transcranial), `Chapter 10`→`Chapter 15`.
   (3) §17.11.2 example used non-existent PSTDSolver methods — `add_focused_transducer`→
   `add_source_arc`, `run_steps`→`step_forward` loop, `pressure()`→`pressure_field()`; `BODY_TEMP_K`
   import→`kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K`. (4) §17.11.4
   `ClinicalSafetyError` (no such type)→`KwaversError::Validation`. (5) Figures: chapter had a
   descriptive §17.14 list with a nonexistent `.svg`; embedded the **5 real `figures/ch18/*`**
   (channel_gating, radiation_force, streaming_shear, safety_budget, activation_comparison) inline
   as Figs 17.1–17.5 and rewrote §17.14 as an index, honestly flagging the 2 ungenerated figures.
   No missing core capability.

26. ✅ **AUDITED: Chapter 15 (Transcranial Ultrasound).** Large canonical chapter; the heaviest
   drift found so far. Fixes: (1) **flat `kwavers::` paths** (all 23 occurrences) → split crates
   (`kwavers_physics`/`kwavers_solver`/`kwavers_medium`/`kwavers_imaging`/`kwavers_receiver`).
   (2) **Equation numbering** — all 28 equation tags were `(10.x)` (legacy Ch10) while prose
   referenced `(15.x)`; renumbered tags + §15.12 theorem-table to 15.x. (3) **5 fictional API
   names** in the code-map / §15.11.5 example that don't compile — verified real names:
   `TimeReversalFocus`→`TranscranialAberrationCorrection::apply_time_reversal_correction`,
   `TranscranialAberrationCorrector`→`TranscranialAberrationCorrection` (`compute_delays`→
   `calculate_correction`), `SkullThermalMonitor`→`TranscranialSafetyMonitor`, `ElementDelayTable`
   (no such type — delays are the `PhaseCorrection` output), `FocusAnalyzer`→
   `analytical::skull::strehl_ratio`; CT API `CT::load_dicom/to_acoustic`→`CTImageLoader::load` +
   `hu_to_sound_speed/hu_to_density`; PSTDSolver `add_point_source/run_to_end`→orchestrator API.
   The §15.11.5 "Full Workflow Example" (non-compiling Rust) was rewritten as a clearly-labeled
   conceptual outline over the real building blocks. (4) **Figures**: all 9 referenced
   `figures/ch_tc/*` — a directory that does not exist (every image broken). Remapped to the 5 real
   `figures/ch16/*` generated by `ch16_transcranial_ultrasound.py` (insertion_loss, phase_aberration,
   strehl_ratio, ct_conversion, skull_temperature) with corrected captions; removed the 4 with no
   generated image (intro schematic, TR-workflow diagram, hot-spot map, NICE plot — I do not
   fabricate figures). (5) **Physics note**: documented that the shipped HU→acoustic map is the
   Aubry 2003 bilinear form, not the chapter's piecewise-linear Eqs. (15.9)–(15.10). No missing
   *core* capability — all transcranial physics exists (`analytical::skull::{strehl_ratio,
   skull_transfer_matrix_transmission, skull_phase_screen}`, `acoustics::transcranial::*`);
   filed a `[minor]` ergonomics item for a turnkey transcranial pipeline + the 4 ungenerated figures.

25. ✅ **AUDITED: Chapter 13 (Theranostics).** Chapter was already de-duplicated (recap headers
   defer to Cavitation §5.x / Histotripsy), so the ✂️ flag was largely satisfied. Code-map §13.7
   verified against source — found **3 wrong struct names** (name-only drift): `RayleighPlesset`→
   `RayleighPlessetSolver`, `CavitationCloud`→`CavitationCloudDynamics`, `PlaneWaveCompounding`→
   `PlaneWaveCompound` (also updated detection structs to `{Broadband,Spectral,Subharmonic}Detector`
   and added `KellerMiksisModel`). **Honesty fix:** the code-map claimed image registration "via
   RITK crate (`DeformableRegistration`)" — no such type exists in kwavers (only ITK DICOM-loader
   mentions); marked **not implemented** in the code-map and the loop algorithm step 2, and filed a
   backlog component. **Structural:** theorems renumbered 13.5/13.6/13.7→13.1/13.2/13.3 and
   equations 13.7–13.11→13.1–13.5 (both started at a stale high number after recap stripping; no
   external cross-refs). **Physics:** §13.8 control-window intensities were ~5–9× wrong
   ([1300,11800]→[6.1×10³,5.5×10⁴] W/m², I=P_neg²/2ρc verified). Kalman state-estimator (loop step
   3) confirmed backed by `kwavers_analysis::…::localization::bayesian::filter`.

24. ✅ **AUDITED: Chapter 12 (Therapeutic Ultrasound).** Physics reviewed; code-map §12.9
   verified 19/19 (all symbols FOUND, re-exported at the cited level despite deeper internal
   submodules — `HIFUPlanner`, `PennesBioheat`, `ThermalDiffusionSolver`, `ThermalDoseCalculator`,
   `ThermalCEM43Grid`, lithotripsy `{ShockWaveGenerator, StoneFractureModel, CavitationCloudDynamics}`,
   `MicrobubbleDynamicsService`, `IntensityTracker`, `SafetyController`,
   `TherapyIntegrationOrchestrator`, sonogenetics `{VolumetricArfField, MechanoChannel, LifNeuron}`,
   `HistotripsyScenario`, `PulsePattern`, `clinical_scenarios`). All 5 figures (ch06 dir, kept per
   the renumber convention) exist. **Physics fixes:** (a) §12.10 worked example had a 10× units
   error — I_face = (3e5)²/(2·1060·1540) = 2.76 W/cm² (27,566 W/m²), mislabeled "27.5 W/cm²",
   propagating to an implausible 479 °C; corrected to I_focal ≈ 2690 W/cm², Q ≈ 3.77e8 W/m³,
   ΔT_adiabatic ≈ 49 °C (focal ≈ 86 °C), with G corrected 30.8→31.2; (b) Eq 12.6 liver/240-min
   contradiction with the §12.3.1 table (liver = 25 min) resolved; (c) Eq 12.10 MI units
   annotation (`kPa/√MHz = MPa^0.5`) corrected to the dimensionless MPa·MHz⁻⁰·⁵ convention.
   No missing components.

23. ✅ **WIRED: L-BFGS quasi-Newton FWI driver** (future-enhancement #3).
   Factored the Nocedal two-loop recursion out of `kwavers_math::optimization::minimize` into a
   reusable `LbfgsMemory` (SSOT: `minimize` now consumes it). Added
   `FwiProcessor::invert_lbfgs(observed, initial, geometry, grid, memory)` and
   `FwiProcessor::misfit_and_gradient` (the forward+adjoint+regularization pass, factored out of
   `descent_update`). The driver uses `LbfgsMemory::direction` for `d=−H·g`, the **un-normalized**
   gradient (so curvature pairs keep physical units), and an Armijo projected line search; the
   first step (empty memory ⇒ steepest) is scaled by `step_size/‖g‖∞`. 2 value-semantic tests
   (single-shot recovers a +60 m/s anomaly: misfit < ½ initial, anomaly-cell + illuminated-region
   error fall; stationary at the zero-misfit truth). Clippy-clean. Inverse §9.1.
   NOTE: the existing default Tikhonov weight (O(1)) swamps few-voxel physical gradients — the
   test runs pure data-misfit; production use sets regularization to the problem scale.

22. ✅ **IMPLEMENTED: CMUT squeeze-film damping** (MEMS-depth future-enhancement #6, part).
   `CmutCell::{squeeze_film_damping (3πμa⁴/2g₀³), squeeze_number (12μωa²/p_a g₀²),
   squeeze_film_quality_factor}` — the gap-gas damping channel for vented/non-evacuated CMUTs
   (sealed-vacuum immersion CMUTs are radiation-damped). 1 value-semantic test (c∝a⁴/g³, Q falls
   with viscosity, σ rises with f). Chapter 33 §33.6 + code-map updated. MEMS-depth remaining:
   crosstalk, collapse-mode nonlinear drive, flexible MEMS beamformer.

21. ✅ **COMPLETED: full Goldstein residue-aware unwrap** (future-enhancement #5b — auto
   branch-cut placement). Added `goldstein_branch_cut_mask` (robust **ground-each-residue-to-
   nearest-border** strategy — no valid loop encircles a residue) and `goldstein_unwrap_2d`.
   Dipole test verifies branch-cut correctness by **continuity** (seam-free unwrap, no 2π jump
   between adjacent valid pixels); residue-free reduces to the Itoh plane. 6 tests total green.
   Residue-aware MRE unwrapping is now end-to-end.

20. ✅ **IMPLEMENTED: Residue-aware phase unwrapping** (future-enhancement #5).
   `kwavers_signal::phase::goldstein::{phase_residues, residue_count, is_unwrap_reliable,
   masked_unwrap_2d}` — exact plaquette residue detection (Goldstein step 1), an Itoh-reliability
   gate, and a masked BFS flood-fill unwrap that routes around residues. Elastography §11.13 +
   `unwrap` module doc updated.

19. ✅ **IMPLEMENTED: Bulk-piezo thickness-mode resonator (Mason/IEEE)** (future-enhancement #8).
   `kwavers_transducer::bulk_piezo::BulkPiezoResonator` — stiffened sound speed, antiresonance
   `f_p=c_D/2t`, clamped capacitance, series resonance via bisection of the IEEE thickness `k_t²`
   relation, `coupling_from_frequencies`. 4 value-semantic tests green. Closes the Sources §2
   Mason theory gap; the bulk-PZT therapy workhorse behind Chapter 33 §33.9.

18. ✅ **CMUT/PMUT therapeutic-regime extension** (user follow-up: therapy needs high pressure at
   2–5 MHz, and flexing a capacitive CMUT cuts output). Added CMUT gap-limited output ceiling +
   `flex_gap_derating`, PMUT drive-scaled output, `plate::flexible_output_factor`, and
   `comparison::evaluate_therapy`. 4 tests prove: CMUT output saturates (gap-limited) and *flexing
   reduces it further, tighter gaps worst*; PMUT output ∝ drive; therapy verdict = PMUT (opposite
   of the IVUS imaging verdict). Chapter 33 §33.9 added; scope broadened (imaging→CMUT,
   therapy→PMUT/bulk-PZT); PyO3 bindings + ch33 fig06; kwavers-python compiles.

17. ✅ **IMPLEMENTED: Zener (standard-linear-solid) viscoelastic model** (future-enhancement #2).
   `kwavers_medium::viscoelastic::ZenerModel` — bounded-dispersion SLS companion to
   `KelvinVoigtModel`; complex modulus, storage/loss, Debye loss peak at ωτ=1, relaxed/unrelaxed
   speed bounds. 4 value-semantic tests green. Elastography §11.8 updated.

16. ✅ **NEW CHAPTER 33 + CMUT/PMUT models** (user request; `[major]`, gated by **ADR 015**;
   supersedes the bulk-piezo/CMUT backlog item). New `kwavers_transducer::mems` (plate / CMUT /
   PMUT / IVUS comparison) — 13 value-semantic tests green; PyO3 `mems`/`cmut`/`pmut`/
   `ivus_figure_of_merit` bindings (kwavers-python compiles); new Chapter 33
   `cmut_vs_pmut.md` (electrical, manufacturing, heating, bandwidth, flexible/IVUS verdict —
   CMUT favoured by the simulation) + `ch33_cmut_vs_pmut.py` figure script; README Part VIII
   added (book now 1–33); Sources §2 cross-referenced. ch33 figure PNGs pending a maturin
   rebuild. **Backlog queue now empty** — all documented-but-missing components implemented;
   future-enhancement list recorded in backlog.md.

15. ✅ **IMPLEMENTED: Acousto-elasticity (Murnaghan) — stress-dependent wave speed + pre-stress
   inversion** (user directive; `[major]`, gated by **ADR 014**; scope = analytical
   relation/inversion, full 3rd-order PDE deferred). `kwavers_physics::analytical::elastography::
   {acoustoelastic_sensitivity, acoustoelastic_shear_speed, estimate_prestress,
   estimate_prestress_sequence}` — `ρc_S²=μ+Aσ₀`, `A=(m+n)/(2(λ+μ))`, `σ₀=ρ(c_S²−c_S0²)/A`.
   4 value-semantic tests (σ₀=0→√(μ/ρ); A formula; round-trip exact; cardiac-sequence per-frame
   recovery). Elastography §11.9 updated. Re-verified missing first (no Murnaghan/acousto-elastic
   anywhere; the nonlinear path is hyperelastic, a distinct formulation).

14. ✅ **IMPLEMENTED: CEUS contrast pulse sequences** (user directive).
   `kwavers_physics::acoustics::imaging::modalities::ceus::pulse_sequences::{pulse_inversion,
   amplitude_modulation, cps_combine}` — multi-pulse linear-cancellation combiners (Simpson
   1999 PI / Phillips 2001 CPS). 3 value-semantic tests with a quadratic scatterer (PI cancels
   the fundamental, keeps 2f; AM cancels linear, nonlinear residual survives; CPS≡PI).
   Diagnostics §9.4 updated — that chapter now has **no** "not yet implemented" markers.
   Re-verified missing first (the ceus module had harmonic *filtering* + coherence weighting,
   not the multi-pulse PI/AM combiner).

13. ✅ **IMPLEMENTED: Acoustic CT — Radon + filtered backprojection** (user directive; first
   `[major]`, gated by **ADR 013**). `kwavers_diagnostics::reconstruction::radon::
   {radon_transform, filtered_backprojection}` — parallel-beam forward projection (bilinear ray
   sampling) + Ram-Lak ramp-filtered backprojection. 3 value-semantic tests green (centred-disk
   round-trip Pearson>0.8 + centroid; off-centre disk localizes within 4 px; empty→0). Inverse
   §6 updated; bent-ray SIRT/ART + reflection-CT remain (SIRT path already exists). ADR at
   docs/adr/013-acoustic-ct-radon-fbp.md.

12. ✅ **IMPLEMENTED: f-k (Stolt) migration** (user directive).
   `kwavers_diagnostics::workflows::fk_migration::fk_stolt_migration` — exploding-reflector
   Stolt k-space remap (ω=v·sign(k_z)√(k_x²+k_z²), v=c/2) with ω-interpolation + obliquity
   Jacobian over the 2-D FFT helpers. 2 value-semantic tests green (flat reflector → correct
   migrated depth ≤3 bins; point scatterer focuses to (x0,z0) ±2 lateral/±5 axial bins and
   concentrates energy vs the raw hyperbola). Diagnostics §9.2.2 / Beamforming §7.5.2 updated.
   This clears the **last `[minor]` backlog item** — remaining are `[major]` (acoustic-CT
   Radon/FBP, CEUS sequences, Murnaghan, piezo/CMUT), each needing an ADR.

11. ✅ **IMPLEMENTED: Kelvin–Voigt viscoelastic medium kernel** (user directive).
   `kwavers_medium::viscoelastic::KelvinVoigtModel` — frequency-domain complex shear modulus
   G*(ω)=μ+iωη, storage/loss/Q, dispersive phase velocity and attenuation via k=ω√(ρ/G*).
   5 value-semantic tests green (storage+i·loss, tanδ·Q=1, ω→0 elastic limit √(μ/ρ),
   dispersion+attenuation rise with ω, lossless η=0 limit). Elastography §11.8/§11.13 updated.
   Re-verified missing first (the medium layer only stored the viscosity *coefficient*; the
   complex modulus/dispersion existed only as analytical Vec-helpers in kwavers_physics).
   f-k/Stolt migration deferred (apollo Fft2d conventions + synthetic-RF forward-model test
   warrant a dedicated effort).

10. ✅ **IMPLEMENTED: L-BFGS quasi-Newton optimiser** (user directive).
   `kwavers_math::optimization::{minimize, LbfgsConfig, LbfgsResult}` — Nocedal two-loop
   recursion + Armijo backtracking, curvature-guarded limited-memory updates. 3 value-semantic
   tests green (SPD quadratic → A⁻¹b in ≤15 iters, separable quartic, zero-gradient immediate
   return). Inverse §9.1 updated; FwiProcessor wiring is the remaining integration step.
   Re-verified missing first (matches were domain `optimize()` methods + MAML, not a general
   optimiser).

9. ✅ **IMPLEMENTED: ConstrainedInversion (projected-gradient box constraints)** (user
   directive). `kwavers_math::inverse_problems::{BoxConstraints, projected_gradient_descent}` —
   pointwise box projection (Π) with `sound_speed_tissue()`/`density_tissue()` presets + PGD
   over any gradient closure; converges to the projection of the unconstrained minimiser for
   separable convex objectives. 4 value-semantic tests green (bound ordering, clamp/keep, PGD
   convergence on a quadratic, zero-gradient fixpoint). Inverse §8.4 updated (no longer "design
   target"). Re-verified genuinely missing first (matches were unrelated projections).

8. ✅ **IMPLEMENTED: 2-D phase unwrapping** (user directive). `kwavers_signal::phase::{unwrap_1d,
   unwrap_2d}` — separable Itoh path-following (exact for residue-free fields). 4 value-semantic
   tests green (1-D ramp exact recovery across a genuine wrap, 2-D plane exact, identity on
   smooth, empty-input). Elastography §11.13 updated; backlog notes the residue-aware Goldstein
   branch-cut variant as the remaining upgrade. (Confirmed genuinely missing first — only a
   private 1-D unwrap existed in `modulation::phase`.)

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

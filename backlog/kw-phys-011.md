<a id="kw-phys-011"></a>

## KW-PHYS-011 — Population evolution under an imaging pulse train (validated negative: acquisition is inert) [patch] — done (Nov 21, 2026)

priority: feature; needs: none; scope: `kwavers-physics` bubble-dynamics population (`acoustics/bubble_dynamics/population/imaging_evolution.rs`); milestone: none

- **Outcome:** pulse-to-pulse evolution of a bubble over a triplet-pulse imaging acquisition — rectified diffusion during each pulse, Epstein–Plesset dissolution in the inter-pulse gaps, fragmentation gate per pulse — is a library component with its validated regime documented: imaging pressures only, never trigger pressures.
- **Driver:** the prototype holds the cloud fixed across the 3P acquisition; Build 14 B2 tested whether PNP-dependent evolution *during* imaging could create Kuji's 0.26 MPa peak. It cannot: the acquisition is four pulses over 0.6 ms, population loss is 0 % at every PNP, and the only change is PNP-independent Epstein–Plesset shrink of 0.24–1.74 %. The fixed-cloud assumption is vindicated for a single acquisition, and the negative result is library knowledge worth keeping in code.
- **Acceptance:** Epstein–Plesset inter-pulse dissolution in the full form (Laplace + transient terms, as in the formation harness); rectified term reuses the Louisnard–Gomez law (KW-PHYS-004) via caller-supplied per-pulse rates — the law is not duplicated; fragmentation gate reuses the population module's Stage-7 gate (Rmax/R0 > 5); tests reproduce the B2 numbers (0 % loss at PNPs 0.12–1.23 MPa over a 4-pulse / 600 µs acquisition; PNP-independent EP shrink 1.74 / 1.05 / 0.53 / 0.24 % for R0 = 1.5 / 2 / 3 / 5 µm).

### Resolution (Nov 21, 2026)

- **Implementation.** `population/imaging_evolution.rs`: `EpsteinPlessetMedium` (with `kuji_degassed()` — Kuji §2.5's degassed condition, saturation ratio 0.175), `epstein_plesset_step` (the formation harness's `ep_shrink_factor` arithmetic), `ImagingTrain` (`kuji_2024()`: 4 pulses, 200 µs gaps, age 1 ms, 5 cycles/pulse; `iwasaki_2018()`: 3 pulses, 0.6 ms), `ImagingPulseEffect` (per-pulse rectified rate + Rmax/R0 supplied by the caller's dynamics solve), `evolve_under_imaging_train`, gate constant `IMAGING_FRAGMENTATION_GATE_RMAX_R0 = 5.0`, removal radius 0.1 µm (B2 rule). Re-exported from `population/mod.rs`.
- **Validation.** Tests reproduce the B2 shrink percentages to within 0.02 percentage points with inert pulses, show the actual B2 Fyrillas–Szeri rates at 0.26 MPa move the result by < 0.01 points (PNP-blind), and pin the gate: Rmax/R0 = 4.55 (the B2 maximum for R0 ≥ 1.5 µm) never fires, 5.17 (the R0 = 1 µm hypothetical) fires on every pulse.
- **Regime warning carried in the docs.** Imaging pressures only (PNP ≈ 0.1–1.3 MPa, µs on-time). At trigger/burst pressures the formation-side laws govern; this module must not be applied there.

<a id="kw-img-052"></a>

## KW-IMG-052 — Representation benchmark for the triplet-pulse replicas (Kuji K1–K6, Iwasaki I1–I4) [minor] — todo

priority: research; needs: none; scope: triplet-pulse imaging replicas (Kuji 2024, Iwasaki 2018); milestone: M-imaging-capability

- **Outcome:** a standing acceptance test, defined from the papers' own reported numbers, that any future replica configuration must pass before its physics verdicts are cited as more than provisional: *"We can't say our physics approaches are reasonable until we can approximately represent the experiments through simulation"* (Ryan, 2026-11-22).
- **Driver:** Build 15 (`B15_RESULTS.md` §1) defined the benchmark and scored the current pipeline against it: **0 of 10 criteria pass**. Every prior mechanism verdict in the programme (Stages 0–8, Builds 1–14) is provisional until a configuration passes — within-pipeline results on a base that does not yet approximately represent either experiment. Tolerances are set from the papers' own spread: Kuji five-experiment averages with Fig. 14 error bars of order ±1.5–2 dB (digitisation adds ±0.4–0.7 dB, WS-C); Iwasaki SDs ±1.93 / ±3.53 / ±3.74 dB. All quantities are in the corrected metrics (KW-IMG-047 for Kuji, KW-IMG-049 for Iwasaki).
- **Benchmark — Kuji 2024 (paper values: WS-C digitisation of Figs. 11–14):**
  - K1: tissue reduction @ fc 3.0 MHz, PNP 0.12/0.26/0.72/1.23 MPa = 31.5 / 31.4 / 29.3 / 25.0 dB, ±3 dB each.
  - K2: tissue-reduction PNP slope @ fc 3.0 / 3.5 MHz (0.12→1.23 MPa) = −6.5 dB / −12.6 dB, slope within ±3 dB.
  - K3: cavitation reduction @ fc 3.0 MHz, PNP 0.26 / 1.23 MPa = 18.5 / 17.3 dB, ±3 dB; PNP-flatness Δ = −1.2 dB, flat within ±2 dB.
  - K4: contrast at optimal fc*, all four PNPs = 10.7 / 12.8 / 10.0 / 8.3 dB, ±3 dB each.
  - K5: peak — contrast(0.26) exceeds contrast(0.12) and contrast(0.72) beyond ensemble SD (margins +2.1 / +2.8 dB; audit step-10 test).
  - K6: fc* trajectory = 3.06 / 3.02 / 2.78 / 2.56 MHz, each ±0.2 MHz; fall ≥ 0.3 MHz.
- **Benchmark — Iwasaki 2018 (paper values: §3, N = 10):**
  - I1: 1P absolute contrast 28.9 ± 1.93 dB, tolerance ±4 dB.
  - I2: PI absolute contrast 34.8 ± 3.53 dB, tolerance ±7 dB.
  - I3: 3P absolute contrast 39.4 ± 3.74 dB, tolerance ±7.5 dB.
  - I4: ordering 1P < PI < 3P, gaps +5.9 / +4.6 dB; ordering with gaps ≥ 2 dB.
- **Acceptance:** a replica configuration (geometry, medium, floor, cloud per `B15_RESULTS.md` §5) whose corrected-metric results meet every applicable criterion above within tolerance, reported against this table; only then are mechanism verdicts obtained on that configuration un-provisional. Iwasaki criteria additionally require the data conditions of KW-IMG-053.

Re-score (Build 16, 2026-11-22, `B16_RESULTS.md`): **1/10 - I2 passes** (PI formable for the first time; all four phases saved per KW-IMG-053). K1-K4 fail (K1 is the controlling failure; every testable Kuji point fails, 0 of 7; Kuji coverage partial at 6/20 pairs - seeds 21-23 x 0.12/0.26 MPa - with K1 failing at both scored bins by 24-31 dB, so the missing cells cannot change the verdict). I1 fails at +5.1 dB (moved halfway), I3 fails at -13.9 dB (moved substantially), I4 still reversed. K5/K6 NOT ADJUDICATED (gate: the benchmark must pass first), so all mechanism verdicts obtained on non-passing configurations remain provisional. The benchmark remains the standing acceptance test. Status stays OPEN.

Re-score note (Build 17, 2026-11-23, `B17_RESULTS.md`): score UNCHANGED at 1/10 (I2). No re-score was triggered: the discrete-scatterer configuration overshoots K1 by +7.7 dB (39.1 floored vs 31.4 +/- 3), outside the +/-3 dB rescue criterion. K1's failure is now attributed to the tissue model's uncalibrated absolute backscatter level (KW-IMG-056): the smooth and discrete media bracket the paper at the 0.26 MPa bin. K5/K6 remain NOT ADJUDICATED. Status stays OPEN.

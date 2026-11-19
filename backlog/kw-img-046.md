<a id="kw-img-046"></a>

## KW-IMG-046 — Triplet-pulse cavitation-bubble imaging [minor] — todo

priority: feature; needs: none; scope: the triplet-pulse (3P) sequence, coherent 1P/PI/3P sums, Kuji high-pass + 3P/1P ratio filter, and Keller-Miksis bubble-cloud coupling; milestone: M-imaging-capability

- **Outcome:** a 1.74 MHz diverging-burst acquisition transmits at 0°/120°/240° (+180° for PI), combines the echoes into 1P/PI/3P, high-pass filters before the combination, and couples a Keller-Miksis bubble cloud into the received RF via monopole re-radiation — so the bubble's 1.5th fractional harmonic survives the 3P sum (~1/3 amplitude) while the tissue fundamental cancels (~1%), and image contrast orders 1P < PI < 3P.
- **Driver:** no triplet-pulse/PI/3P sequence, coherent sum, or 3P/1P filter exists in the workspace (closest: KW-IMG-045 frame-resolved I/Q ensemble; KWAVERS-AEQ-MET-69 B-mode scan-conversion). The Python prototype (pykwavers PSTD + Keller-Miksis, validated 2026-11-16 against Iwasaki 2018 / Kuji 2024) is the reference.
- **Acceptance:** phasor rule (0.5th→2/3, 1st→0, 1.5th→1/3, 2nd→0, 3rd→1 in 3P); tissue-only 3P/1P f0 ≤ −105 dB and 1.5f0 ≤ −83 dB at 0.69 MPa slab PNP; bubble-only 3P/1P at 1.5f0 within ±0.05 of 0.34/0.29/0.15 for R0 = 2/3/5 µm; 1.5f0-band image contrast orders 1P < PI < 3P at every R0; Kuji sweep: optimal high-pass cutoff falls as imaging PNP rises and contrast peaks at 0.26 MPa. R0 is always a sweep, never a single value; all experimental values are chicken-breast research values, not Suizenji clinical settings.

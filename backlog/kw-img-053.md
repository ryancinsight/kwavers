<a id="kw-img-053"></a>

## KW-IMG-053 — Iwasaki replica data requirements (four-phase acquisition; interface-free numerator) [minor] — blocked

priority: feature; needs: none; basis: eb91da65678bcd26cba72e67ae42b1b80e732132; scope: triplet-pulse imaging replicas (Iwasaki 2018 acquisitions); milestone: M-imaging-capability

- **Outcome:** future Iwasaki-condition acquisitions (1.74 MHz) save everything the corrected Iwasaki statistic (KW-IMG-049) needs, so criteria I1–I4 of the representation benchmark (KW-IMG-052) are adjudicable as an ensemble rather than blocked or artefact-driven.
- **Driver:** Build 15 WS-I (`B15_IWASAKI.md`) re-ran the Iwasaki analysis on the saved D5 ensemble and found two data-level blockers. (1) **The 180° phase was never saved for tissue:** the D5 ensemble holds tissue PSTD RF at 0°/120°/240° only, so PI = (0°+180°)/2 cannot be formed — I2 is NOT TESTABLE as an ensemble, and synthesising a bubble-only 180° to pair with a non-existent tissue 180° would fabricate the tissue half of the PI sum. (2) **The prototype's 1P numerator is not the cloud:** the 1P full-image spatiotemporal maximum is captured by the slab-top specular echo (2802 Pa vs the cloud disk's 166 Pa, seed S2) — a phase-invariant interface echo that the phase sums then cancel, which is the identified reason the corrected statistic gives a REVERSED ordering (1P > PI > 3P) on this data. The experiment (sector probe, chicken breast, focus 95 mm) has no such flat interface in its field of view.
- **Requirements for any future 1.74 MHz acquisition intended for Iwasaki comparison:**
  - Save **all four phases including 180°** (0°/120°/180°/240°), tissue and combined records alike, so 1P, PI and 3P all form from saved data.
  - **Sector geometry or an interface-free field of view:** no flat slab-top (or equivalent specular interface) may sit where it can capture the 1P spatiotemporal maximum; if an interface is unavoidable, the numerator region must be declared and the interface echo excluded by construction, with the choice reported.
  - Ensemble N = 10 (the paper's N), per KW-IMG-052.
- **Acceptance:** an Iwasaki-condition dataset on file satisfying both requirements, on which I1–I4 can be scored without a NOT-TESTABLE entry or a specular-echo numerator.
- **Blocker (2026-11-24):** Build 16 reports the requirements met (`b16/gen_iwa16.py`: four phases, interface-free diverging geometry, NY = 512; I2 passes, I1/I3/I4 fail on physics), but neither the generator nor the dataset is tracked on main, so the acceptance cannot be checked from the repository. The residual Iwasaki gap is physics, tracked under KW-IMG-050/052.
- **Re-open trigger:** the acquisition generator `b16/gen_iwa16.py` becoming tracked, or a dataset locator being recorded beside KW-IMG-052; this item deletes with that change.

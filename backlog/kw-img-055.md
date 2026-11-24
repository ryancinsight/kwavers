## KW-IMG-055 — PNP indexing estimator: replace max|p| (E0) with windowed true PNP (E1) [minor] — todo

priority: research; needs: none; scope: triplet-pulse replica scoring / run_metric tooling; milestone: M-imaging-capability

- **Finding (Build 17 Lever 4, `B17_PNP.md`):** the programme's PNP indexing estimator E0 (max|p| over the probe column, all time) conflates the positive and negative peaks of a distorted waveform. The positive peak grows faster with drive, so E0 overstates PNP by a drive-dependent factor (E1/E0 = 0.96 at 0.09 MPa → 0.80 at 0.37 MPa) and adds estimator scatter on top of the genuine focal-field speckle: cross-seed CV at fixed source amp 0.273/0.291 (E0) vs 0.239/0.219 (E1) at the two replica amp groups.
- **E1 definition:** per-probe −min(p) inside the focal arrival window (30–60 µs at the long geometry), spatial peak over the 9 probe-column probes with parabolic depth interpolation. Computable from the same saved probe rows at zero extra cost.
- **Non-finding (recorded so it is not retried):** a focal-volume estimator from DAS-beamformed receive RF (E2) is NOT viable on the replica's saved data — the beamformed ROI signal is the backscattered echo (~3–70 Pa) while the in-situ PNP is ~1e5 Pa; it measures the echo, not the field. A focal-volume estimator requires in-situ sampling or a transmit-side field model.
- **Ask:** switch replica scoring/calibration tooling to E1 for PNP indexing and Γ evaluation; keep E0 values in historical records labelled as E0. B16's bin assignments are unchanged under E1 (verified on all 6 scored pairs); PNP labels shift down ~12% on average.
- **Acceptance:** scoring tooling reports E1 (with E0 retained as a diagnostic column) and reproduces the B17_PNP.md table from the saved B16 RF.

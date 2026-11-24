<a id="kw-img-049"></a>

## KW-IMG-049 — Corrected Iwasaki 2018 statistic (spatiotemporal max ÷ same-frame background) [minor] — done (Nov 21, 2026)

priority: feature; needs: KW-IMG-047; scope: `kwavers-analysis` signal processing (`signal_processing/triplet_pulse_metrics.rs`); milestone: M-imaging-capability

- **Outcome:** Iwasaki 2018's contrast — spatiotemporal maximum brightness ÷ mean background brightness over depth 70–100 mm × width ±16 mm in the same frame set, per realisation over N = 10 — is computable from coherent-DAS envelope frames with the averaging domain declared.
- **Driver:** the Build 14 Stage A audit (`B14_AUDIT.md` §c/§e) found the Stage 2 / D3 implementation diverged from the paper on the numerator (max restricted to an r = 2 mm disk, single frame, instead of space-and-time), the denominator (a 5 × 4 mm far strip from a *separate* tissue-only run instead of the paper's 30 × 32 mm same-image background), and the ensemble (single realisation, no SD, vs the paper's N = 10 mean ± SD: 28.9 ± 1.93 / 34.8 ± 3.53 / 39.4 ± 3.74 dB for 1P / PI / 3P). The corrected-spec implementation was left to this entry if it did not fit the Kuji component's shape.
- **Acceptance:** coherent phase sum → coherent DAS → envelope (shared with KW-IMG-047; no band filter beyond a declared probe band); numerator = maximum envelope over the full image and all frames of the sequence; denominator = mean envelope over the paper's background region in the same frames; per-realisation contrast with mean ± SD over N = 10 reported by the caller; the paper's unresolved linear-vs-dB-brightness background averaging is exposed as a declared choice, not silently picked.

### Resolution (Nov 21, 2026)

- **Implementation.** It fit the KW-IMG-047 component shape, so it lives in the same module: `iwasaki_contrast(frames, xs, ys, background, averaging)` in `crates/kwavers-analysis/src/signal_processing/triplet_pulse_metrics.rs`, with `RectRoi::iwasaki_background()` and `IwasakiBackgroundAveraging::{Linear, DbBrightness}`. Frames are the caller's coherent-DAS envelope images (the module's `coherent_das_envelope` on phase-summed RF).
- **Tests.** Known-value image (max 10, background mean 2 including the peak pixel) → 20·log10(5) dB; spatiotemporal max verified across frames (a larger peak in a later frame raises the contrast).
- **Open residue (not this entry).** A D3 re-run under the corrected statistic has not been performed; the prototype's geometry divergences (linear array / plane wave vs the paper's diverging-wave sector probe at 95 mm) remain declared prototype limitations, so a re-run would still not be a clean paper test.

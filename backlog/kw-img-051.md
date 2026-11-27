<a id="kw-img-051"></a>

## KW-IMG-051 — Coherent DAS fails on smooth volume-scattering tissue models (medium caveat for the corrected metrics) [minor] — todo

priority: feature; needs: none; scope: triplet-pulse imaging pipeline medium model; milestone: M-imaging-capability

- **Outcome:** the imaging pipeline's tissue model provides discrete (point-like) speckle that a coherent beamformer can focus, so the corrected metrics' coherent-DAS path (KW-IMG-047) is exercised on a medium of the kind the papers' beamformers operate on.
- **Driver:** Build 14 B3 implemented coherent DAS against the prototype's tissue model (smooth ±4 % sound-speed volume scattering) and measured the failure directly: coherent/per-element envelope ratio 0.99 at the bubble pixel (delays exact by construction) but 0.13 / 0.28 / 0.09 at tissue pixels — the volume echoes' phases do not align at any homogeneous-delay DAS, and two-layer water/tissue delays do not repair it (0.20 / 0.28 / 0.07). B1's discrete-speckle slab (±4 % c- and ρ-speckle) did **not** collapse under coherent DAS (tissue reductions finite and stable at 2.4–2.7), which localises the caveat: coherent DAS works on discrete-speckle media and fails on smooth volume-scattering media.
- **Acceptance:** a declared discrete-speckle tissue configuration for the 2.45 MHz pipeline (scatterer density/size stated), with coherent-DAS tissue images demonstrated focusable (coherent/per-element ratio near unity in tissue), and the envelope-DAS fallback labelled as the prototype-parity beamformer wherever it is used for a corrected-metric number.

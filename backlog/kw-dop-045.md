<a id="kw-dop-045"></a>

## KW-DOP-045 — Signed pulsed-wave spectral Doppler [minor] — review

priority: feature; needs: none; scope: `crates/kwavers-analysis/src/signal_processing/doppler/`

- **Driver:** a downstream consumer's moving-scatterer sector ensemble needs a pulsed-wave provider that retains reverse-flow bins; the former one-sided magnitude API discarded that physical degree of freedom.
- **Acceptance:** a physical complex-I/Q trace yields a two-sided spectrum with explicit negative and positive velocity bins; reverse-flow energy stays in the returned spectrum; invalid Doppler geometry is rejected rather than mapped through an artificial positive angle cosine; an FFT shorter than the acquired ensemble fails rather than silently discarding pulses.
- **Evidence:** `kwavers-analysis` compiles and its warning-denied Clippy passes; the focused PW Nextest regression passes 8/8. All-feature Clippy reaches the separate KW-LINT-047 solver ratchet.
- **Next step:** re-validate the three rejection and two-sided-spectrum tests at the default branch; close if present.

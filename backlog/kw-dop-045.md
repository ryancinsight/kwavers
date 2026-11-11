<a id="kw-dop-045"></a>

## KW-DOP-045 — Signed pulsed-wave spectral Doppler [minor] — todo

priority: feature; needs: none; basis: acef6520f6e5c590ff73e1f39ee3e66361df7dcc; scope: `crates/kwavers-analysis/src/signal_processing/doppler/`

- **Driver:** a downstream consumer's moving-scatterer sector ensemble needs a pulsed-wave provider that retains reverse-flow bins; the former one-sided magnitude API discarded that physical degree of freedom.
- **Outcome:** the pulsed-wave provider returns a signed two-sided spectrum, and every public path of it rejects invalid geometry and an FFT shorter than the ensemble.
- **Acceptance:** a physical complex-I/Q trace yields a two-sided spectrum with explicit negative and positive velocity bins; reverse-flow energy stays in the returned spectrum; invalid Doppler geometry is rejected rather than mapped through an artificial positive angle cosine; an FFT shorter than the acquired ensemble fails rather than silently discarding pulses.
- **Witnessed.** In `pulsed_wave.rs`, `signed_spectrum` (line 173) returns the two-sided spectrum and `signed_velocity_per_hz` (line 284) rejects a perpendicular beam. The tests `signed_spectrum_retains_reverse_flow_energy_and_velocity_axis` (line 432), `signed_spectrum_rejects_perpendicular_beam_geometry` (line 482) and `waveform_rejects_an_ensemble_longer_than_its_fft` (line 398) pass in pipeline run 42970466152 (job 130357540799 log lines 1056 and 1058; job 130357540831 log line 1061).
- **Open clause.** "invalid Doppler geometry is rejected rather than mapped through an artificial positive angle cosine": the public `velocity_axis` (line 212) and `max_velocity` (line 230) still compute `beam_angle.cos().abs().max(f64::EPSILON)` (lines 215 and 231), so a perpendicular beam yields a finite velocity scale there. The overlong-ensemble test reaches the check through the one-sided `extract_waveform` (line 147), not through `signed_spectrum`.
- **Next step:** make `velocity_axis` and `max_velocity` return the geometry error `signed_velocity_per_hz` returns, or delete the one-sided API with its callers; add the overlong-ensemble rejection through `signed_spectrum`; then close.

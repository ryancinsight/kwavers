<a id="kw-img-045"></a>

## KW-IMG-045 — Frame-resolved physical I/Q ensemble [minor] — todo

priority: feature; needs: none; scope: the direct I/Q primitives as consumed by a downstream consumer's Python reference path

- **Outcome:** a color/power/PW/fUS sector sequence derives its frame-to-frame phase from submitted scatterer position and reflectivity evolution, then applies the provider I/Q demodulator and complex DAS.
- **Driver:** KW-IMG-044 made individual real-RF frames complex-I/Q capable but deliberately does not claim a physical slow-time evolution law.
- **Acceptance:** no deterministic phase-animation surrogate remains in the Python reference path.

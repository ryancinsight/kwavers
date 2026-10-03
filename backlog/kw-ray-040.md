<a id="kw-ray-040"></a>

## KW-RAY-040 — Layered focus-path contract [minor] — in-progress

priority: feature; needs: none; scope: `kwavers-transducer` layered Rayleigh propagation API, a downstream consumer's focus integration, reference-Python parity

- **Driver:** the provider's Rayleigh kernel integrates each straight-ray layer, but a downstream consumer's focus steering receives only one sound speed while the reference script separately recreates the layered phase law.
- **Acceptance:** Kwavers exposes the validated segmentwise propagation phase; the consumer focuses through that provider contract; the reference delegates instead of keeping an independent layered phase implementation.

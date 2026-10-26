<a id="kw-py-angular-spectrum"></a>

## KW-PY-ANGULAR-SPECTRUM — The k-Wave parity helper propagates angular spectra in Python [patch] — todo

priority: architecture; needs: none; scope: `crates/kwavers-python/python/pykwavers/kwave_parity.py`, `crates/kwavers-python/src/`, the Rust angular-spectrum owner in `kwavers-solver`

- **Finding (basis `0da4122f3`):** `angular_spectrum_cw` and `backward_angular_spectrum_cw` in `kwave_parity.py` (from line 226) build the k-space grid, the propagator `exp(i·z·kz)`, the absorption and angular-restriction factors, and the 2-D FFTs in numpy. That is domain logic in Python, which the binding contract forbids. Found by the judge of #964.
- **Acceptance:** Python calls a Rust angular-spectrum binding. Python-side tests show it agrees with the k-Wave reference fixtures within a derived tolerance. No numpy FFT or propagator math is left in `kwave_parity.py`.
- **Next step:** search `kwavers-solver` for an existing CW angular-spectrum operator and bind it; implement it there only if none exists.

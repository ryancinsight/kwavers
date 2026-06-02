# KWave.jl parity scripts

These five `*_jl_compare.py` scripts wire `pykwavers` against the matching
KWave.jl example for physics that has **no equivalent example** in
`external/k-wave-python/examples/`. Each script ships with a companion
`run_kwave_julia_*.jl` driver invoked via subprocess.

| Script | Physics | KWave.jl example | Status |
|---|---|---|---|
| `diff_bioheat_1d_jl_compare.py` | 1-D Pennes bioheat | `examples/diff_bioheat_1d.jl` | PASS (Pearson 0.999997, L∞ 13 mK on 2.6 °C signal) |
| `ewp_elastic_2d_jl_compare.py` | 2-D elastic (P+S) wave | `examples/ewp_elastic_2d.jl` | DIAGNOSTIC — surfaces pre-existing pykwavers `SolverType.Elastic` source-scaling regression that also breaks `external/elastic_julia_parity/compare_elastic.py`; runs without `--strict` by default |
| `pr_time_reversal_2d_jl_compare.py` | 2-D photoacoustic time-reversal recon | `examples/pr_time_reversal_2d.jl` | PASS (recon-vs-recon r=0.71, both reconstructions r≈0.4–0.6 vs p0 truth — TR with one line sensor is intrinsically lossy) |
| `us_phased_array_3d_jl_compare.py` | 3-D flat phased-array transmit | `examples/us_phased_array_3d.jl` | PASS (Pearson 0.94, peak_ratio 1.0011 after pykwavers source-injection prefactor calibration documented in script) |
| `us_beamforming_2d_jl_compare.py` | 2-D delay-and-sum beamforming | `examples/us_beamforming_2d.jl` | PASS (KWave.jl active-imaging DAS vs pykwavers PAM-DAS — different DAS variants but localise the point scatterer to within 1 cell) |

## Running

Prerequisites: Julia 1.10+ with KWave.jl's `Project.toml` resolved
(`JSON.jl` is already added). pykwavers built (`PYKWAVERS_EXTENSION_PATH`
points at the maturin-produced `.pyd`).

```bash
# Single pair
python pykwavers/examples/diff_bioheat_1d_jl_compare.py

# Full sweep
python pykwavers/examples/_run_julia_parity_sweep.py
```

Outputs land under `pykwavers/examples/output/<script>_jl_{compare.png,
metrics.txt}`.

## Why this exists

`external/k-wave-python` does not publish examples for diffusion / bioheat,
elastic-wave propagation, time-reversal photoacoustic reconstruction,
multi-element phased-array transmit, or delay-and-sum beamforming. The
existing pykwavers `*_compare.py` scripts validate those physics against
analytical references rather than against the `k-wave-python` example
files. KWave.jl publishes all five — these scripts fill the
KWave.jl-side parity coverage gap.

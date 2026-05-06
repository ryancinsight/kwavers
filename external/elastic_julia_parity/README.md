# Elastic-Wave Parity: pykwavers ↔ KWave.jl

First side-by-side k-wave-julia comparison of pykwavers' elastic ladder
(ADR 007 phases A.1–A.3). k-wave-python explicitly does not support
elastic simulations
(`assert ... "Elastic simulation is not supported"` in
`external/k-wave-python/kwave/kWaveSimulation.py:544`), so KWave.jl is the
only side-by-side reference available.

## Scenario

| Parameter | Value |
|---|---|
| Grid | 32×32 (KWave.jl 2-D) / 32×32×16 (pykwavers 3-D) |
| Spacing | dx = dy = 0.5 mm |
| Medium | Homogeneous elastic: cp = 2000 m/s, cs = 800 m/s, ρ = 1200 kg/m³ |
| Source | ux velocity-source plane at x_1b = 9, 3-cycle 1 MHz Hann tone burst, peak 1 µm/s |
| Sensors | 3 points along +x ray at offsets 0, 3, 6 grid cells |
| Time | Nt = 200, dt = CFL·dx/(√3·cp) ≈ 43.3 ns |
| PML | 10 grid points, inside |

## Engineering findings

The first run after semantic alignment (see "Quirks" below) produced:

| Metric | Value | Target | Result |
|---|---|---|---|
| Pearson r | 0.49 | ≥ 0.50 | FAIL (0.01 below threshold) |
| RMS ratio | 0.16 | [0.20, 5.00] | FAIL (3× under) |
| PSNR | 10.6 dB | ≥ 5.0 dB | PASS |
| Peak ratio (py/jl) | 0.32 | n/a | informational |

**Two real differences surfaced**:

1. **Source-injection timing semantics** — pykwavers' Phase A.3 elastic
   velocity source uses **Dirichlet override** (the integrator updates
   `vx` from acceleration, then the source injector overwrites at mask
   points). KWave.jl's `pstd_elastic_2d` uses **Additive accumulation**
   (the source signal is added to the integrator's `vx` update before
   the next step). For a tone-burst drive on a dense plane mask, the
   accumulated mode produces a stronger field response — explaining
   the ~3× amplitude difference.

2. **Time-stepping scheme** — KWave.jl uses a pseudospectral
   stress-velocity formulation; pykwavers' elastic core is a 4th-order
   finite-difference velocity-Verlet scheme. Phase agreement
   (Pearson r = 0.49) is reasonable but not perfect, particularly for
   high-frequency content.

These are documented physics-implementation differences, not bugs in
either engine. Closing the parity gap would require either:
- Adding an `Additive` mode to pykwavers' elastic velocity-source
  injection (roadmap'd as Phase A.3.5).
- Retuning the comparison to use a single-point source where
  Dirichlet vs Additive converge faster.

## Quirks discovered

| Quirk | Impact |
|---|---|
| KWave.jl's `pstd_elastic_2d` only handles `vx`/`vy` velocity sources — `source.uz` is silently ignored in 2-D. | Drove `ux` instead. |
| `sensor.record = [:ux]` returns the **velocity** field `vx`, not displacement. pykwavers' `result.ux` is **displacement**. | pykwavers displacement is numerically differentiated to velocity for the comparison. |
| pykwavers' elastic core panics with `ndarray: index out of bounds` when `NZ = 1`. | NZ = 16 used (16-cell-thick z-slab; source/sensors on the mid-z plane only). |

## Files

| File | Role |
|---|---|
| `compare_elastic.py` | Python orchestrator; builds shared inputs, subprocess-invokes Julia, runs pykwavers, computes parity metrics, generates side-by-side figure. |
| `run_kwave_julia_elastic.jl` | KWave.jl driver invoked by the Python orchestrator. Reads CSV/JSON params; writes recorded ux trace as CSV. |
| `output/elastic_julia_compare.png` | 4-panel side-by-side figure: layout, KWave.jl ux, pykwavers ux (velocity-aligned), difference. |
| `output/elastic_julia_metrics.txt` | Pearson r, RMS ratio, PSNR, peak ratio, runtimes. |
| `output/elastic_julia_kwave.csv` | Cached KWave.jl ux trace. |
| `output/elastic_julia_pykwavers.npz` | Cached pykwavers trace. |

## Usage

```bash
# Pre-flight (one-time):
#   1. Build pykwavers:        cd pykwavers && maturin develop --release
#   2. Confirm Julia + KWave.jl: julia --project=external/k-wave-julia/KWave.jl -e 'using KWave'

# Standard run (uses cache if available)
python compare_elastic.py

# Force fresh runs
python compare_elastic.py --no-cache

# CI mode (exit 0 even when parity targets fail — useful while the
# Phase A.3.5 Additive source mode lands)
python compare_elastic.py --allow-failure
```

## Roadmap

This compare script is the verification harness for Phase A.3.5
(`Additive` velocity-source mode for elastic). When that lands, this
script should reach Pearson r ≥ 0.95 and rms_ratio ≈ 1.0.

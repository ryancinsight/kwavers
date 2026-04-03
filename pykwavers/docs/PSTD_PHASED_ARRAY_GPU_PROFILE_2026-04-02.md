# PSTD Phased-Array GPU Profile

## Scope

This note records the measured performance state of the `pykwavers` GPU PSTD
phased-array parity path after the first cache-structure remediation pass on
April 2, 2026.

It is a validation artifact, not a design placeholder. The goal is to keep the
current performance facts, invariants, and next optimization targets inside the
repository rather than only in ephemeral terminal output.

## Validation Scenario

- Example: `pykwavers/examples/us_bmode_phased_array_compare.py`
- Mode: quick steering-angle sweep
- Steering angles: 9 / 33
- Reference: cached `k-wave-python` GPU run
- Candidate: `pykwavers` GPU run using `GpuPstdSession`
- Grid:
  - full grid with PML: `256 x 256 x 128`
  - phased array: 64 elements, element length 40 voxels
- Physics:
  - `c0 = 1540 m/s`
  - `rho0 = 1000 kg/m^3`
  - `alpha_coeff = 0.75 dB / (MHz^y cm)`
  - `alpha_power = 1.5`
  - `BonA = 6.0`

## Measured Result

From [us_bmode_phased_array_metrics.txt](/d:/kwavers/pykwavers/examples/output/us_bmode_phased_array_metrics.txt):

- `kwave_runtime_s = 234.573`
- `pykwavers_runtime_s = 443.741`

Per-angle GPU timing summary from `GpuPstdSession.last_run_profile` aggregated by
the phased-array comparison script:

- `total`: mean `47652.829 ms`, max `48228.952 ms`, min `46946.536 ms`
- `solver_run`: mean `47618.417 ms`, max `48195.351 ms`, min `46905.963 ms`
- `materialize`: mean `34.410 ms`, max `47.683 ms`, min `23.228 ms`
- `medium_upload`: mean `0.000 ms`, max `0.000 ms`, min `0.000 ms`

## Interpretation

The current cached phased-array path is no longer dominated by Python overhead.

Measured consequence:

- `materialize_ns / total_ns << 1`
- `medium_upload_ns = 0` in the cached run path
- `solver_run_ns ≈ total_ns`

Therefore the dominant cost is inside the GPU PSTD time loop itself rather than:

- source/sensor mask rebuilding
- medium buffer re-upload
- NumPy materialization
- Python-level scan-line orchestration

## Applied Remediations Before This Profile

The following changes were already applied before this measurement:

- persistent `GpuPstdSession` in `pykwavers`
- cached source/sensor mask via `set_source_sensor_mask(...)`
- cached-medium execution via `run_scan_line_cached()`
- phased-array harness split between fixed mask construction and per-angle
  velocity-signal generation
- host-side packed source buffer reuse in the GPU PSTD run path

These changes reduced orchestration overhead, but they did not remove the main
kernel-level bottleneck.

## Numerical State

Current quick parity metrics:

- Fundamental:
  - `pearson_r = 0.990278`
  - `rms_ratio = 1.442263`
  - `psnr_db = 25.771`
- Harmonic:
  - `pearson_r = 0.973833`
  - `rms_ratio = 2.111152`
  - `psnr_db = 24.367`

This means:

- structural similarity is relatively strong
- amplitude parity is still materially incorrect
- performance and correctness remain coupled concerns in this path

## Scientific Constraint

For a k-space pseudospectral method following Treeby and Cox, the spatial
derivative path is expected to be spectrally accurate for resolvable modes, so a
large persistent phased-array amplitude mismatch is more likely to arise from:

- source injection normalization
- staggered-grid shift/operator mismatch
- pressure-density constitutive update scaling
- attenuation/nonlinearity discretization mismatch
- transducer ordering or beamforming-delay semantics

and not from Python container overhead.

## Next Targets

The next optimization and correction passes should focus on the GPU solver
interior, in this order:

1. Add internal PSTD batch-loop timing around:
   - FFT / IFFT path
   - `pipeline_kspace_shift`
   - `pipeline_vel_update`
   - `pipeline_dens_update`
   - `pipeline_pres_density`
   - sensor recording
2. Validate source-injection scaling against the CPU PSTD reference for a single
   steering angle and homogeneous medium.
3. Compare staggered-grid phase shifts against the k-Wave reference conventions
   used in the phased-array examples.
4. Re-measure quick phased-array runtime and parity after each kernel change.

## References

1. Treeby, B. E., & Cox, B. T. (2010). k-Wave: MATLAB toolbox for the simulation
   and reconstruction of photoacoustic wave fields. *Journal of Biomedical
   Optics*, 15(2), 021314.
2. `k-wave-python` documentation, grid and transducer example behavior:
   https://k-wave-python.readthedocs.io/

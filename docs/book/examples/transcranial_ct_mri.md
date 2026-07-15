# Example: Transcranial CT/MRI Reconstruction

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --release --example transcranial_ct_mri_reconstruction --features nifti`  
**Source**: [`crates/kwavers/examples/transcranial_ct_mri_reconstruction.rs`](../../../../crates/kwavers/examples/transcranial_ct_mri_reconstruction.rs)

## What This Example Demonstrates

This example performs masked full-wave inversion on a real CT-derived head model and validates the recovered brain sound-speed image against co-registered MRI. The skull remains fixed to the CT acoustic model while the brain is reconstructed from full-wave transmission data.

| Component | API | Value |
|---|---|---|
| Reconstruction grid | `GRID`, `PAD`, `N_SRC`, `ITERS` | Uses a 40×40 slice, 16 ring sources, and 6 masked-FWI iterations |
| Acoustic drive | `F0_HZ`, `P0_PA` | Runs the inversion at 150 kHz with 0.1 MPa source amplitude |
| Dataset | `data/cfb_gbm_sample/{ct,t1}.nii.gz` | Requires the bundled real CT/T1 pair and the `nifti` feature |

## Key Code Snippet

```rust
const GRID: usize = 40; // CT head slice resampled to GRID×GRID
const PAD: usize = 8; // water-bath border (CPML-absorbed); sources sit here
const N_SRC: usize = 16; // ring transmit/receive elements (coverage: N·(N−1) data)
const ITERS: usize = 6; // masked-FWI iterations
const F0_HZ: f64 = 150_000.0;
const P0_PA: f64 = 1.0e5;
const C_WATER: f64 = 1500.0;
const RHO_WATER: f64 = 1000.0;
const C_BRAIN_FLAT: f64 = 1540.0; // homogeneous brain start
const C_BRAIN_MIN: f64 = 1480.0;
```

## Expected Output (if applicable)

If NIfTI support or sample data is missing, the example prints instructions; otherwise it reports reconstruction/validation progress and MRI correlation metrics.

## Book Chapter

[← Transcranial HIFU and BBB Treatment Planning](../hifu_transcranial_ablation.md)

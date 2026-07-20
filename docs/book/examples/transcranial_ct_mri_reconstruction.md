# Example: Transcranial CT/MRI Reconstruction

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example transcranial_ct_mri_reconstruction --features ritk`  
**Source**: [`crates/kwavers/examples/transcranial_ct_mri_reconstruction.rs`](../../../crates/kwavers/examples/transcranial_ct_mri_reconstruction.rs)

## What This Example Demonstrates

This example demonstrates transcranial CT/MRI reconstruction for focused ultrasound imaging. It loads co-registered NIfTI volumes through RITK, derives a CT acoustic model, reconstructs brain sound speed with masked FWI, and validates the result against MRI.

## Pipeline

```text
CT/MRI NIfTI  →  RITK Image  →  CT acoustic slice  →  masked FWI  →  MRI validation
```

## Features

| Feature | Description |
|---------|-------------|
| NIfTI Loading | Co-registered CT and T1 MRI through RITK |
| Image Processing | Head-slice selection and grid resampling |
| Model Generation | CT-derived sound speed, density, and brain mask |
| Reconstruction | Skull-frozen brain FWI with MRI correlation |

## Key Code Snippet

```rust
let ct_path = std::env::var("KWAVERS_CT_PATH")
    .unwrap_or_else(|_| "data/cfb_gbm_sample/ct.nii.gz".to_owned());
let Some(ct) = load_nifti(&ct_path) else {
    return Ok(());
};

let slice_index = select_head_slice(&ct.data)?;
let ct_slice = resample_head_slice(&ct.data, ct.spacing_mm, slice_index, GRID)?;
let acoustic = AcousticSlice::from_ct_hu(ct_slice.hu.clone(), ct_slice.spacing_m)?;
let grid = Grid::new(
    GRID + 2 * PAD,
    GRID + 2 * PAD,
    2,
    ct_slice.spacing_m,
    ct_slice.spacing_m,
    ct_slice.spacing_m,
)?;
```

## Book Chapter

[← Transcranial Ultrasound: Physics, Aberration Correction, and Therapy](../transcranial_ultrasound.md)

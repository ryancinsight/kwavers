# Example: Transcranial CT/MRI Reconstruction

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example transcranial_ct_mri_reconstruction --features ritk`  
**Source**: [`crates/kwavers/examples/transcranial_ct_mri_reconstruction.rs`](../../../../crates/kwavers/examples/transcranial_ct_mri_reconstruction.rs)

## What This Example Demonstrates

This example demonstrates transcranial CT/MRI reconstruction for focused ultrasound therapy planning. It uses RITK DICOM loading to create patient-specific models from medical imaging data.

## Pipeline

```text
DICOM Series  →  RITK Image  →  CT/MRI Processing  →  3D Model  →  Therapy Planning
```

## Features

| Feature | Description |
|---------|-------------|
| DICOM Loading | Multi-slice CT/MRI series |
| Image Processing | Slice extraction and 3D reconstruction |
| Model Generation | Patient-specific skull and brain model |
| Therapy Planning | Target localization and treatment planning |

## Key Code Snippet

```rust
// Load DICOM series
let dicom_series = load_dicom_series("patient_001")?;

// Create RITK image
let image = NiftiReader::read("skull.nii.gz")?;

// Generate 3D model
let model = create_therapy_model(&image)?;

// Plan therapy
therapy_planner.plan(&model)?;
```

## Book Chapter

[← Transcranial Ultrasound: Physics, Aberration Correction, and Therapy](../transcranial_ultrasound.md)

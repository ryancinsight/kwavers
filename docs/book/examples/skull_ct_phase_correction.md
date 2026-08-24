# Example: Skull CT Phase Correction

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example skull_ct_phase_correction -- <dicom_dir> <output.ppm> [series_instance_uid]`\
**Source**: [`crates/kwavers/examples/skull_ct_phase_correction.rs`](../../../crates/kwavers/examples/skull_ct_phase_correction.rs)

## What This Example Demonstrates

This example loads a skull CT DICOM series, constructs the validated,
Aequitas-typed `AcousticSkullProperties` configuration SSOT, converts
Hounsfield units through `HeterogeneousSkull::from_ct_hill`, and computes
thin-phase-screen corrections for a hemispherical focused bowl. It also exports
per-element correction data and visualization panels.

| Component | API | Value |
|---|---|---|
| Input loader | `load_native_dicom_series` | Reads the DICOM directory, with an optional series UID selector |
| Therapy array | `TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT` | Samples corrections for a 1024-element hemispherical bowl |
| Operating point | `FREQUENCY_HZ` | Computes phase correction at 650 kHz through the CT-derived skull map |

## Key Code Snippet

```rust,ignore
const FREQUENCY_HZ: f64 = 650_000.0;
const TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT: usize = 1024;
const TRANSCRANIAL_FOCUSED_BOWL_RADIUS_M: f64 = 0.150;
const UNIT_FREQUENCY_HZ: f64 = 1.0;
const UNIT_AMPLITUDE_PA: f64 = 1.0;
const C_WATER_M_PER_S: f64 = 1482.0;
const C_CORTICAL_BONE_M_PER_S: f64 = 2800.0;
const RHO_CORTICAL_BONE_KG_PER_M3: f64 = 1900.0;
const HU_BONE_LOWER: f64 = 300.0;

let bone = AcousticSkullProperties::new(
    Velocity::from_base(C_CORTICAL_BONE_M_PER_S),
    MassDensity::from_base(RHO_CORTICAL_BONE_KG_PER_M3),
    ReciprocalLength::from_base(20.0),
    Length::from_base(0.007),
    None,
)?;
let skull = HeterogeneousSkull::from_ct_hill(&ct.hu, &bone)?;
```

## Expected Output (if applicable)

A successful run writes the requested PPM plus companion CSV/SVG/PPM artifacts describing phase corrections and element-level skull metrics.

## Book Chapter

[← Transcranial Ultrasound: Physics, Aberration Correction, and Therapy](../transcranial_ultrasound.md)

# Example: Phantom Builder Demo

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example phantom_builder_demo`  
**Source**: [`crates/kwavers/examples/phantom_builder_demo.rs`](../../../../crates/kwavers/examples/phantom_builder_demo.rs)

## What This Example Demonstrates

This example is a guided tour of the phantom-construction APIs used by optical and photoacoustic workflows. It demonstrates blood oxygenation targets, layered tissues, tumors, vascular networks, custom region models, and predefined clinical presets.

| Component | API | Value |
|---|---|---|
| Builder API | `PhantomBuilder` | Creates phantoms procedurally for multiple imaging tasks |
| Sample dimensions | `GridDimensions::new` | Uses small 3-D volumes such as 40×40×40 at 1 mm spacing for demo phantoms |
| Map analysis | `OpticalPropertyMapBuilder` / `OpticalPropertyMapAnalysis` | Converts phantom definitions into inspectable optical-property maps |

## Key Code Snippet

```rust
let dims = GridDimensions::new(40, 40, 40, 0.001, 0.001, 0.001);

// Build phantom with arterial/venous vessels and hypoxic tumor
let phantom = PhantomBuilder::blood_oxygenation()
    .dimensions(dims)
    .wavelength(800.0)
    .background(OpticalPropertyData::soft_tissue())
    // Arterial vessel: high oxygenation (sO₂ = 98%)
    .add_artery([0.015, 0.020, 0.020], 0.002, 0.98)
    // Venous vessel: lower oxygenation (sO₂ = 65%)
```

## Expected Output (if applicable)

The executable prints one section per phantom family and reports summary statistics for the generated property maps.

## Book Chapter

[← Sources and Transducers](../sources_and_transducers.md)

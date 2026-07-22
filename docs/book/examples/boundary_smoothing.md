# Example: Boundary Smoothing

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example boundary_smoothing`  
**Source**: [`crates/kwavers/examples/boundary_smoothing.rs`](../../../crates/kwavers/examples/boundary_smoothing.rs)

## Overview

Demonstrates staircase boundary smoothing techniques to reduce grid artifacts at curved boundaries in ultrasound simulations.

## Techniques shown

- **Staircase boundaries**: unsmoothed reference with visible aliasing artifacts
- **Subpixel smoothing**: trilinear interpolation of boundary crossing fractions
- **Normal vector smoothing**: use surface normals to weight boundary cells

Compares transmitted pressure fields through a curved obstacle for each method, quantifying the reduction in spurious reflections.

## Part Reference

Part I — Sources and Transducers

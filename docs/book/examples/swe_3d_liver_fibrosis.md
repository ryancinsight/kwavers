# Example: 3D Shear Wave Elastography Liver Fibrosis

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example swe_3d_liver_fibrosis`  
**Source**: [`crates/kwavers/examples/swe_3d_liver_fibrosis.rs`](../../../crates/kwavers/examples/swe_3d_liver_fibrosis.rs)

## Overview

Complete clinical workflow for 3D SWE assessment of liver fibrosis using multi-directional shear waves and volumetric analysis.

## Clinical Workflow

1. Generate synthetic liver phantom with spatially-varying stiffness (fibrosis grades F0–F4)
2. Simulate multi-directional shear wave propagation (lateral + axial)
3. Extract displacement fields via cross-correlation tracking
4. Estimate 3D shear modulus map via local frequency estimation
5. Classify fibrosis grade using METAVIR-calibrated thresholds
6. Output volumetric stiffness map and per-ROI statistics

## Fibrosis Grades

| Grade | Stiffness | Clinical significance |
|---|---|---|
| F0-F1 | 2–4 kPa | Normal to mild |
| F2 | 5–7 kPa | Moderate |
| F3-F4 | 8–12 kPa | Severe / cirrhosis |

## Part Reference

Part II — Elastography

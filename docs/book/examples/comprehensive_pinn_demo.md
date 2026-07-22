# Example: Comprehensive PINN Demo

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example comprehensive_pinn_demo --features pinn`  
**Source**: [`crates/kwavers/examples/comprehensive_pinn_demo.rs`](../../../crates/kwavers/examples/comprehensive_pinn_demo.rs)

## Overview

Showcases the complete Physics-Informed Neural Network (PINN) ecosystem: basic training → advanced physics domains → meta-learning → uncertainty quantification → cloud deployment.

## Capabilities demonstrated

1. **Wave equation PINN** — 1D and 2D acoustic wave equation training
2. **Tissue heterogeneity** — spatially varying speed of sound
3. **Inverse problem** — parameter estimation from synthetic measurements
4. **Uncertainty quantification** — dropout and ensemble variance
5. **Transfer learning** — pre-trained → fine-tune on new geometry
6. **Deployment** — JIT compilation for real-time inference

## Part Reference

Part IV — Inverse Problems and Physics-Informed Neural Networks

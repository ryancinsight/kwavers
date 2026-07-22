# Example: PINN Meta-Learning and Uncertainty Quantification

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example pinn_meta_uncertainty --features pinn`  
**Source**: [`crates/kwavers/examples/pinn_meta_uncertainty.rs`](../../../crates/kwavers/examples/pinn_meta_uncertainty.rs)

## Overview

Demonstrates advanced PINN capabilities: meta-learning for rapid adaptation to new physics regimes, and uncertainty quantification for reliability estimation.

## Meta-Learning

MAML-style meta-learning pre-trains on a distribution of acoustic problems so the network adapts to new tissue properties in a few gradient steps — useful for patient-specific real-time inference.

## Uncertainty Quantification

Monte Carlo dropout and ensemble methods estimate prediction uncertainty. Physics residual variance provides a second uncertainty signal orthogonal to epistemic uncertainty.

## Part Reference

Part IV — Inverse Problems and Physics-Informed Neural Networks

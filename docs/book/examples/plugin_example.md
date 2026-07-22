# Example: Plugin Architecture

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example plugin_example`  
**Source**: [`crates/kwavers/examples/plugin_example.rs`](../../../crates/kwavers/examples/plugin_example.rs)

## Overview

Demonstrates the extensible plugin architecture for composing custom physics into a kwavers simulation without modifying core crates.

## Patterns shown

1. **Implement a custom physics plugin** — implement the `PhysicsPlugin` trait
2. **Register with `PluginManager`** — zero-overhead plugin composition
3. **Adapt existing components** — wrap a `Solver` as a plugin with `SolverPlugin`

## Why plugins

The plugin pattern separates the simulation *orchestration* (when to call what) from physics *implementations* (what each module computes), following the Dependency Inversion Principle.

## Part Reference

Part IV — Simulation Orchestration: The Capability Catalog

//! GPU compute shaders for physics simulations.
//!
//! # Shader layout
//!
//! Each physical kernel has a canonical `.wgsl` source file (loaded via
//! `include_str!` at compile time by the respective dispatch modules):
//!
//! | File                       | Used by                             |
//! |----------------------------|-------------------------------------|
//! | `absorption.wgsl`          | `gpu::kernels` (acoustic decay)     |
//! | `fdtd.wgsl`                | `gpu::fdtd` (two-pass leapfrog)     |
//! | `fdtd_pressure.wgsl`       | `gpu::compute::fdtd_gpu`            |
//! | `kspace_propagate.wgsl`    | `gpu::kernels`, `gpu::kspace`       |
//! | `nonlinear.wgsl`           | `gpu::kernels` (Westervelt)         |
//! | `pstd.wgsl`                | `gpu::kspace` (PSTD propagation)    |
//!
//! The `neural_network` sub-module is the only exception: it uses an inline
//! WGSL string constant so that the shader remains co-located with the Rust
//! binding types it must match.

pub mod neural_network;

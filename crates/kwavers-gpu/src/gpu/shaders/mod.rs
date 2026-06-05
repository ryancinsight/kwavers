//! GPU compute shaders for physics simulations.
//!
//! # Shader layout
//!
//! Each physical kernel has a canonical `.wgsl` source file (loaded via
//! `include_str!` at compile time by its dispatch module). Only the shaders
//! actually wired into a pipeline are kept here:
//!
//! | File                       | Loaded by                                   |
//! |----------------------------|---------------------------------------------|
//! | `fdtd.wgsl`                | `gpu::fdtd` (two-pass leapfrog)             |
//! | `fdtd_pressure.wgsl`       | `gpu::compute::fdtd_gpu`                    |
//! | `acoustic_field.wgsl`      | `gpu::compute_kernels::acoustic_field`      |
//!
//! The active PSTD propagation shader lives with its solver at
//! `crate::pstd_gpu::shaders::pstd.wgsl` (loaded by
//! `pstd_gpu::pipeline::construction::solver`), not in this directory. The
//! `neural_network` sub-module is the other exception: it uses an inline WGSL
//! string constant so the shader stays co-located with the Rust binding types
//! it must match.
//!
//! Obsolete standalone shaders (`absorption`, `pml`, `kspace_propagate`,
//! `kspace_shift`, `nonlinear`, `chirp`, `electromagnetic`) were removed: their
//! functionality is either superseded by entry points inside `pstd.wgsl`
//! (`kspace_shift_apply`, the `precomp_bon_a` nonlinear term, Treeby–Cox
//! absorption, precomputed PML decay) or had no remaining dispatch site.

pub mod neural_network;

//! Field-surrogate utilities — cached PSTD focal kernels for fast
//! treatment-planning queries.
//!
//! Histotripsy / HIFU treatment planners run thousands of focal-spot
//! evaluations per scan. Calling a full PSTD wave solver per shot is
//! ~minutes of compute; instead we precompute a small set of focal
//! kernels (one per `(f0, pnp)` scenario) on a homogeneous medium,
//! cache them, and serve per-shot queries by:
//!
//!   1. resampling the cached kernel onto the planner grid,
//!   2. embedding it at the planner's focal voxel,
//!   3. blending across `(f0, pnp)` axes when a query lies between
//!      cached corners,
//!   4. normalizing so the focal voxel reads 1 — the planner then
//!      multiplies by its own `source_pa` and tissue absorption.
//!
//! This is the kernel cache; the per-voxel pressure-statistics
//! accumulator lives in
//! [`kwavers_domain::sensor::recorder::pressure_statistics`] and feeds
//! kernel construction.
//!
//! ## Module layout
//!
//! | submodule    | responsibility                                  |
//! |--------------|-------------------------------------------------|
//! | `kernel`     | `FocalKernel` struct + metadata accessors       |
//! | `resample`   | trilinear 3-D resampling (no new deps)          |
//! | `placement`  | embed kernel into target grid centred on focus  |
//! | `cube`       | `KernelCube` — `(f0, pnp)` interpolator         |
//!
//! ## Physics
//!
//! In water (B/A ≈ 0) the focal pressure scales linearly with source
//! amplitude, so the `pnp` dimension of the kernel cube is degenerate —
//! the *shape* is `pnp`-independent and `kernel_focal_envelope`
//! normalizes by the global max anyway. The `pnp` parameter is
//! retained on the cube API for symmetry with `f0` but does not
//! drive shape selection.
//!
//! The `f0` dimension is real: focal-spot dimensions scale with
//! wavelength (FWHM_lat = 1.41·λ·F#, FWHM_ax = 7·λ·F#² per Penttinen
//! 1976), so the cube linearly blends the two nearest `f0` corners
//! and re-normalizes after the blend.

pub mod cube;
pub mod helmholtz;
pub mod kernel;
pub mod npz_loader;
pub mod placement;
pub mod resample;

#[cfg(test)]
mod tests;

pub use cube::KernelCube;
pub use helmholtz::{
    helmholtz_residual_field, helmholtz_residual_kernel, helmholtz_residual_stats,
    HelmholtzResidualStats, HELMHOLTZ_C0_WATER,
};
pub use kernel::FocalKernel;
pub use npz_loader::{discover_focal_kernels, load_focal_kernel};
pub use placement::place_kernel_at_focus;
pub use resample::resample_trilinear;

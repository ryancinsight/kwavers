//! `GpuPstdSolver` constructor and GPU pipeline initialisation.
//!
//! SRP split:
//! - `bgl`         — bind group layout builders (one per layout)
//! - `bind_groups` — bind group assembly from already-created buffers
//! - `construction`— `new()`: buffer allocation, shader, pipelines, Ok(Self)
//! - `auto_device` — `with_auto_device()`: adapter selection + delegates to `new()`

mod auto_device;
mod bgl;
mod bind_groups;
mod buffers;
mod construction;
mod pipelines;

/// Medium acoustic property arrays (input slices for `GpuPstdSolver::new`).
pub struct MediumArrays<'a> {
    /// Sound speed field `c₀` (m/s), flattened row-major order.
    pub c0_flat: &'a [f32],
    /// Ambient density `ρ₀` (kg/m³), flattened row-major order.
    pub rho0_flat: &'a [f32],
}

/// Scalar solver configuration for `GpuPstdSolver::new`.
#[derive(Copy, Clone)]
pub struct SolverParams {
    /// Time step `Δt` (s).
    pub dt: f64,
    /// Total number of time steps.
    pub nt: usize,
    /// Reference sound speed used for k-space shift operators (m/s).
    pub c_ref: f64,
    /// Enable nonlinear (B/A) pressure term.
    pub nonlinear: bool,
    /// Enable power-law absorption correction.
    pub absorbing: bool,
}

/// PML boundary layer arrays for `GpuPstdSolver::new`.
pub struct PmlArrays<'a> {
    /// PML attenuation along x.
    pub x: &'a [f32],
    /// PML attenuation along y.
    pub y: &'a [f32],
    /// PML attenuation along z.
    pub z: &'a [f32],
    /// PML staggered-grid x (σ).
    pub sgx: &'a [f32],
    /// PML staggered-grid y (σ).
    pub sgy: &'a [f32],
    /// PML staggered-grid z (σ).
    pub sgz: &'a [f32],
}

/// Power-law absorption operator arrays for `GpuPstdSolver::new`.
pub struct AbsorptionArrays<'a> {
    /// B/A nonlinearity parameter field, flattened.
    pub bon_a_flat: &'a [f32],
    /// Fractional-Laplacian absorption operator ∇^(y+1)  (∇₁).
    pub nabla1: &'a [f32],
    /// Fractional-Laplacian absorption operator ∇^(y-1) (∇₂).
    pub nabla2: &'a [f32],
    /// Dispersion absorption coefficient τ.
    pub tau: &'a [f32],
    /// Absorption coefficient η.
    pub eta: &'a [f32],
}

pub use bgl::{PstdBindGroupLayoutProvider, WgpuPstdBindGroupLayoutFactory};
pub use bind_groups::{PstdBindGroupProvider, WgpuPstdBindGroupFactory};
pub use buffers::{PstdBufferProvider, WgpuPstdBufferFactory};
pub use pipelines::{PstdPipelineProvider, WgpuPstdPipelineFactory};

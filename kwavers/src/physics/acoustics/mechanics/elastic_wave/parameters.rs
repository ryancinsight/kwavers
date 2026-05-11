//! Parameter structures for elastic wave updates
//!
//! This module defines parameter containers that reduce coupling between
//! components following SOLID principles.

use ndarray::{Array3, ArrayView3};

/// Re-exported from [`super::fields`] — single canonical definition.
pub use super::fields::Complex3D;

/// Parameters for the spectral elastic stress update.
///
/// # Update law
///
/// Each call writes
///
/// ```text
/// σ̃(t+dt) = σ̃(t) + dt · Cᵢⱼₖₗ · ε̃ₖₗ(t+dt/2)
/// ```
///
/// into the output buffer. The current spectral stress (`txx_fft` …
/// `tyz_fft`) is required so the kernel can ADD the per-step increment to
/// persistent stress, which is the correct behaviour for elastic
/// propagation (μ ≠ 0). Passing zero-initialised stress fields recovers
/// the legacy "single-step from rest" behaviour the old `update_wave`
/// stepper relied on for its acoustic-fluid limit.
///
/// # Spectral derivative operators
///
/// `dkx_op[i] = i · k_x[i] · shift_x(i)` is the per-axis spectral
/// derivative operator the kernel applies to the velocity field. For a
/// **collocated** scheme pass `i · k_x[i]` (no shift). For a
/// **staggered** scheme pass `i · k_x[i] · exp(− i · k_x[i] · Δx / 2)` —
/// matches KWave.jl `pstd_elastic_2d`'s `ddx_k_shift_neg` used during the
/// stress update. Caller owns the choice; the kernel merely consumes the
/// pre-built operator.
///
/// # k-space correction (Tabei et al. 2002)
///
/// `kappa[i,j,k] = sinc(c_ref · dt · |k| / 2)` is the Treeby–Cox spectral
/// correction factor that eliminates temporal dispersion at all CFL values.
/// Use `Array3::ones(shape)` for unit kappa (no correction, preserves the
/// O(dt²) leapfrog accuracy).
#[derive(Debug)]
pub struct StressUpdateParams<'a> {
    pub vx_fft: &'a Complex3D,
    pub vy_fft: &'a Complex3D,
    pub vz_fft: &'a Complex3D,
    /// Current normal-stress spectrum, persisted from the previous step.
    pub txx_fft: &'a Complex3D,
    pub tyy_fft: &'a Complex3D,
    pub tzz_fft: &'a Complex3D,
    /// Current shear-stress spectrum, persisted from the previous step.
    pub txy_fft: &'a Complex3D,
    pub txz_fft: &'a Complex3D,
    pub tyz_fft: &'a Complex3D,
    /// Per-axis complex derivative operator: `i · kα · shift`.
    /// See type-level docs.
    pub dkx_op: &'a Complex3D,
    pub dky_op: &'a Complex3D,
    pub dkz_op: &'a Complex3D,
    pub lame_lambda: &'a Array3<f64>,
    pub lame_mu: &'a Array3<f64>,
    pub density: ArrayView3<'a, f64>,
    pub dt: f64,
    /// k-space correction factor `sinc(c_ref·dt·|k|/2)` (Tabei et al. 2002).
    /// Pass `Array3::ones(shape)` for no correction (unit kappa).
    pub kappa: &'a Array3<f64>,
}

/// Parameters for the spectral elastic velocity update.
///
/// `dkx_op` … `dkz_op` are the per-axis complex derivative operators
/// applied to the stress field. For a **collocated** scheme pass
/// `i · k_α[i]`. For a **staggered** scheme pass
/// `i · k_α[i] · exp(+ i · k_α[i] · Δα / 2)` — matches KWave.jl
/// `pstd_elastic_2d`'s `ddx_k_shift_pos` used during the velocity update.
#[derive(Debug)]
pub struct VelocityUpdateParams<'a> {
    pub vx_fft: &'a Complex3D,
    pub vy_fft: &'a Complex3D,
    pub vz_fft: &'a Complex3D,
    pub txx_fft: &'a Complex3D,
    pub tyy_fft: &'a Complex3D,
    pub tzz_fft: &'a Complex3D,
    pub txy_fft: &'a Complex3D,
    pub txz_fft: &'a Complex3D,
    pub tyz_fft: &'a Complex3D,
    /// Per-axis complex derivative operator: `i · kα · shift`.
    pub dkx_op: &'a Complex3D,
    pub dky_op: &'a Complex3D,
    pub dkz_op: &'a Complex3D,
    pub density: ArrayView3<'a, f64>,
    pub dt: f64,
    /// k-space correction factor `sinc(c_ref·dt·|k|/2)` (Tabei et al. 2002).
    /// Pass `Array3::ones(shape)` for no correction (unit kappa).
    pub kappa: &'a Array3<f64>,
}

//! Per-step compute-pass encoding: velocity, source, density, pressure, record.
//!
//! Each `encode_*` method appends GPU dispatches to an already-open `ComputePass`.
//! The caller creates one pass per time step and drops it after all phases complete,
//! keeping all dispatches inside one uninterrupted compute pass (no UAV barriers).

mod density;
mod pressure;
mod source;
mod velocity;

use super::super::{GpuPstdSolver, PstdParams};

/// Scalar constants and workgroup sizes extracted from `GpuPstdSolver` once per
/// batch and threaded to all per-step encoders.
pub(super) struct StepCtx {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub dt: f32,
    pub n_sensors: u32,
    pub nt: u32,
    pub nonlinear: u32,
    pub absorbing: u32,
    pub n_src: usize,
    pub n_vel_x: usize,
    pub elem_wg: u32,
}

impl StepCtx {
    /// Build a `PstdParams` with `n_fft/n_batches/log2n = 0` (physics dispatch).
    pub(super) fn params(&self, step: u32, axis: u32) -> PstdParams {
        PstdParams {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            axis,
            n_fft: 0,
            n_batches: 0,
            log2n: 0,
            inverse: 0,
            step,
            dt: self.dt,
            n_sensors: self.n_sensors,
            nt: self.nt,
            nonlinear: self.nonlinear,
            absorbing: self.absorbing,
        }
    }

    #[inline]
    pub(super) fn ceil_div(n: usize, d: usize) -> u32 {
        n.div_ceil(d) as u32
    }
}

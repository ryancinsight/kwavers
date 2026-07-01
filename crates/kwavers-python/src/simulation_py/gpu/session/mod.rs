#[cfg(feature = "gpu")]
mod absorption;
mod construction;
mod control;
#[cfg(feature = "gpu")]
mod pml;
mod run;
mod source;

use pyo3::prelude::*;

/// Persistent GPU PSTD session for efficient B-mode scan-line loops.
///
/// Creating a new `GpuPstdSolver` per scan line is expensive (~500 ms) because
/// wgpu must compile ~13 WGSL compute pipelines from scratch.  `GpuPstdSession`
/// creates the solver **once** and re-uses compiled pipelines.  Between scan
/// lines you only re-upload the medium arrays via `run_scan_line()`.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
#[pyclass(unsendable)]
pub struct GpuPstdSession {
    #[cfg(feature = "gpu")]
    pub(crate) solver: kwavers_gpu::pstd_gpu::GpuPstdSolver,

    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) nz: usize,

    pub(crate) time_steps: usize,

    pub(crate) sensor_indices: Vec<u32>,
    pub(crate) vel_x_indices: Vec<u32>,
    pub(crate) vel_x_signals: Vec<f32>,
    pub(crate) last_medium_upload_ns: u64,
    pub(crate) last_medium_variable_upload_ns: u64,
    pub(crate) last_medium_static_upload_ns: u64,
    pub(crate) last_solver_run_ns: u64,
    pub(crate) last_materialize_ns: u64,
    pub(crate) last_total_ns: u64,
}

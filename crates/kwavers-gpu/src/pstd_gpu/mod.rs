//! GPU-resident PSTD (Pseudospectral Time Domain) acoustic solver.
//!
//! # Design
//!
//! All acoustic fields (p, ux, uy, uz, rhox, rhoy, rhoz) remain on the GPU
//! throughout the simulation. The caller selects sensor-only output or an
//! explicit final-state readback at the end of the run.
//!
//! # Bind group layout (up to eight storage buffers per group)
//!
//! - group(0) 8 storage: p, ux, uy, uz, rhox, rhoy, rhoz, source_kappa
//! - immediate data: `PstdParams`
//! - group(1) 7 storage: kspace_re, kspace_im, kappa, rho0_inv, c0_sq,
//!   rho0, bon_a
//! - group(2) 8 storage: pml_sgx, pml_sgy, pml_sgz, pml_xyz (packed),
//!   shifts_all (packed), sensor_flat_indices, sensor_data,
//!   source_data (packed)
//!
//! Lossless PSTD compiles only these three storage bind groups, requiring 23
//! storage buffers per compute-shader stage. Fractional-Laplacian absorption
//! compiles its additional eight-buffer group only when enabled and therefore
//! requires a device exposing 31 storage buffers per compute-shader stage.
//!
//! # Packed buffer formats
//!
//! **pml_xyz**: three concatenated f32 arrays `[pml_x | pml_y | pml_z]`,
//! each of size `nx * ny * nz`. Index via `ax * total + flat_idx`.
//!
//! **shifts_all**: twelve 1D arrays packed in order:
//! `x_pos_re, x_pos_im, x_neg_re, x_neg_im` (each size nx),
//! `y_pos_re, y_pos_im, y_neg_re, y_neg_im` (each size ny),
//! `z_pos_re, z_pos_im, z_neg_re, z_neg_im` (each size nz).
//! Total: `4*(nx+ny+nz)` f32 values.
//!
//! **source_data**: `[bitcast<f32>(mask_indices[n_src]) | signals[n_src*nt]]`.
//! Mask indices are stored as bit-cast f32 values of u32 flat indices.
//!
//! # Module structure
//!
//! | Submodule        | Responsibility                                              |
//! |------------------|-------------------------------------------------------------|
//! | `pipeline`       | `new()` constructor, buffer allocation, BGL, pipeline compile |
//! | `time_loop`      | `run()` time-marching loop + internal dispatch helpers      |
//! | `medium_update`  | `update_medium_variable()` for scan lines, `update_medium()` for full refresh |
//!
//! # References
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314.
//! - Liu (1998). Microwave Opt. Technol. Lett. 15(3), 158-165.

mod medium;
mod medium_update;
mod pipeline;
mod runner;
mod source;
mod state;
mod time_loop;

use std::marker::PhantomData;

use kwavers_grid::Grid;

pub use medium::PstdMediumSnapshot;
pub use pipeline::{AbsorptionArrays, MediumArrays, PmlArrays, SolverParams};
pub use runner::{
    cpml_thickness_limits, run_gpu_pstd, run_gpu_pstd_with_outputs, run_gpu_pstd_with_provider,
    run_gpu_pstd_with_provider_outputs, run_gpu_pstd_with_snapshot_outputs, GpuPstdRunConfig,
};
pub use source::{prepare_pstd_pressure_source, PstdPressureSourceSchedule};
pub use state::{
    PstdAutoDeviceProvider, PstdFinalFields, PstdMediumUpdateState, PstdOutputRequest,
    PstdRunInputs, PstdRunResult, PstdRunScalars, PstdRunState, PstdStateBuilder,
    PstdStateProvider, WgpuPstdStateProvider,
};

/// Per-run timing profile for GPU PSTD execution.
///
/// Durations are measured on the host and include queue submission / wait costs.
/// They are intended for regression tracking and hotspot attribution, not for
/// cycle-accurate GPU kernel benchmarking.
#[derive(Debug, Clone, Default)]
pub struct GpuPstdRunProfile {
    pub total_ns: u64,
    pub host_pack_ns: u64,
    pub upload_ns: u64,
    pub zero_clear_ns: u64,
    pub encode_submit_ns: u64,
    pub gpu_wait_ns: u64,
    pub sensor_copy_ns: u64,
    pub map_read_ns: u64,
    pub cache_miss: bool,
    pub n_sensors: usize,
    pub n_src: usize,
    pub n_vel_x: usize,
}

// Immediate-data struct; must match `PstdParams` in `pstd.wgsl` exactly.
// 12 x u32/f32 = 48 bytes. `max_immediate_size` must be at least 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PstdParams {
    pub(super) nx: u32,
    pub(super) ny: u32,
    pub(super) nz: u32,
    pub(super) axis: u32,
    pub(super) step: u32,
    pub(super) dt: f32,
    pub(super) n_sensors: u32,
    pub(super) nt: u32,
    pub(super) nonlinear: u32, // 1 = BonA EOS active
    pub(super) absorbing: u32, // 1 = alpha-decay active
    pub(super) peak_offset: u32,
    pub(super) record_peak_pressure: u32,
}

const _: () = assert!(std::mem::size_of::<PstdParams>() == 48);

fn validate_pstd_grid_shape(grid: &Grid) -> Result<usize, String> {
    let shape = [grid.nx, grid.ny, grid.nz];
    for (axis, &length) in ["x", "y", "z"].iter().zip(&shape) {
        if length == 0 {
            return Err(format!("GPU PSTD {axis}-axis length must be positive"));
        }
        u32::try_from(length).map_err(|_| {
            format!("GPU PSTD {axis}-axis length {length} exceeds u32 shader addressing")
        })?;
    }

    let total = grid
        .nx
        .checked_mul(grid.ny)
        .and_then(|xy| xy.checked_mul(grid.nz))
        .ok_or_else(|| {
            format!(
                "GPU PSTD grid shape overflows usize: {}x{}x{}",
                grid.nx, grid.ny, grid.nz
            )
        })?;
    u32::try_from(total).map_err(|_| {
        format!(
            "GPU PSTD grid has {total} points; u32 shader addressing supports at most {}",
            u32::MAX
        )
    })?;
    Ok(total)
}

/// GPU-resident PSTD acoustic solver.
///
/// Keeps all field data on the GPU throughout the time loop. Its
/// [`Self::run`] caller explicitly selects sensor traces, final-state fields,
/// and/or the temporal peak-pressure envelope.
// Provider-owned state keeps GPU handles alive for bind groups that outlive
// construction and are driven by dispatch-only Rust paths.
#[allow(dead_code)]
pub struct GpuPstdSolver<P: PstdStateProvider = WgpuPstdStateProvider> {
    // Note: Manual Debug impl because wgpu types don't implement Debug.
    pub(super) nx: usize,
    pub(super) ny: usize,
    pub(super) nz: usize,
    pub(super) nt: usize,
    pub(super) dt: f64,

    // Provider-owned GPU buffers, pipelines, bind groups, layouts, and caches.
    pub(in crate::pstd_gpu) state: P::State,
    _provider: PhantomData<P>,

    // Physics flags (drive shader branches via push-constant nonlinear/absorbing)
    pub(super) nonlinear: bool,
    pub(super) absorbing: bool,
    // CPU-side medium scratch buffers are preallocated to avoid per-scan-line allocation.
    // update_medium_variable() computes squared sound speed and inverse density here.
    // Sized to nx*ny*nz at construction; never reallocated.
    // Persistent unity staging buffer for disable_source_correction(); avoids a
    // per-call allocation when the caller needs raw additive injection.
    // Packed host-side source upload buffers. The index prefix is stable across
    // cache-hit runs, so only the signal tail is overwritten between scan lines.
    // â”€â”€ Cached run() buffers (reused when sensor/source layout is unchanged) â”€â”€
    // Allocated on first run(); reused on subsequent calls to eliminate per-scan-line
    // VRAM allocation overhead (~500Âµs per allocation on discrete GPUs).
    // Invalidated and reallocated only when n_sensors / n_src / n_vel_x changes.
}

impl<P: PstdStateProvider> std::fmt::Debug for GpuPstdSolver<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPstdSolver")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .field("nt", &self.nt)
            .field("dt", &self.dt)
            .field("nonlinear", &self.nonlinear)
            .field("absorbing", &self.absorbing)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;

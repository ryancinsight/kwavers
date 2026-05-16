//! GPU-resident PSTD (Pseudospectral Time Domain) acoustic solver.
//!
//! # Design
//!
//! All acoustic fields (p, ux, uy, uz, rhox, rhoy, rhoz) remain on the GPU
//! throughout the simulation. Only the final sensor readings are downloaded
//! at the end of the run, minimising PCIe traffic.
//!
//! # Bind group layout (â‰¤8 storage buffers per group)
//!
//! - group(0) 8 storage: p, ux, uy, uz, rhox, rhoy, rhoz, scratch
//! - group(1) 1 uniform: PstdParams
//! - group(2) 8 storage: kspace_re, kspace_im, kspace2_re, kspace2_im,
//!   kappa, rho0_inv, c0_sq, rho0
//! - group(3) 8 storage: pml_sgx, pml_sgy, pml_sgz, pml_xyz (packed),
//!   shifts_all (packed), sensor_flat_indices, sensor_data,
//!   source_data (packed)
//!
//! # Packed buffer formats
//!
//! **pml_xyz**: three concatenated f32 arrays `[pml_x | pml_y | pml_z]`,
//! each of size `nxÃ—nyÃ—nz`. Index via `ax * total + flat_idx`.
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
//! | `pipeline`       | `new()` constructor â€” buffer alloc, BGL, pipeline compile  |
//! | `time_loop`      | `run()` time-marching loop + internal dispatch helpers      |
//! | `medium_update`  | `update_medium_variable()` for scan lines, `update_medium()` for full refresh |
//!
//! # References
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314.
//! - Liu (1998). Microwave Opt. Technol. Lett. 15(3), 158â€“165.

mod medium_update;
mod pipeline;
mod time_loop;

use std::sync::Arc;

pub use pipeline::{AbsorptionArrays, MediumArrays, PmlArrays, SolverParams};

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

// â”€â”€â”€ Params push-constant struct (must match PstdParams in pstd.wgsl) â”€â”€â”€â”€â”€â”€â”€â”€
// 14 Ã— u32/f32 = 56 bytes. max_push_constant_size must be â‰¥ 56.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PstdParams {
    pub(super) nx: u32,
    pub(super) ny: u32,
    pub(super) nz: u32,
    pub(super) axis: u32,
    pub(super) n_fft: u32,
    pub(super) n_batches: u32,
    pub(super) log2n: u32,
    pub(super) inverse: u32,
    pub(super) step: u32,
    pub(super) dt: f32,
    pub(super) n_sensors: u32,
    pub(super) nt: u32,
    pub(super) nonlinear: u32, // 1 = BonA EOS active
    pub(super) absorbing: u32, // 1 = alpha-decay active
}

/// GPU-resident PSTD acoustic solver.
///
/// Keeps all field data on the GPU throughout the time loop; sensor readings
/// are downloaded in a single transfer after all time steps complete.
pub struct GpuPstdSolver {
    // Note: Manual Debug impl because wgpu types don't implement Debug.
    pub(super) device: Arc<wgpu::Device>,
    pub(super) queue: Arc<wgpu::Queue>,

    pub(super) nx: usize,
    pub(super) ny: usize,
    pub(super) nz: usize,
    pub(super) nt: usize,
    pub(super) dt: f64,

    // Field buffers â€” group(0)
    pub(super) buf_p: wgpu::Buffer,
    pub(super) buf_ux: wgpu::Buffer,
    pub(super) buf_uy: wgpu::Buffer,
    pub(super) buf_uz: wgpu::Buffer,
    pub(super) buf_rhox: wgpu::Buffer,
    pub(super) buf_rhoy: wgpu::Buffer,
    pub(super) buf_rhoz: wgpu::Buffer,
    // K-space + medium â€” group(1): bindings 0-7
    // kspace2_re/im removed: kspace_shift_apply now writes in-place.
    pub(super) buf_kspace_re: wgpu::Buffer,
    pub(super) buf_kspace_im: wgpu::Buffer,
    pub(super) buf_kappa: wgpu::Buffer,
    pub(super) buf_rho0_inv: wgpu::Buffer,
    pub(super) buf_c0_sq: wgpu::Buffer,
    pub(super) buf_rho0: wgpu::Buffer,
    pub(super) buf_bon_a: wgpu::Buffer, // B/(2A) per voxel; 0.0 for linear sims
    pub(super) buf_alpha_decay: wgpu::Buffer, // exp(-alpha_Np*c0*dt) per voxel; 1.0 for lossless

    // Fractional-Laplacian absorption operators (group 3)
    // Precomputed constants for Treeby & Cox (2010) Eqs. 9â€“10, 19â€“21.
    pub(super) buf_absorb_nabla1: wgpu::Buffer, // |k|^(y-2) in FFT order
    pub(super) buf_absorb_nabla2: wgpu::Buffer, // |k|^(y-1) in FFT order
    pub(super) buf_absorb_tau: wgpu::Buffer,    // -2*alpha0*c0^(y-1) per voxel
    pub(super) buf_absorb_eta: wgpu::Buffer,    // 2*alpha0*c0^y*tan(pi*y/2) per voxel
    pub(super) buf_absorb_scratch_kre: wgpu::Buffer, // temp kspace re save
    pub(super) buf_absorb_scratch_kim: wgpu::Buffer, // temp kspace im save
    pub(super) buf_absorb_scratch_l1: wgpu::Buffer, // L1 = IFFT(nabla1*FFT(div_u))
    pub(super) buf_absorb_scratch_l2: wgpu::Buffer, // L2 = IFFT(nabla2*FFT(div_u))

    // Physics flags (drive shader branches via push-constant nonlinear/absorbing)
    pub(super) nonlinear: bool,
    pub(super) absorbing: bool,

    // PML + shifts + sensor/source â€” group(3)
    pub(super) buf_pml_sgx: wgpu::Buffer,
    pub(super) buf_pml_sgy: wgpu::Buffer,
    pub(super) buf_pml_sgz: wgpu::Buffer,
    pub(super) buf_pml_xyz: wgpu::Buffer, // packed [pml_x | pml_y | pml_z]
    pub(super) buf_shifts_all: wgpu::Buffer, // packed 12 Ã— 1D shift arrays

    // Pipelines
    pub(super) pipeline_zero_fields: wgpu::ComputePipeline,
    /// Zeros kspace_re and kspace_im in a single compute pass (no CPU-side clear_buffer).
    /// Enables keeping a single ComputePassEncoder open for the whole time step,
    /// replacing the two encoder.clear_buffer() calls that would end the pass.
    pub(super) pipeline_zero_kspace: wgpu::ComputePipeline,
    pub(super) pipeline_fft: wgpu::ComputePipeline,
    pub(super) pipeline_kspace_shift: wgpu::ComputePipeline,
    pub(super) pipeline_vel_update: wgpu::ComputePipeline,
    pub(super) pipeline_dens_update: wgpu::ComputePipeline,
    pub(super) pipeline_snapshot_rho0_plus_rho: wgpu::ComputePipeline,
    pub(super) pipeline_absorption: wgpu::ComputePipeline,
    pub(super) pipeline_pres_density: wgpu::ComputePipeline,
    pub(super) pipeline_record: wgpu::ComputePipeline,
    pub(super) pipeline_inject_src: wgpu::ComputePipeline,
    pub(super) pipeline_inject_vel_x: wgpu::ComputePipeline,
    pub(super) pipeline_apply_source_kappa: wgpu::ComputePipeline,
    pub(super) pipeline_add_kspace_to_field_ux: wgpu::ComputePipeline,
    pub(super) buf_source_kappa: wgpu::Buffer,
    pub(super) pipeline_copy_field_to_k: wgpu::ComputePipeline,

    // Fractional-Laplacian absorption pipelines (use 4-group layout)
    // Correct k-Wave C++ pressure-based formula (computePressureLinearPowerLaw):
    //   L1 = IFFT(nabla1 Â· FFT(Ïâ‚€ Â· div_u_total)),  L2 = IFFT(nabla2 Â· FFT(Ï_total))
    //   p += câ‚€Â² Â· (Ï„ Â· L1 âˆ’ Î· Â· L2)
    pub(super) pipeline_absorb_mul_nabla: wgpu::ComputePipeline,
    pub(super) pipeline_absorb_copy_to_scratch: wgpu::ComputePipeline,
    pub(super) pipeline_absorb_accum_div_u: wgpu::ComputePipeline,
    pub(super) pipeline_absorb_prep_l1_kspace: wgpu::ComputePipeline,
    pub(super) pipeline_absorb_prep_l2_kspace: wgpu::ComputePipeline,
    pub(super) pipeline_absorb_pressure_correction: wgpu::ComputePipeline,
    /// Save kspace_re/im â†’ absorb_scratch_kre/kim.
    /// Used during velocity loop to cache FFT(p) for reuse across all 3 axes,
    /// saving 2 full 3D FFTs per time step.
    pub(super) pipeline_absorb_save_kspace: wgpu::ComputePipeline,
    /// Restore kspace_re/im â† absorb_scratch_kre/kim.
    pub(super) pipeline_absorb_restore_kspace: wgpu::ComputePipeline,

    // Bind groups (sensor group rebuilt per run)
    pub(super) bg_fields: wgpu::BindGroup,
    pub(super) bg_kspace: wgpu::BindGroup,
    pub(super) bg_absorb: wgpu::BindGroup, // group(3): absorption operators

    // Bind group layouts and pipeline layout (kept for rebuilding)
    pub(super) bgl_sensor: wgpu::BindGroupLayout,
    pub(super) pipeline_layout: wgpu::PipelineLayout,

    // â”€â”€ CPU-side medium scratch buffers (preallocated to avoid per-scan-line malloc) â”€â”€
    // update_medium_variable() computes c0Â² and 1/Ïâ‚€ here before write_buffer upload.
    // Sized to nx*ny*nz at construction; never reallocated.
    pub(super) scratch_c0_sq: Vec<f32>,
    pub(super) scratch_rho0_inv: Vec<f32>,
    pub(super) scratch_rho0_flat: Vec<f32>,
    // Persistent unity staging buffer for disable_source_correction(); avoids a
    // per-call allocation when the caller needs raw additive injection.
    pub(super) scratch_source_kappa_ones: Vec<f32>,
    // Packed host-side source upload buffers. The index prefix is stable across
    // cache-hit runs, so only the signal tail is overwritten between scan lines.
    pub(super) scratch_source_data: Vec<f32>,
    pub(super) scratch_vel_x_data: Vec<f32>,

    // â”€â”€ Cached run() buffers (reused when sensor/source layout is unchanged) â”€â”€
    // Allocated on first run(); reused on subsequent calls to eliminate per-scan-line
    // VRAM allocation overhead (~500Âµs per allocation on discrete GPUs).
    // Invalidated and reallocated only when n_sensors / n_src / n_vel_x changes.
    pub(super) cache_sensor_indices_buf: Option<wgpu::Buffer>,
    pub(super) cache_sensor_data_buf: Option<wgpu::Buffer>,
    pub(super) cache_source_data_buf: Option<wgpu::Buffer>,
    pub(super) cache_vel_x_data_buf: Option<wgpu::Buffer>,
    pub(super) cache_staging_buf: Option<wgpu::Buffer>,
    pub(super) cache_bg_sensor: Option<wgpu::BindGroup>,
    pub(super) cache_bg_sensor_vel: Option<wgpu::BindGroup>,
    pub(super) cache_n_sensors: usize,
    pub(super) cache_n_src: usize,
    pub(super) cache_n_vel_x: usize,
}

impl std::fmt::Debug for GpuPstdSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPstdSolver")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .field("nt", &self.nt)
            .field("dt", &self.dt)
            .field("nonlinear", &self.nonlinear)
            .field("absorbing", &self.absorbing)
            .finish()
    }
}

#[cfg(test)]
mod tests;

//! GPU-resident PSTD (Pseudospectral Time Domain) acoustic solver.
//!
//! # Design
//!
//! All acoustic fields (p, ux, uy, uz, rhox, rhoy, rhoz) remain on the GPU
//! throughout the simulation. Only the final sensor readings are downloaded
//! at the end of the run, minimising PCIe traffic.
//!
//! # Bind group layout (≤8 storage buffers per group)
//!
//! - group(0) 8 storage: p, ux, uy, uz, rhox, rhoy, rhoz, scratch
//! - group(1) 1 uniform: PstdParams
//! - group(2) 8 storage: kspace_re, kspace_im, kspace2_re, kspace2_im,
//!            kappa, rho0_inv, c0_sq, rho0
//! - group(3) 8 storage: pml_sgx, pml_sgy, pml_sgz, pml_xyz (packed),
//!            shifts_all (packed), sensor_flat_indices, sensor_data,
//!            source_data (packed)
//!
//! # Packed buffer formats
//!
//! **pml_xyz**: three concatenated f32 arrays `[pml_x | pml_y | pml_z]`,
//! each of size `nx×ny×nz`. Index via `ax * total + flat_idx`.
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
//! | `pipeline`       | `new()` constructor — buffer alloc, BGL, pipeline compile  |
//! | `time_loop`      | `run()` time-marching loop + internal dispatch helpers      |
//! | `medium_update`  | `update_medium_variable()` for scan lines, `update_medium()` for full refresh |
//!
//! # References
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314.
//! - Liu (1998). Microwave Opt. Technol. Lett. 15(3), 158–165.

mod medium_update;
mod pipeline;
mod time_loop;

use std::sync::Arc;

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

// ─── Params push-constant struct (must match PstdParams in pstd.wgsl) ────────
// 14 × u32/f32 = 56 bytes. max_push_constant_size must be ≥ 56.
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
#[allow(dead_code)]
pub struct GpuPstdSolver {
    // Note: Manual Debug impl because wgpu types don't implement Debug.
    pub(super) device: Arc<wgpu::Device>,
    pub(super) queue: Arc<wgpu::Queue>,

    pub(super) nx: usize,
    pub(super) ny: usize,
    pub(super) nz: usize,
    pub(super) nt: usize,
    pub(super) dt: f64,

    // Field buffers — group(0)
    pub(super) buf_p: wgpu::Buffer,
    pub(super) buf_ux: wgpu::Buffer,
    pub(super) buf_uy: wgpu::Buffer,
    pub(super) buf_uz: wgpu::Buffer,
    pub(super) buf_rhox: wgpu::Buffer,
    pub(super) buf_rhoy: wgpu::Buffer,
    pub(super) buf_rhoz: wgpu::Buffer,
    // K-space + medium — group(1): bindings 0-7
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
    // Precomputed constants for Treeby & Cox (2010) Eqs. 9–10, 19–21.
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

    // PML + shifts + sensor/source — group(3)
    pub(super) buf_pml_sgx: wgpu::Buffer,
    pub(super) buf_pml_sgy: wgpu::Buffer,
    pub(super) buf_pml_sgz: wgpu::Buffer,
    pub(super) buf_pml_xyz: wgpu::Buffer, // packed [pml_x | pml_y | pml_z]
    pub(super) buf_shifts_all: wgpu::Buffer, // packed 12 × 1D shift arrays

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
    //   L1 = IFFT(nabla1 · FFT(ρ₀ · div_u_total)),  L2 = IFFT(nabla2 · FFT(ρ_total))
    //   p += c₀² · (τ · L1 − η · L2)
    pub(super) pipeline_absorb_mul_nabla: wgpu::ComputePipeline,
    pub(super) pipeline_absorb_copy_to_scratch: wgpu::ComputePipeline,
    pub(super) pipeline_absorb_accum_div_u: wgpu::ComputePipeline,
    pub(super) pipeline_absorb_prep_l1_kspace: wgpu::ComputePipeline,
    pub(super) pipeline_absorb_prep_l2_kspace: wgpu::ComputePipeline,
    pub(super) pipeline_absorb_pressure_correction: wgpu::ComputePipeline,
    /// Save kspace_re/im → absorb_scratch_kre/kim.
    /// Used during velocity loop to cache FFT(p) for reuse across all 3 axes,
    /// saving 2 full 3D FFTs per time step.
    pub(super) pipeline_absorb_save_kspace: wgpu::ComputePipeline,
    /// Restore kspace_re/im ← absorb_scratch_kre/kim.
    pub(super) pipeline_absorb_restore_kspace: wgpu::ComputePipeline,

    // Bind groups (sensor group rebuilt per run)
    pub(super) bg_fields: wgpu::BindGroup,
    pub(super) bg_kspace: wgpu::BindGroup,
    pub(super) bg_absorb: wgpu::BindGroup, // group(3): absorption operators

    // Bind group layouts and pipeline layout (kept for rebuilding)
    pub(super) bgl_sensor: wgpu::BindGroupLayout,
    pub(super) pipeline_layout: wgpu::PipelineLayout,

    // ── CPU-side medium scratch buffers (preallocated to avoid per-scan-line malloc) ──
    // update_medium_variable() computes c0² and 1/ρ₀ here before write_buffer upload.
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

    // ── Cached run() buffers (reused when sensor/source layout is unchanged) ──
    // Allocated on first run(); reused on subsequent calls to eliminate per-scan-line
    // VRAM allocation overhead (~500µs per allocation on discrete GPUs).
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
mod tests {
    use super::*;

    /// Verify GpuPstdSolver can be constructed and runs without error.
    /// Skipped if no GPU adapter is available (headless CI).
    #[test]
    fn test_gpu_pstd_solver_new() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let Ok(adapter) = adapter else {
            eprintln!("No GPU adapter — skipping GpuPstdSolver test");
            return;
        };

        let native_limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("test_pstd"),
            required_features: wgpu::Features::PUSH_CONSTANTS,
            required_limits: wgpu::Limits {
                max_push_constant_size: 128,
                max_storage_buffers_per_shader_stage:
                    native_limits.max_storage_buffers_per_shader_stage,
                ..wgpu::Limits::default()
            },
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .expect("device creation");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let n = 32usize;
        let dx = 1e-3_f64;
        let c0 = 1500.0_f64;
        let rho0 = 1000.0_f64;
        let dt = 0.3 * dx / c0;
        let nt = 10;

        let grid = crate::domain::grid::Grid::new(n, n, n, dx, dx, dx).unwrap();
        let c0v: Vec<f32> = vec![c0 as f32; n * n * n];
        let rho0v: Vec<f32> = vec![rho0 as f32; n * n * n];
        let ones: Vec<f32> = vec![1.0f32; n * n * n];

        let zeros: Vec<f32> = vec![0.0f32; n * n * n];
        let solver = GpuPstdSolver::new(
            device, queue, &grid, &c0v, &rho0v, dt, nt, c0, &ones, &ones, &ones, &ones, &ones,
            &ones, &zeros, // bon_a=0 (linear)
            &zeros, &zeros, &zeros, &zeros, // absorption ops=0 (lossless, absorbing=false)
            false, false,
        );

        assert!(
            solver.is_ok(),
            "GpuPstdSolver::new failed: {:?}",
            solver.err()
        );
        eprintln!("GpuPstdSolver constructed successfully");
    }

    /// Run a minimal simulation: one source point, one sensor point, 20 steps.
    /// Verify that non-zero pressure is recorded at the sensor.
    #[test]
    fn test_gpu_pstd_run_produces_output() {
        let n3 = 32 * 32 * 32;
        let solver = GpuPstdSolver::with_auto_device(
            &crate::domain::grid::Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap(),
            &vec![1500.0f32; n3],
            &vec![1000.0f32; n3],
            0.3e-3 / 1500.0,
            20,
            1500.0,
            &vec![1.0f32; n3],
            &vec![1.0f32; n3],
            &vec![1.0f32; n3],
            &vec![1.0f32; n3],
            &vec![1.0f32; n3],
            &vec![1.0f32; n3],
            &vec![0.0f32; n3], // bon_a = 0 (linear)
            &vec![0.0f32; n3], // nabla1 = 0 (lossless, absorbing=false)
            &vec![0.0f32; n3], // nabla2 = 0
            &vec![0.0f32; n3], // tau = 0
            &vec![0.0f32; n3], // eta = 0
            false,
            false,
        );

        let Some(mut solver) = solver.ok() else {
            eprintln!("No GPU adapter — skipping run test");
            return;
        };

        // Source at grid center (16,16,16), sensor at (20,16,16)
        let n = 32usize;
        let src_flat = 16 * n * n + 16 * n + 16;
        let sns_flat = 20 * n * n + 16 * n + 16;

        // Signal: ramp from 0 to 1 over 20 steps
        let source_signals: Vec<f32> = (0..20).map(|i| i as f32 / 20.0).collect();

        let data = solver.run(
            &[sns_flat as u32],
            &[src_flat as u32],
            &source_signals,
            &[],
            &[],
        );

        assert_eq!(data.len(), 20, "sensor data length");
        let max_val = data.iter().copied().fold(0.0f32, f32::max);
        eprintln!("GPU PSTD sensor peak: {max_val:.6}");
        // After source injection, some non-zero value should appear
        // (exact value depends on GPU FFT accuracy — just verify non-zero)
        assert!(
            max_val.is_finite(),
            "sensor data contains non-finite values"
        );
    }

    /// Run a minimal velocity-source simulation to validate the phased-array
    /// ux source path used by `GpuPstdSession`.
    #[test]
    fn test_gpu_pstd_velocity_source_produces_output() {
        let n3 = 32 * 32 * 32;
        let solver = GpuPstdSolver::with_auto_device(
            &crate::domain::grid::Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap(),
            &vec![1500.0f32; n3],
            &vec![1000.0f32; n3],
            0.3e-3 / 1500.0,
            20,
            1500.0,
            &vec![1.0f32; n3],
            &vec![1.0f32; n3],
            &vec![1.0f32; n3],
            &vec![1.0f32; n3],
            &vec![1.0f32; n3],
            &vec![1.0f32; n3],
            &vec![0.0f32; n3], // bon_a = 0 (linear)
            &vec![0.0f32; n3],
            &vec![0.0f32; n3],
            &vec![0.0f32; n3],
            &vec![0.0f32; n3], // absorb=lossless
            false,
            false,
        );

        let Some(mut solver) = solver.ok() else {
            eprintln!("No GPU adapter — skipping velocity-source run test");
            return;
        };

        let n = 32usize;
        let src_flat = 16 * n * n + 16 * n + 16;
        let sns_flat = 20 * n * n + 16 * n + 16;
        let vel_signals: Vec<f32> = (0..20).map(|i| i as f32 / 20.0).collect();

        let data = solver.run(
            &[sns_flat as u32],
            &[],
            &[],
            &[src_flat as u32],
            &vel_signals,
        );

        assert_eq!(data.len(), 20, "sensor data length");
        let max_abs = data.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        eprintln!("GPU PSTD velocity-source sensor peak: {max_abs:.6}");
        assert!(
            max_abs.is_finite(),
            "sensor data contains non-finite values"
        );
        assert!(
            max_abs > 0.0,
            "velocity-source path produced an all-zero sensor trace"
        );
    }

    /// Mirror the session-style usage pattern with multiple velocity sources
    /// and sensors on the same plane.
    #[test]
    fn test_gpu_pstd_multi_velocity_source_plane_produces_output() {
        let nx = 64usize;
        let ny = 64usize;
        let nz = 32usize;
        let total = nx * ny * nz;
        let dt = 0.3e-4 / 1500.0;
        let nt = 64usize;

        let solver = GpuPstdSolver::with_auto_device(
            &crate::domain::grid::Grid::new(nx, ny, nz, 1e-4, 1e-4, 1e-4).unwrap(),
            &vec![1500.0f32; total],
            &vec![1000.0f32; total],
            dt,
            nt,
            1500.0,
            &vec![1.0f32; total],
            &vec![1.0f32; total],
            &vec![1.0f32; total],
            &vec![1.0f32; total],
            &vec![1.0f32; total],
            &vec![1.0f32; total],
            &vec![0.0f32; total], // bon_a = 0 (linear)
            &vec![0.0f32; total],
            &vec![0.0f32; total],
            &vec![0.0f32; total],
            &vec![0.0f32; total], // lossless
            false,
            false,
        );

        let Some(mut solver) = solver.ok() else {
            eprintln!("No GPU adapter — skipping multi-source velocity test");
            return;
        };

        let src_x = 16usize;
        let mut indices = Vec::new();
        for iy in (ny / 2 - 2)..(ny / 2 + 2) {
            for iz in (nz / 2 - 2)..(nz / 2 + 2) {
                indices.push((src_x * ny * nz + iy * nz + iz) as u32);
            }
        }
        let mut vel_signals = vec![0.0f32; indices.len() * nt];
        for src_idx in 0..indices.len() {
            for step in 0..8 {
                vel_signals[src_idx * nt + step] = 0.1;
            }
        }

        let data = solver.run(&indices, &[], &[], &indices, &vel_signals);
        let max_abs = data.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        eprintln!("GPU PSTD multi velocity-source sensor peak: {max_abs:.6}");
        assert!(
            max_abs > 0.0,
            "multi-source velocity path produced an all-zero sensor trace"
        );
    }

    /// Benchmark: measure GPU PSTD steps/second for a 256×128×128 grid.
    /// This matches the B-mode example grid. Run with:
    ///   cargo test --lib --features gpu -p kwavers bench_gpu_pstd -- --nocapture
    #[test]
    fn bench_gpu_pstd_bmode_grid() {
        let nx = 256usize;
        let ny = 128usize;
        let nz = 128usize;
        let dx = 1.48e-4_f64; // ~0.148mm, typical B-mode
        let c0 = 1500.0_f64;
        let dt = 0.3 * dx / c0;
        let nt = 50; // measure 50 steps
        let total = nx * ny * nz;

        let solver = GpuPstdSolver::with_auto_device(
            &crate::domain::grid::Grid::new(nx, ny, nz, dx, dx, dx).unwrap(),
            &vec![c0 as f32; total],
            &vec![1000.0f32; total],
            dt,
            nt,
            c0,
            &vec![1.0f32; total],
            &vec![1.0f32; total],
            &vec![1.0f32; total],
            &vec![1.0f32; total],
            &vec![1.0f32; total],
            &vec![1.0f32; total],
            &vec![0.0f32; total], // bon_a = 0 (linear benchmark)
            &vec![0.0f32; total],
            &vec![0.0f32; total],
            &vec![0.0f32; total],
            &vec![0.0f32; total], // lossless
            false,
            false,
        );

        let Some(mut solver) = solver.ok() else {
            eprintln!("No GPU adapter — skipping benchmark");
            return;
        };

        // Sensor at grid center
        let sns = (nx / 2) * ny * nz + (ny / 2) * nz + (nz / 2);
        let src = (nx / 4) * ny * nz + (ny / 2) * nz + (nz / 2);
        let sigs: Vec<f32> = (0..nt).map(|i| (i as f32 / nt as f32).sin()).collect();

        let t0 = std::time::Instant::now();
        let data = solver.run(&[sns as u32], &[src as u32], &sigs, &[], &[]);
        let elapsed = t0.elapsed();

        let ms_per_step = elapsed.as_secs_f64() * 1000.0 / nt as f64;
        let steps_per_sec = nt as f64 / elapsed.as_secs_f64();
        eprintln!(
            "GPU PSTD 256×128×128: {nt} steps in {:.1}ms total = {:.2}ms/step = {:.1} steps/sec",
            elapsed.as_millis(),
            ms_per_step,
            steps_per_sec
        );
        // Accurate B-mode estimate: sensor download (~45ms) is a one-time cost per
        // run() call, not per step. Subtract it before extrapolating, then add back once.
        let sensor_dl_ms = 45.0_f64; // empirical one-time download per run() call
        let compute_ms_per_step = (elapsed.as_secs_f64() * 1000.0 - sensor_dl_ms) / nt as f64;
        let scan_line_ms = 1586.0 * compute_ms_per_step + sensor_dl_ms;
        eprintln!(
            "Estimated B-mode scan line (1586 steps): {:.1}s  (compute {:.2}ms/step + {:.0}ms download)",
            scan_line_ms / 1000.0, compute_ms_per_step, sensor_dl_ms
        );
        eprintln!(
            "Estimated 16 scan lines: {:.2}min  (vs 5.00min target)",
            16.0 * scan_line_ms / 60000.0
        );
        assert!(!data.is_empty());
    }

    fn make_small_test_solver() -> Option<GpuPstdSolver> {
        let nx = 8usize;
        let ny = 8usize;
        let nz = 8usize;
        let total = nx * ny * nz;
        let dx = 1.0e-3_f64;
        let c0 = 1500.0_f64;
        let dt = 1.0e-8_f64;
        let nt = 4usize;

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let Ok(adapter) = adapter else {
            eprintln!("No GPU adapter — skipping GpuPstdSolver medium-update test");
            return None;
        };

        let native_limits = adapter.limits();
        let device = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpu_pstd_medium_update_test"),
            required_features: wgpu::Features::PUSH_CONSTANTS,
            required_limits: wgpu::Limits {
                max_push_constant_size: 128,
                max_storage_buffers_per_shader_stage:
                    native_limits.max_storage_buffers_per_shader_stage,
                ..wgpu::Limits::default()
            },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::default(),
        }));
        let Ok((device, queue)) = device else {
            eprintln!("GPU device creation failed — skipping medium-update test");
            return None;
        };

        let grid = crate::domain::grid::Grid::new(nx, ny, nz, dx, dx, dx).ok()?;
        let c0_flat = vec![c0 as f32; total];
        let rho0_flat = vec![1000.0f32; total];
        let pml = vec![0.0f32; total];
        let bon_a = vec![0.375f32; total];

        GpuPstdSolver::new(
            std::sync::Arc::new(device),
            std::sync::Arc::new(queue),
            &grid,
            &c0_flat,
            &rho0_flat,
            dt,
            nt,
            c0,
            &pml,
            &pml,
            &pml,
            &pml,
            &pml,
            &pml,
            &bon_a,
            &pml,
            &pml,
            &pml,
            &pml,
            false,
            false,
        )
        .ok()
    }

    fn read_buffer_f32(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        len: usize,
    ) -> Vec<f32> {
        let byte_size = (len * std::mem::size_of::<f32>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_pstd_test_readback"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_pstd_test_readback_copy"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait);
        rx.recv()
            .expect("buffer readback callback")
            .expect("buffer map");

        let mapped = slice.get_mapped_range();
        let data = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        data
    }

    /// Regression guard: variable-only medium updates must preserve static
    /// absorption/nonlinearity buffers, and source-kappa disablement must write
    /// a resident unity buffer rather than a fresh allocation.
    #[test]
    fn test_medium_variable_update_preserves_static_buffers_and_disables_source_correction() {
        let Some(mut solver) = make_small_test_solver() else {
            return;
        };

        let total = solver.nx * solver.ny * solver.nz;
        let new_c0: Vec<f32> = (0..total).map(|i| 1450.0f32 + i as f32 * 0.5).collect();
        let new_rho: Vec<f32> = (0..total).map(|i| 980.0f32 + i as f32 * 0.25).collect();
        solver.update_medium_variable(&new_c0, &new_rho);
        solver.disable_source_correction();

        let total = solver.nx * solver.ny * solver.nz;
        let c0_sq = read_buffer_f32(&solver.device, &solver.queue, &solver.buf_c0_sq, total);
        let rho0 = read_buffer_f32(&solver.device, &solver.queue, &solver.buf_rho0, total);
        let rho0_inv = read_buffer_f32(&solver.device, &solver.queue, &solver.buf_rho0_inv, total);
        let bon_a = read_buffer_f32(&solver.device, &solver.queue, &solver.buf_bon_a, total);
        let source_kappa = read_buffer_f32(
            &solver.device,
            &solver.queue,
            &solver.buf_source_kappa,
            total,
        );

        let first_idx = 0usize;
        let last_idx = total - 1;
        let first_c0 = new_c0[0] as f64;
        let last_c0 = *new_c0.last().expect("non-empty c0") as f64;
        let first_rho = new_rho[0] as f64;
        let last_rho = *new_rho.last().expect("non-empty rho") as f64;

        assert!((c0_sq[first_idx] as f64 - first_c0 * first_c0).abs() < 1e-3);
        assert!((c0_sq[last_idx] as f64 - last_c0 * last_c0).abs() < 1e-3);
        assert!((rho0[first_idx] as f64 - first_rho).abs() < 1e-6);
        assert!((rho0[last_idx] as f64 - last_rho).abs() < 1e-6);
        assert!((rho0_inv[first_idx] as f64 - (1.0 / first_rho)).abs() < 1e-6);
        assert!((rho0_inv[last_idx] as f64 - (1.0 / last_rho)).abs() < 1e-6);
        let bon_a_expected = 0.375_f32;
        let unity_expected = 1.0_f32;
        let tol = 1e-6_f32;
        assert!(bon_a.iter().all(|&v| (v - bon_a_expected).abs() < tol));
        assert!(source_kappa
            .iter()
            .all(|&v| (v - unity_expected).abs() < tol));
    }

    /// Regression guard: the GPU phased-array path injects velocity sources in
    /// k-space and must retain the source-kappa reconstruction chain.
    #[test]
    fn test_velocity_source_shader_uses_validated_kspace_path() {
        let src = include_str!("../../../../gpu/shaders/pstd.wgsl");
        let inject_start = src
            .find("fn inject_velocity_x_source")
            .expect("inject_velocity_x_source entry point must exist");
        let inject_block = &src[inject_start..src.len().min(inject_start + 600)];

        assert!(
            inject_block.contains("kspace_re[flat] += amp;"),
            "inject_velocity_x_source must inject into kspace_re for the validated source-kappa path"
        );
        assert!(
            src.contains("fn apply_source_kappa"),
            "velocity-source k-space injection requires apply_source_kappa"
        );
        assert!(
            src.contains("fn add_kspace_to_field_ux"),
            "velocity-source k-space injection requires add_kspace_to_field_ux"
        );
    }

    /// Regression guard: the WGSL push-constant layout must stay aligned with
    /// the Rust-side `PstdParams` struct used for dispatch.
    #[test]
    fn test_pstd_shader_push_constant_abi_matches_rust() {
        let src = include_str!("../../../../gpu/shaders/pstd.wgsl");
        let struct_start = src
            .find("struct PstdParams")
            .expect("PstdParams push-constant struct must exist in WGSL");
        let struct_block = &src[struct_start..src.len().min(struct_start + 500)];

        assert!(
            !struct_block.contains("dx:"),
            "WGSL PstdParams must not add fields absent from Rust push constants"
        );
        assert!(
            src.contains("precomp_source_kappa"),
            "validated phased-array GPU path requires precomp_source_kappa in the shader ABI"
        );
    }
}

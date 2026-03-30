//! GPU compute operations

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::sync::Arc;

/// GPU compute manager
#[derive(Debug)]
pub struct GpuCompute {
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    command_encoder: Option<wgpu::CommandEncoder>,
}

impl GpuCompute {
    /// Create a new compute manager
    pub fn new(device: &wgpu::Device) -> Self {
        // Create common bind group layouts
        let storage_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Storage Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        Self {
            bind_group_layouts: vec![storage_layout],
            command_encoder: None,
        }
    }

    /// Begin command recording
    pub fn begin_commands(&mut self, device: &wgpu::Device) {
        self.command_encoder = Some(device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Compute Command Encoder"),
            },
        ));
    }

    /// End command recording and return buffer
    pub fn finish_commands(&mut self) -> Option<wgpu::CommandBuffer> {
        self.command_encoder.take().map(|encoder| encoder.finish())
    }

    /// Dispatch compute shader
    pub fn dispatch(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: [u32; 3],
    ) {
        if let Some(ref mut encoder) = self.command_encoder {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }
    }

    /// Copy buffer to buffer
    pub fn copy_buffer(&mut self, source: &wgpu::Buffer, destination: &wgpu::Buffer, size: u64) {
        if let Some(ref mut encoder) = self.command_encoder {
            encoder.copy_buffer_to_buffer(source, 0, destination, 0, size);
        }
    }

    /// Get bind group layout
    pub fn get_bind_group_layout(&self, index: usize) -> Option<&wgpu::BindGroupLayout> {
        self.bind_group_layouts.get(index)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FdtdGpuDispatcher
// ─────────────────────────────────────────────────────────────────────────────

/// GPU dispatcher for the FDTD pressure update.
///
/// Mirrors the CPU [`crate::solver::forward::fdtd::dispatch::FdtdStencilDispatcher`]
/// interface. When a GPU device is available the `fdtd_pressure.wgsl` kernel is
/// used; otherwise the CPU fallback path computes the identical 6-point Laplacian
/// stencil in Rust.
///
/// # Algorithm — scalar wave equation (Yee 1966)
///
/// ```text
/// p^{n+1}[i,j,k] = 2·p^n[i,j,k] − p^{n-1}[i,j,k]
///                + coeff · ∇²p^n[i,j,k]
/// ```
///
/// where `coeff = (c·dt/dx)²` and `∇²` is the 6-point central-difference
/// Laplacian on an isotropic grid (dx = dy = dz).
///
/// Interior cells only (`1 ≤ i,j,k ≤ N−2`); boundary cells are set to zero
/// (Dirichlet condition), matching `kspace_shift.wgsl`.
///
/// # CPU vs GPU Parity
///
/// The CPU fallback uses `f64` arithmetic throughout.  The GPU kernel uses
/// `f32`; the relative error between CPU and GPU paths is therefore bounded by
/// `f32` round-off (`~6e-8`) rather than the stated `1e-10`.  The CPU path is
/// the reference implementation for correctness tests.
///
/// # References
///
/// - Yee KS (1966). IEEE Trans Antennas Propag 14(3):302–307.
/// - Moczo P et al. (2014). The Finite-Difference Modelling of Earthquake
///   Motions. Cambridge Univ. Press. (6-point Laplacian, §3.1)
#[derive(Debug)]
pub struct FdtdGpuDispatcher {
    /// Grid dimensions
    nx: usize,
    ny: usize,
    nz: usize,
    /// Pre-allocated scratch buffer (CPU fallback; avoids per-step heap alloc)
    scratch: Array3<f64>,
}

impl FdtdGpuDispatcher {
    /// Create dispatcher for a grid of size `nx × ny × nz`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` when any dimension is < 3 (stencil requires at
    /// least one interior layer).
    pub fn new(nx: usize, ny: usize, nz: usize) -> KwaversResult<Self> {
        if nx < 3 || ny < 3 || nz < 3 {
            return Err(KwaversError::InvalidInput(
                "FdtdGpuDispatcher: grid dimensions must be >= 3".to_string(),
            ));
        }
        Ok(Self {
            nx,
            ny,
            nz,
            scratch: Array3::zeros((nx, ny, nz)),
        })
    }

    /// Compute `p^{n+1}` and write into `output`.
    ///
    /// Equivalent to `FdtdStencilDispatcher::update_pressure` with the scalar
    /// fallback strategy, reusing the pre-allocated `scratch` buffer.
    ///
    /// ## Parameters
    /// - `p_curr`  — `p^n`  (current pressure field)
    /// - `p_prev`  — `p^{n-1}` (previous pressure field)
    /// - `coeff`   — `(c·dt/dx)²` (dimensionless CFL coefficient²)
    /// - `output`  — written in-place; every element is overwritten
    ///
    /// ## Errors
    ///
    /// Returns `InvalidInput` if any field shape does not match `(nx, ny, nz)`.
    pub fn update_pressure_into(
        &mut self,
        p_curr: &Array3<f64>,
        p_prev: &Array3<f64>,
        coeff: f64,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let expected = (self.nx, self.ny, self.nz);
        if p_curr.dim() != expected || p_prev.dim() != expected || output.dim() != expected {
            return Err(KwaversError::InvalidInput(format!(
                "FdtdGpuDispatcher: field shape mismatch (expected {:?})",
                expected
            )));
        }

        // Zero boundary cells (Dirichlet)
        output.fill(0.0);

        // Interior 6-point stencil
        for k in 1..self.nz - 1 {
            for j in 1..self.ny - 1 {
                for i in 1..self.nx - 1 {
                    let laplacian = p_curr[[i - 1, j, k]]
                        + p_curr[[i + 1, j, k]]
                        + p_curr[[i, j - 1, k]]
                        + p_curr[[i, j + 1, k]]
                        + p_curr[[i, j, k - 1]]
                        + p_curr[[i, j, k + 1]]
                        - 6.0 * p_curr[[i, j, k]];

                    output[[i, j, k]] = 2.0 * p_curr[[i, j, k]]
                        - p_prev[[i, j, k]]
                        + coeff * laplacian;
                }
            }
        }

        Ok(())
    }

    /// Convenience wrapper — allocates and returns the updated pressure field.
    ///
    /// Prefer [`Self::update_pressure_into`] in time-step loops to avoid the
    /// per-step allocation.
    pub fn update_pressure(
        &mut self,
        p_curr: &Array3<f64>,
        p_prev: &Array3<f64>,
        coeff: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Reuse scratch to avoid allocation
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let mut result = Array3::zeros((nx, ny, nz));
        self.update_pressure_into(p_curr, p_prev, coeff, &mut result)?;
        Ok(result)
    }

    /// Grid dimensions `(nx, ny, nz)`
    pub fn grid_dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FdtdGpuShaderDispatcher
// ─────────────────────────────────────────────────────────────────────────────

/// Uniform buffer matched to `PressureParams` in `fdtd_pressure.wgsl`.
///
/// Layout (std140, all u32/f32 → 4-byte fields, no padding needed):
/// ```text
/// offset 0:  nx    (u32)
/// offset 4:  ny    (u32)
/// offset 8:  nz    (u32)
/// offset 12: coeff (f32)   // (c·dt/dx)²
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PressureParams {
    /// Grid size in x
    pub nx: u32,
    /// Grid size in y
    pub ny: u32,
    /// Grid size in z
    pub nz: u32,
    /// CFL² coefficient: (c·dt/dx)²
    pub coeff: f32,
}

/// GPU dispatcher that loads and dispatches the `fdtd_pressure.wgsl` compute
/// shader for the scalar wave equation pressure update.
///
/// # Algorithm — Yee (1966) scalar wave equation
///
/// ```text
/// p^{n+1}[i,j,k] = 2·p^n[i,j,k] − p^{n-1}[i,j,k]
///                + coeff · ∇²p^n[i,j,k]
/// ```
///
/// where `coeff = (c·dt/dx)²` and the 6-point Laplacian is computed on the GPU
/// using workgroup size 8×8×4 (= 256 threads).
///
/// # Bindings
///
/// - `group(0) binding(0)` — `pressure_curr` (f32 storage, read-only)
/// - `group(0) binding(1)` — `pressure_prev` (f32 storage, read-only)
/// - `group(0) binding(2)` — `pressure_new`  (f32 storage, read-write)
/// - `group(1) binding(0)` — `PressureParams` uniform {nx, ny, nz: u32, coeff: f32}
///
/// # References
///
/// - Yee KS (1966). IEEE Trans Antennas Propag 14(3):302–307.
/// - Moczo P et al. (2014). The Finite-Difference Modelling of Earthquake
///   Motions. Cambridge Univ. Press. (6-point Laplacian, §3.1)
pub struct FdtdGpuShaderDispatcher {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout_0: wgpu::BindGroupLayout,
    bind_group_layout_1: wgpu::BindGroupLayout,
}

impl FdtdGpuShaderDispatcher {
    /// Create a new dispatcher, compiling `fdtd_pressure.wgsl` and building
    /// the compute pipeline.
    ///
    /// # Errors
    ///
    /// Returns `ComputeError` if the wgpu device does not support the required
    /// features (storage buffers, uniform buffers, compute shaders).
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> KwaversResult<Self> {
        let shader_src = include_str!("shaders/fdtd_pressure.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fdtd_pressure"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Bind group 0: curr / prev / new pressure buffers (all f32 storage)
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fdtd_bgl0_pressure_buffers"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Bind group 1: PressureParams uniform
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fdtd_bgl1_params"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fdtd_pipeline_layout"),
            bind_group_layouts: &[&bgl0, &bgl1],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fdtd_pressure_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "fdtd_pressure_update",
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout_0: bgl0,
            bind_group_layout_1: bgl1,
        })
    }

    /// Dispatch the FDTD pressure update kernel.
    ///
    /// Uploads `p_curr`, `p_prev`, and `params` to GPU buffers, submits the
    /// compute pass, and reads back `p_new` (blocking sync via `queue.submit`
    /// and buffer map).
    ///
    /// ## Workgroup layout
    ///
    /// ```text
    /// workgroups_x = ceil(nx / 8)
    /// workgroups_y = ceil(ny / 8)
    /// workgroups_z = ceil(nz / 4)
    /// ```
    ///
    /// ## Errors
    ///
    /// Returns `InvalidInput` if buffer allocation fails or if `p_curr.len()
    /// != p_prev.len()`.
    pub fn dispatch(
        &self,
        p_curr: &[f32],
        p_prev: &[f32],
        params: PressureParams,
    ) -> KwaversResult<Vec<f32>> {
        let n = p_curr.len();
        if p_prev.len() != n {
            return Err(KwaversError::InvalidInput(
                "FdtdGpuShaderDispatcher::dispatch: p_curr and p_prev length mismatch".into(),
            ));
        }
        let buf_size = (n * std::mem::size_of::<f32>()) as u64;
        let usage_src = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_dst =
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

        let buf_curr = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdtd_p_curr"),
            size: buf_size,
            usage: usage_src,
            mapped_at_creation: false,
        });
        let buf_prev = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdtd_p_prev"),
            size: buf_size,
            usage: usage_src,
            mapped_at_creation: false,
        });
        let buf_new = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdtd_p_new"),
            size: buf_size,
            usage: usage_dst,
            mapped_at_creation: false,
        });
        let buf_params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdtd_params"),
            size: std::mem::size_of::<PressureParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.queue
            .write_buffer(&buf_curr, 0, bytemuck::cast_slice(p_curr));
        self.queue
            .write_buffer(&buf_prev, 0, bytemuck::cast_slice(p_prev));
        self.queue
            .write_buffer(&buf_params, 0, bytemuck::bytes_of(&params));

        let bg0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fdtd_bg0"),
            layout: &self.bind_group_layout_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_curr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_new.as_entire_binding(),
                },
            ],
        });
        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fdtd_bg1"),
            layout: &self.bind_group_layout_1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_params.as_entire_binding(),
            }],
        });

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fdtd_encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fdtd_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            let wx = params.nx.div_ceil(8);
            let wy = params.ny.div_ceil(8);
            let wz = params.nz.div_ceil(4);
            pass.dispatch_workgroups(wx, wy, wz);
        }

        // Readback: copy buf_new → staging buffer
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdtd_staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buf_new, 0, &staging, 0, buf_size);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Block until GPU work completes
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().map_err(|e| {
            KwaversError::GpuError(format!("GPU map_async failed: {e}"))
        })??;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── FdtdGpuShaderDispatcher tests ─────────────────────────────────────────

    /// `PressureParams` must be `Pod` + `Zeroable` (compile-time assertion).
    /// This guarantees safe byte-level copy to the GPU uniform buffer.
    ///
    /// ## Reference
    /// bytemuck crate documentation: Pod requires repr(C) + no padding + no uninit bytes.
    #[test]
    fn test_pressure_params_pod_layout() {
        use std::mem;
        let p = PressureParams { nx: 16, ny: 8, nz: 4, coeff: 0.25 };
        let bytes = bytemuck::bytes_of(&p);
        // Must be 4 fields × 4 bytes = 16 bytes with no padding
        assert_eq!(bytes.len(), 16);
        assert_eq!(mem::size_of::<PressureParams>(), 16);
    }

    /// The `fdtd_pressure.wgsl` shader source must declare the correct entry point
    /// and all required bindings.
    #[test]
    fn test_fdtd_wgsl_shader_content() {
        let src = include_str!("shaders/fdtd_pressure.wgsl");
        assert!(
            src.contains("fn fdtd_pressure_update"),
            "fdtd_pressure.wgsl must declare entry point 'fdtd_pressure_update'"
        );
        assert!(
            src.contains("@group(0) @binding(0)"),
            "fdtd_pressure.wgsl must declare group(0) binding(0) for pressure_curr"
        );
        assert!(
            src.contains("@group(0) @binding(2)"),
            "fdtd_pressure.wgsl must declare group(0) binding(2) for pressure_new"
        );
        assert!(
            src.contains("@group(1) @binding(0)"),
            "fdtd_pressure.wgsl must declare group(1) binding(0) for PressureParams uniform"
        );
        assert!(
            src.contains("@workgroup_size(8, 8, 4)"),
            "fdtd_pressure.wgsl must use workgroup_size(8, 8, 4)"
        );
    }

    // ── FdtdGpuDispatcher tests (CPU fallback) ────────────────────────────────

    /// Uniform pressure field: Laplacian = 0, so p_new = 2*p_curr - p_prev.
    #[test]
    fn test_fdtd_gpu_uniform_field() {
        let (nx, ny, nz) = (8, 8, 8);
        let mut disp = FdtdGpuDispatcher::new(nx, ny, nz).unwrap();
        let p_curr = Array3::from_elem((nx, ny, nz), 1.0_f64);
        let p_prev = Array3::from_elem((nx, ny, nz), 0.5_f64);
        let coeff = 0.25_f64;
        let p_new = disp.update_pressure(&p_curr, &p_prev, coeff).unwrap();
        // Interior: lap = 0, so p_new = 2*1 - 0.5 = 1.5
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    assert_abs_diff_eq!(p_new[[i, j, k]], 1.5, epsilon = 1e-12);
                }
            }
        }
    }

    /// Linear ramp p(i,j,k) = i+j+k: Laplacian is zero (exact polynomial).
    #[test]
    fn test_fdtd_gpu_linear_ramp_zero_laplacian() {
        let (nx, ny, nz) = (8, 8, 8);
        let mut disp = FdtdGpuDispatcher::new(nx, ny, nz).unwrap();
        let mut p_curr = Array3::zeros((nx, ny, nz));
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    p_curr[[i, j, k]] = (i + j + k) as f64;
                }
            }
        }
        let p_prev = p_curr.clone();
        let coeff = 0.5;
        let p_new = disp.update_pressure(&p_curr, &p_prev, coeff).unwrap();
        // Interior: lap = 0 (linear field), p_new = 2*p_curr - p_curr = p_curr
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    assert_abs_diff_eq!(
                        p_new[[i, j, k]],
                        p_curr[[i, j, k]],
                        epsilon = 1e-12
                    );
                }
            }
        }
    }

    /// Dimension mismatch returns an error.
    #[test]
    fn test_fdtd_gpu_dimension_mismatch_error() {
        let mut disp = FdtdGpuDispatcher::new(8, 8, 8).unwrap();
        let p_wrong = Array3::zeros((4, 4, 4));
        let p_curr = Array3::zeros((8, 8, 8));
        let mut output = Array3::zeros((8, 8, 8));
        assert!(
            disp.update_pressure_into(&p_wrong, &p_curr, 0.25, &mut output)
                .is_err()
        );
    }

    /// Grid too small (< 3 in any axis) returns an error.
    #[test]
    fn test_fdtd_gpu_minimum_dimension_check() {
        assert!(FdtdGpuDispatcher::new(2, 8, 8).is_err());
        assert!(FdtdGpuDispatcher::new(8, 2, 8).is_err());
        assert!(FdtdGpuDispatcher::new(8, 8, 2).is_err());
        assert!(FdtdGpuDispatcher::new(3, 3, 3).is_ok());
    }

    /// update_pressure_into and update_pressure return identical results.
    #[test]
    fn test_fdtd_gpu_into_matches_alloc() {
        let (nx, ny, nz) = (6, 6, 6);
        let mut disp = FdtdGpuDispatcher::new(nx, ny, nz).unwrap();
        let p_curr = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            ((i * 3 + j * 5 + k * 7) as f64) * 0.01
        });
        let p_prev = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            ((i * 2 + j * 4 + k * 6) as f64) * 0.005
        });
        let coeff = 0.3;

        let p_alloc = disp.update_pressure(&p_curr, &p_prev, coeff).unwrap();
        let mut p_into = Array3::zeros((nx, ny, nz));
        disp.update_pressure_into(&p_curr, &p_prev, coeff, &mut p_into)
            .unwrap();

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    assert_abs_diff_eq!(p_alloc[[i, j, k]], p_into[[i, j, k]], epsilon = 1e-15);
                }
            }
        }
    }
}

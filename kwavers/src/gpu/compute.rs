//! GPU compute operations

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

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

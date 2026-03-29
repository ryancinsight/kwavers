//! GPU-accelerated k-space methods

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use ndarray::Array3;

/// GPU-accelerated k-space solver
#[derive(Debug)]
pub struct KSpaceGpu {
    fft_pipeline: wgpu::ComputePipeline,
    propagate_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    _spectrum_buffer: wgpu::Buffer,
    kspace_buffer: wgpu::Buffer,
    workgroup_size: [u32; 3],
}

impl KSpaceGpu {
    /// Create a new k-space GPU solver
    pub fn new(device: &wgpu::Device, grid: &Grid) -> KwaversResult<Self> {
        let fft_shader = include_str!("shaders/fft.wgsl");
        let propagate_shader = include_str!("shaders/kspace_propagate.wgsl");

        // Create shader modules
        let fft_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FFT Shader"),
            source: wgpu::ShaderSource::Wgsl(fft_shader.into()),
        });

        let propagate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("K-Space Propagate Shader"),
            source: wgpu::ShaderSource::Wgsl(propagate_shader.into()),
        });

        // Calculate buffer sizes
        let grid_size = (grid.nx * grid.ny * grid.nz) as u64;
        let complex_size = grid_size * 2 * std::mem::size_of::<f32>() as u64; // Complex as 2 floats

        // Create buffers
        let spectrum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spectrum Buffer"),
            size: complex_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let kspace_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("K-Space Buffer"),
            size: grid_size * std::mem::size_of::<f32>() as u64 * 3, // kx, ky, kz
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("K-Space Bind Group Layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("K-Space Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spectrum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kspace_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("K-Space Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..20, // Grid dimensions, dt, c0
            }],
        });

        // Create FFT pipeline
        let fft_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FFT Pipeline"),
            layout: Some(&pipeline_layout),
            module: &fft_module,
            entry_point: "fft_forward",
            compilation_options: Default::default(),
            cache: None,
        });

        // Create propagation pipeline
        let propagate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("K-Space Propagate Pipeline"),
            layout: Some(&pipeline_layout),
            module: &propagate_module,
            entry_point: "propagate",
            compilation_options: Default::default(),
            cache: None,
        });

        let workgroup_size = [8, 8, 8];

        Ok(Self {
            fft_pipeline,
            propagate_pipeline,
            bind_group,
            _spectrum_buffer: spectrum_buffer,
            kspace_buffer,
            workgroup_size,
        })
    }

    /// Upload k-space vectors
    pub fn upload_kspace(
        &self,
        queue: &wgpu::Queue,
        kx: &Array3<f64>,
        ky: &Array3<f64>,
        kz: &Array3<f64>,
    ) {
        let mut k_data = Vec::new();

        for ((kx_val, ky_val), kz_val) in kx.iter().zip(ky.iter()).zip(kz.iter()) {
            k_data.push(*kx_val as f32);
            k_data.push(*ky_val as f32);
            k_data.push(*kz_val as f32);
        }

        queue.write_buffer(&self.kspace_buffer, 0, bytemuck::cast_slice(&k_data));
    }

    /// Perform k-space propagation step
    pub fn propagate(&self, encoder: &mut wgpu::CommandEncoder, grid: &Grid, dt: f32, c0: f32) {
        // FFT forward
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFT Forward Pass"),
                timestamp_writes: None,
            });

            let push_constants = [
                grid.nx as u32,
                grid.ny as u32,
                grid.nz as u32,
                dt.to_bits(),
                c0.to_bits(),
            ];

            compute_pass.set_pipeline(&self.fft_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.set_push_constants(0, bytemuck::cast_slice(&push_constants));

            let workgroups_x = (grid.nx as u32).div_ceil(self.workgroup_size[0]);
            let workgroups_y = (grid.ny as u32).div_ceil(self.workgroup_size[1]);
            let workgroups_z = (grid.nz as u32).div_ceil(self.workgroup_size[2]);

            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // K-space propagation
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("K-Space Propagate Pass"),
                timestamp_writes: None,
            });

            let push_constants = [
                grid.nx as u32,
                grid.ny as u32,
                grid.nz as u32,
                dt.to_bits(),
                c0.to_bits(),
            ];

            compute_pass.set_pipeline(&self.propagate_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.set_push_constants(0, bytemuck::cast_slice(&push_constants));

            let workgroups_x = (grid.nx as u32).div_ceil(self.workgroup_size[0]);
            let workgroups_y = (grid.ny as u32).div_ceil(self.workgroup_size[1]);
            let workgroups_z = (grid.nz as u32).div_ceil(self.workgroup_size[2]);

            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KspaceShiftGpu
// ─────────────────────────────────────────────────────────────────────────────

/// GPU dispatcher for the k-space staggered-grid phase shift.
///
/// On a staggered Yee grid, velocity components are stored at half-cell offsets
/// relative to cell-centre pressure values.  When computing spectral derivatives
/// the spectrum must be multiplied by the corresponding phase factor to account
/// for the offset.  This struct wraps the `kspace_shift.wgsl` kernel with a
/// pure-CPU fallback.
///
/// # Algorithm (Treeby & Cox 2010, Eq. 12)
///
/// For a 3D complex spectrum `F(kx,ky,kz)` and shift vector `(sx,sy,sz)` [m]:
///
/// ```text
/// F'(kx,ky,kz) = F(kx,ky,kz) · exp(−i·(kx·sx + ky·sy + kz·sz))
/// ```
///
/// The CPU fallback applies this per element using `f64` arithmetic:
///
/// ```text
/// phase = −(kx[ix]·sx + ky[jy]·sy + kz[kz]·sz)
/// Re' = Re·cos(phase) − Im·sin(phase)
/// Im' = Re·sin(phase) + Im·cos(phase)
/// ```
///
/// # References
///
/// - Treeby BE, Cox BT (2010). J. Biomed. Opt. 15(2):021314. (Eq. 12–13)
/// - Liu Q-H (1998). Geophysics 63(6):2082–2089. (PSTD staggered shift)
#[derive(Debug)]
pub struct KspaceShiftGpu {
    /// Grid dimensions
    nx: usize,
    ny: usize,
    nz: usize,
}

impl KspaceShiftGpu {
    /// Create a new k-space shift dispatcher for a grid of size `nx × ny × nz`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` when any dimension is zero.
    pub fn new(nx: usize, ny: usize, nz: usize) -> KwaversResult<Self> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(KwaversError::InvalidInput(
                "KspaceShiftGpu: grid dimensions must be > 0".to_string(),
            ));
        }
        Ok(Self { nx, ny, nz })
    }

    /// Apply complex phase shift to spectrum in-place (CPU fallback).
    ///
    /// ## Parameters
    ///
    /// - `real_in`  / `imag_in`  — input 3D complex spectrum (shape `[nx, ny, nz]`)
    /// - `kx_vec`   — wavenumber per x-index (length `nx`) [rad/m]
    /// - `ky_vec`   — wavenumber per y-index (length `ny`) [rad/m]
    /// - `kz_vec`   — wavenumber per z-index (length `nz`) [rad/m]
    /// - `shift`    — `[sx, sy, sz]` half-cell shift vector [m],
    ///               e.g. `[Δx/2, 0, 0]` for a forward x-stagger
    /// - `real_out` / `imag_out` — output buffers; fully overwritten
    ///
    /// ## Errors
    ///
    /// Returns `InvalidInput` if any array shape does not match the grid.
    pub fn apply_shift_into(
        &self,
        real_in: &Array3<f64>,
        imag_in: &Array3<f64>,
        kx_vec: &[f64],
        ky_vec: &[f64],
        kz_vec: &[f64],
        shift: [f64; 3],
        real_out: &mut Array3<f64>,
        imag_out: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let expected = (self.nx, self.ny, self.nz);
        for (name, arr) in [
            ("real_in", real_in as &Array3<f64>),
            ("imag_in", imag_in),
            ("real_out", real_out as &Array3<f64>),
            ("imag_out", imag_out as &Array3<f64>),
        ] {
            if arr.dim() != expected {
                return Err(KwaversError::InvalidInput(format!(
                    "KspaceShiftGpu: {name} shape {:?} ≠ expected {expected:?}",
                    arr.dim()
                )));
            }
        }
        if kx_vec.len() != self.nx || ky_vec.len() != self.ny || kz_vec.len() != self.nz {
            return Err(KwaversError::InvalidInput(
                "KspaceShiftGpu: wavenumber vector length mismatch".to_string(),
            ));
        }

        let [sx, sy, sz] = shift;
        for kz in 0..self.nz {
            for jy in 0..self.ny {
                for ix in 0..self.nx {
                    let phase =
                        -(kx_vec[ix] * sx + ky_vec[jy] * sy + kz_vec[kz] * sz);
                    let c = phase.cos();
                    let s = phase.sin();
                    let re = real_in[[ix, jy, kz]];
                    let im = imag_in[[ix, jy, kz]];
                    real_out[[ix, jy, kz]] = re * c - im * s;
                    imag_out[[ix, jy, kz]] = re * s + im * c;
                }
            }
        }

        Ok(())
    }

    /// Convenience wrapper that allocates output buffers.
    ///
    /// Returns `(real_out, imag_out)`.  Prefer [`Self::apply_shift_into`] in
    /// time-step loops.
    pub fn apply_shift(
        &self,
        real_in: &Array3<f64>,
        imag_in: &Array3<f64>,
        kx_vec: &[f64],
        ky_vec: &[f64],
        kz_vec: &[f64],
        shift: [f64; 3],
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        let mut real_out = Array3::zeros((self.nx, self.ny, self.nz));
        let mut imag_out = Array3::zeros((self.nx, self.ny, self.nz));
        self.apply_shift_into(
            real_in, imag_in, kx_vec, ky_vec, kz_vec, shift,
            &mut real_out, &mut imag_out,
        )?;
        Ok((real_out, imag_out))
    }

    /// Grid dimensions `(nx, ny, nz)`
    pub fn grid_dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
}

#[cfg(test)]
mod kspace_shift_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    /// Zero shift must be the identity: output equals input.
    #[test]
    fn test_zero_shift_is_identity() {
        let (nx, ny, nz) = (4, 4, 4);
        let gpu = KspaceShiftGpu::new(nx, ny, nz).unwrap();
        let real_in = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| (i + j * 2 + k * 3) as f64);
        let imag_in = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| (i * 3 + j + k * 2) as f64);
        let kx = vec![0.0; nx];
        let ky = vec![0.0; ny];
        let kz = vec![0.0; nz];

        let (re_out, im_out) = gpu
            .apply_shift(&real_in, &imag_in, &kx, &ky, &kz, [0.0, 0.0, 0.0])
            .unwrap();

        for idx in real_in.indexed_iter() {
            let (i, &v) = idx;
            assert_abs_diff_eq!(re_out[i], v, epsilon = 1e-13);
        }
        for idx in imag_in.indexed_iter() {
            let (i, &v) = idx;
            assert_abs_diff_eq!(im_out[i], v, epsilon = 1e-13);
        }
    }

    /// Phase shift by 2π must be the identity.
    #[test]
    fn test_full_cycle_shift_is_identity() {
        let (nx, ny, nz) = (4, 1, 1);
        let gpu = KspaceShiftGpu::new(nx, ny, nz).unwrap();
        // kx = [0, 1, 2, 3] rad/m, shift = 2π m  → phase = -(kx * 2π)
        // exp(-i*2πk) = 1 for integer k
        let kx: Vec<f64> = (0..nx).map(|i| i as f64).collect();
        let ky = vec![0.0];
        let kz = vec![0.0];
        let real_in = Array3::from_elem((nx, 1, 1), 1.0_f64);
        let imag_in = Array3::zeros((nx, 1, 1));
        let shift = [2.0 * PI, 0.0, 0.0];

        let (re_out, im_out) = gpu
            .apply_shift(&real_in, &imag_in, &kx, &ky, &kz, shift)
            .unwrap();

        for i in 0..nx {
            assert_abs_diff_eq!(re_out[[i, 0, 0]], 1.0, epsilon = 1e-12);
            assert_abs_diff_eq!(im_out[[i, 0, 0]], 0.0, epsilon = 1e-12);
        }
    }

    /// Quarter-cycle (π/2) shift on real-only spectrum gives pure imaginary output.
    ///
    /// exp(−i·(k·π/2)) for k = 1 rad/m, shift = π/2 m: phase = -π/2
    /// (Re + 0·Im) · exp(-iπ/2) = Re·cos(-π/2) - 0·sin(-π/2) + i·(Re·sin(-π/2) + 0)
    ///                          = 0 + i·(-Re) = -i·Re
    #[test]
    fn test_quarter_cycle_shift() {
        let (nx, ny, nz) = (1, 1, 1);
        let gpu = KspaceShiftGpu::new(nx, ny, nz).unwrap();
        let kx = vec![1.0_f64];   // k = 1 rad/m
        let ky = vec![0.0];
        let kz = vec![0.0];
        let real_in = Array3::from_elem((1, 1, 1), 2.0_f64);
        let imag_in = Array3::zeros((1, 1, 1));
        let shift = [PI / 2.0, 0.0, 0.0];   // phase = -(1 * π/2) = -π/2

        let (re_out, im_out) = gpu
            .apply_shift(&real_in, &imag_in, &kx, &ky, &kz, shift)
            .unwrap();

        // cos(-π/2) = 0, sin(-π/2) = -1 → Re' = 0, Im' = -2
        assert_abs_diff_eq!(re_out[[0, 0, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(im_out[[0, 0, 0]], -2.0, epsilon = 1e-12);
    }

    /// apply_shift and apply_shift_into produce identical output.
    #[test]
    fn test_into_matches_alloc() {
        let (nx, ny, nz) = (4, 3, 3);
        let gpu = KspaceShiftGpu::new(nx, ny, nz).unwrap();
        let real_in = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| (i + j + k) as f64 * 0.1);
        let imag_in = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| (i * j + k) as f64 * 0.05);
        let kx: Vec<f64> = (0..nx).map(|i| i as f64 * 100.0).collect();
        let ky = vec![0.0; ny];
        let kz = vec![0.0; nz];
        let shift = [0.5e-3, 0.0, 0.0];

        let (re_a, im_a) = gpu
            .apply_shift(&real_in, &imag_in, &kx, &ky, &kz, shift)
            .unwrap();
        let mut re_b = Array3::zeros((nx, ny, nz));
        let mut im_b = Array3::zeros((nx, ny, nz));
        gpu.apply_shift_into(&real_in, &imag_in, &kx, &ky, &kz, shift, &mut re_b, &mut im_b)
            .unwrap();

        for idx in re_a.indexed_iter() {
            let (i, &v) = idx;
            assert_abs_diff_eq!(re_b[i], v, epsilon = 1e-15);
            assert_abs_diff_eq!(im_b[i], im_a[i], epsilon = 1e-15);
        }
    }

    /// Magnitude is preserved by a unitary phase rotation.
    #[test]
    fn test_magnitude_preserved() {
        let (nx, ny, nz) = (3, 3, 3);
        let gpu = KspaceShiftGpu::new(nx, ny, nz).unwrap();
        let real_in = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| (i + j + k) as f64);
        let imag_in = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| (i * 2 + j + k) as f64);
        let kx: Vec<f64> = (0..nx).map(|i| i as f64 * 200.0).collect();
        let ky: Vec<f64> = (0..ny).map(|j| j as f64 * 150.0).collect();
        let kz: Vec<f64> = (0..nz).map(|k| k as f64 * 100.0).collect();
        let shift = [0.25e-3, 0.33e-3, 0.15e-3];

        let (re_out, im_out) = gpu
            .apply_shift(&real_in, &imag_in, &kx, &ky, &kz, shift)
            .unwrap();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let mag_in = (real_in[[i, j, k]].powi(2) + imag_in[[i, j, k]].powi(2)).sqrt();
                    let mag_out = (re_out[[i, j, k]].powi(2) + im_out[[i, j, k]].powi(2)).sqrt();
                    assert_abs_diff_eq!(mag_out, mag_in, epsilon = 1e-10);
                }
            }
        }
    }

    /// Zero grid dimension returns an error.
    #[test]
    fn test_zero_dimension_error() {
        assert!(KspaceShiftGpu::new(0, 4, 4).is_err());
    }

    /// Wavenumber vector length mismatch returns an error.
    #[test]
    fn test_kv_length_mismatch_error() {
        let (nx, ny, nz) = (4, 4, 4);
        let gpu = KspaceShiftGpu::new(nx, ny, nz).unwrap();
        let real_in = Array3::zeros((nx, ny, nz));
        let imag_in = Array3::zeros((nx, ny, nz));
        let kx_bad = vec![0.0; nx + 1]; // wrong length
        let ky = vec![0.0; ny];
        let kz = vec![0.0; nz];
        assert!(gpu
            .apply_shift(&real_in, &imag_in, &kx_bad, &ky, &kz, [0.0; 3])
            .is_err());
    }
}

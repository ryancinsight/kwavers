//! Low-level GPU dispatch helpers: single-group, absorption-group, 2-D, and
//! separable 3-D FFT/IFFT encoders.
//!
//! All methods operate on an already-open `ComputePass` to eliminate the
//! ~250 µs D3D12 UAV barrier that wgpu inserts at every pass boundary.

use super::super::{GpuPstdSolver, PstdParams};

impl GpuPstdSolver {
    /// Encode one dispatch into `cpass` (3-group pipeline layout).
    ///
    /// Push constants carry `params` inline — no `write_buffer()` per dispatch.
    /// Bind groups: fields(0), kspace+medium(1), sensor(2).
    #[inline]
    pub(super) fn dispatch(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        params: &PstdParams,
        pipeline: &wgpu::ComputePipeline,
        bg_sensor: &wgpu::BindGroup,
        workgroups: u32,
        _label: &str,
    ) {
        cpass.set_pipeline(pipeline);
        cpass.set_push_constants(0, bytemuck::bytes_of(params));
        cpass.set_bind_group(0, &self.bg_fields, &[]);
        cpass.set_bind_group(1, &self.bg_kspace, &[]);
        cpass.set_bind_group(2, bg_sensor, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Encode a dispatch that also binds the absorption group(3) (4-group layout).
    ///
    /// Used by fractional-Laplacian absorption shaders. The shared pipeline
    /// layout still requires group(2); the absorption kernels do not read it.
    #[inline]
    pub(super) fn dispatch_absorb(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        params: &PstdParams,
        pipeline: &wgpu::ComputePipeline,
        bg_sensor: &wgpu::BindGroup,
        workgroups: u32,
        _label: &str,
    ) {
        cpass.set_pipeline(pipeline);
        cpass.set_push_constants(0, bytemuck::bytes_of(params));
        cpass.set_bind_group(0, &self.bg_fields, &[]);
        cpass.set_bind_group(1, &self.bg_kspace, &[]);
        cpass.set_bind_group(2, bg_sensor, &[]);
        cpass.set_bind_group(3, &self.bg_absorb, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Encode a 2-D workgroup dispatch (used by FFT to handle lane counts > 65535).
    // Args are independent wgpu handles and scalar dispatch dimensions with no cohesive grouping.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_2d(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        params: &PstdParams,
        pipeline: &wgpu::ComputePipeline,
        bg_sensor: &wgpu::BindGroup,
        workgroups_x: u32,
        workgroups_y: u32,
        _label: &str,
    ) {
        cpass.set_pipeline(pipeline);
        cpass.set_push_constants(0, bytemuck::bytes_of(params));
        cpass.set_bind_group(0, &self.bg_fields, &[]);
        cpass.set_bind_group(1, &self.bg_kspace, &[]);
        cpass.set_bind_group(2, bg_sensor, &[]);
        cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// Encode one FFT lane pass along `axis` with `n_fft`-point transforms.
    ///
    /// `n_batches` FFT lanes are dispatched across a 2-D workgroup grid to
    /// respect the hardware 65535-per-dimension limit.
    // Args are independent wgpu handles and scalar FFT-lane parameters with no cohesive grouping.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn dispatch_fft_lane(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        bg_sensor: &wgpu::BindGroup,
        step: u32,
        n_sensors: u32,
        inverse: u32,
        axis: u32,
        n_fft: u32,
        n_batches: u32,
        log2n: u32,
    ) {
        let params = PstdParams {
            nx: self.nx as u32,
            ny: self.ny as u32,
            nz: self.nz as u32,
            axis,
            n_fft,
            n_batches,
            log2n,
            inverse,
            step,
            dt: self.dt as f32,
            n_sensors,
            nt: self.nt as u32,
            nonlinear: if self.nonlinear { 1 } else { 0 },
            absorbing: if self.absorbing { 1 } else { 0 },
        };
        let wg_x = n_batches.min(65535);
        let wg_y = n_batches.div_ceil(65535);
        self.dispatch_2d(
            cpass,
            &params,
            &self.pipeline_fft,
            bg_sensor,
            wg_x,
            wg_y,
            "fft",
        );
    }

    /// Encode a forward 3-D FFT: Z-axis → Y-axis → X-axis.
    ///
    /// All axis passes run inside the caller's open `ComputePass` — no UAV
    /// barriers between axes.
    pub(super) fn fft_3d(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        bg_sensor: &wgpu::BindGroup,
        step: u32,
        n_sensors: u32,
    ) {
        let (nx, ny, nz) = (self.nx as u32, self.ny as u32, self.nz as u32);
        self.dispatch_fft_lane(
            cpass,
            bg_sensor,
            step,
            n_sensors,
            0,
            2,
            nz,
            nx * ny,
            nz.trailing_zeros(),
        );
        self.dispatch_fft_lane(
            cpass,
            bg_sensor,
            step,
            n_sensors,
            0,
            1,
            ny,
            nx * nz,
            ny.trailing_zeros(),
        );
        self.dispatch_fft_lane(
            cpass,
            bg_sensor,
            step,
            n_sensors,
            0,
            0,
            nx,
            ny * nz,
            nx.trailing_zeros(),
        );
    }

    /// Encode an inverse 3-D FFT: X-axis → Y-axis → Z-axis.
    ///
    /// All axis passes run inside the caller's open `ComputePass`.
    pub(super) fn ifft_3d(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        bg_sensor: &wgpu::BindGroup,
        step: u32,
        n_sensors: u32,
    ) {
        let (nx, ny, nz) = (self.nx as u32, self.ny as u32, self.nz as u32);
        self.dispatch_fft_lane(
            cpass,
            bg_sensor,
            step,
            n_sensors,
            1,
            0,
            nx,
            ny * nz,
            nx.trailing_zeros(),
        );
        self.dispatch_fft_lane(
            cpass,
            bg_sensor,
            step,
            n_sensors,
            1,
            1,
            ny,
            nx * nz,
            ny.trailing_zeros(),
        );
        self.dispatch_fft_lane(
            cpass,
            bg_sensor,
            step,
            n_sensors,
            1,
            2,
            nz,
            nx * ny,
            nz.trailing_zeros(),
        );
    }
}

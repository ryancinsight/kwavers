// Medium property re-upload between consecutive scan-line runs.
//
// **SRP boundary**: this module changes when the set of medium-dependent
// GPU buffers changes (e.g., adding a new material property).  It does not
// change when the time-stepping algorithm or bind-group layout changes.

use super::GpuPstdSolver;

impl GpuPstdSolver {
    /// Update medium-dependent GPU buffers for a new scan-line simulation.
    ///
    /// Reuses the existing device, pipelines, kappa, and PML — only the medium
    /// properties that change between scan lines are re-uploaded. This avoids
    /// ~500ms of shader recompilation and buffer reallocation per scan line.
    ///
    /// Call this before `run()` when the medium changes (e.g., new phantom slice).
    ///
    /// # Arguments
    /// * `c0_flat`         — sound speed [m/s] for new slice, f32 [nx×ny×nz]
    /// * `rho0_flat`       — density [kg/m³] for new slice, f32 [nx×ny×nz]
    /// * `bon_a_flat`      — B/(2A) per voxel for new slice
    /// * `absorb_tau_flat` — −2α₀c₀^(y−1) per voxel (Treeby & Cox 2010 Eq. 9)
    /// * `absorb_eta_flat` — 2α₀c₀^y·tan(πy/2) per voxel (Treeby & Cox 2010 Eq. 10)
    pub fn update_medium(
        &mut self,
        c0_flat: &[f32],
        rho0_flat: &[f32],
        bon_a_flat: &[f32],
        absorb_tau_flat: &[f32],
        absorb_eta_flat: &[f32],
    ) {
        // Compute c0² and 1/ρ₀ into preallocated scratch (avoids per-scan-line malloc).
        self.scratch_c0_sq
            .iter_mut()
            .zip(c0_flat)
            .for_each(|(o, &c)| *o = c * c);
        self.scratch_rho0_inv
            .iter_mut()
            .zip(rho0_flat)
            .for_each(|(o, &r)| *o = 1.0 / r);

        self.queue.write_buffer(
            &self.buf_c0_sq,
            0,
            bytemuck::cast_slice(&self.scratch_c0_sq),
        );
        self.queue
            .write_buffer(&self.buf_rho0, 0, bytemuck::cast_slice(rho0_flat));
        self.queue.write_buffer(
            &self.buf_rho0_inv,
            0,
            bytemuck::cast_slice(&self.scratch_rho0_inv),
        );
        self.queue
            .write_buffer(&self.buf_bon_a, 0, bytemuck::cast_slice(bon_a_flat));
        self.queue.write_buffer(
            &self.buf_absorb_tau,
            0,
            bytemuck::cast_slice(absorb_tau_flat),
        );
        self.queue.write_buffer(
            &self.buf_absorb_eta,
            0,
            bytemuck::cast_slice(absorb_eta_flat),
        );
        // write_buffer calls are staged internally in wgpu; the queue.submit() in
        // the subsequent run() will flush all staged writes to the GPU before
        // any compute dispatch executes, ensuring correct ordering.
    }

    /// Overwrite the source_kappa buffer with `1.0` everywhere, effectively
    /// disabling the k-space source correction `sinc(c·dt·|k|/2)` that
    /// `apply_source_kappa` multiplies onto the injected velocity source.
    ///
    /// This matches k-wave-python's `u_mode = "additive-no-correction"`, which
    /// is the default set by `NotATransducer` (see kwave/ksource.py:186 and
    /// kwave/ktransducer.py:244). Call before `run()` on any scan line where
    /// the caller wants raw additive injection; effects persist across runs
    /// until `enable_source_correction()` is called.
    pub fn disable_source_correction(&self) {
        let nx = self.scratch_c0_sq.len();  // total voxels = nx*ny*nz
        let ones: Vec<f32> = vec![1.0f32; nx];
        self.queue
            .write_buffer(&self.buf_source_kappa, 0, bytemuck::cast_slice(&ones));
    }
}

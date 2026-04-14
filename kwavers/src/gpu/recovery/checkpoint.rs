// ── GPU Checkpoint ──────────────────────────────────────────────────────────
//
// Theorem: Checkpoint Consistency
// A checkpoint taken at step n is consistent iff:
//   pressure[i,j,k] and velocity_{x,y,z}[i,j,k] are all from the same time level n.
//
// The GPU time loop is responsible for ensuring the checkpoint is updated
// atomically with respect to the step counter (no torn reads).
//
// Reference: Elnozahy, E. N. et al. (2002). "A survey of rollback-recovery
// protocols in message-passing systems." ACM Computing Surveys 34(3), 375–408.
// DOI: 10.1145/568522.568525

/// Snapshot of simulation state at a single time step.
///
/// Held in the GPU recovery manager and updated every `checkpoint_interval` steps
/// by the GPU time loop. On OOM or device-lost, the last checkpoint is used to
/// restore state either on CPU (OOM) or re-upload to GPU (device-lost after re-init).
#[derive(Debug, Clone)]
pub struct GpuCheckpoint {
    /// Flattened pressure field p[i,j,k] in C-order (row-major). Length = nx·ny·nz.
    pub pressure: Vec<f32>,
    /// Flattened x-velocity ux[i,j,k]. Length = nx·ny·nz.
    pub velocity_x: Vec<f32>,
    /// Flattened y-velocity uy[i,j,k]. Length = nx·ny·nz.
    pub velocity_y: Vec<f32>,
    /// Flattened z-velocity uz[i,j,k]. Length = nx·ny·nz.
    pub velocity_z: Vec<f32>,
    /// Time step index at which this snapshot was taken.
    pub step: u64,
}

impl GpuCheckpoint {
    /// Create a zeroed checkpoint for a grid of given total size.
    ///
    /// Used to pre-allocate the checkpoint buffer before the simulation starts.
    /// The GPU loop fills it via `update_from_cpu_slices` every checkpoint_interval steps.
    pub fn zeroed(n_cells: usize) -> Self {
        Self {
            pressure: vec![0.0f32; n_cells],
            velocity_x: vec![0.0f32; n_cells],
            velocity_y: vec![0.0f32; n_cells],
            velocity_z: vec![0.0f32; n_cells],
            step: 0,
        }
    }

    /// Update checkpoint fields from CPU-side f32 slices.
    ///
    /// Called by the GPU time loop after a GPU→CPU staging readback.
    /// All four slices must have length `n_cells` (= nx·ny·nz), matching the
    /// checkpoint allocation from [`GpuCheckpoint::zeroed`].
    ///
    /// # Panics
    /// Panics in debug mode if slice lengths are inconsistent.
    pub fn update_from_cpu_slices(
        &mut self,
        pressure: &[f32],
        velocity_x: &[f32],
        velocity_y: &[f32],
        velocity_z: &[f32],
        step: u64,
    ) {
        debug_assert_eq!(
            pressure.len(),
            self.pressure.len(),
            "GpuCheckpoint: pressure length mismatch at step {step}"
        );
        debug_assert_eq!(
            velocity_x.len(),
            self.velocity_x.len(),
            "GpuCheckpoint: velocity_x length mismatch at step {step}"
        );
        self.pressure.copy_from_slice(pressure);
        self.velocity_x.copy_from_slice(velocity_x);
        self.velocity_y.copy_from_slice(velocity_y);
        self.velocity_z.copy_from_slice(velocity_z);
        self.step = step;
    }
}

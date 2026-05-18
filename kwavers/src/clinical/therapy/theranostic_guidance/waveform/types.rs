//! Internal and public types for the 2-D acoustic waveform simulation.

use ndarray::Array2;

/// Output of the source-encoded adjoint RTM waveform simulation.
#[derive(Clone, Debug)]
pub struct WaveformSimulationResult {
    pub reconstruction: Array2<f64>,
    pub residual_energy: f64,
    pub observed_energy: f64,
    pub receiver_count: usize,
    pub time_steps: usize,
    pub dt_s: f64,
    pub model_name: &'static str,
    pub misfit_name: &'static str,
    pub misfit_scale: f32,
    pub objective_value: f64,
}

/// Output of the memory-bounded forward exposure simulation.
#[derive(Clone, Debug)]
pub struct PeakPressureExposureResult {
    pub exposure: Array2<f64>,
    pub raw_peak_pressure: Array2<f64>,
    pub source_count: usize,
    pub time_steps: usize,
    pub dt_s: f64,
    pub workspace_values: usize,
    pub model_name: &'static str,
    pub backend_name: &'static str,
    pub uses_hybrid_pstd_fdtd: bool,
}

/// Forward-run output: receiver traces and optional checkpoint snapshots.
#[derive(Clone, Debug)]
pub(super) struct WavefieldRun {
    pub(super) traces: Vec<f32>,
    /// Checkpoint snapshots stored every `interval` steps.
    pub(super) checkpoints: Option<Vec<f32>>,
    /// Checkpoint interval K = ceil(√T).
    pub(super) checkpoint_interval: usize,
}

/// CPML coefficient arrays along each axis (Komatitsch & Martin 2007, Eq. 8–12).
///
/// `b[i]` and `a[i]` are zero in the interior and nonzero in the PML strips.
#[derive(Clone, Debug)]
pub(super) struct CpmlCoeffs {
    pub(super) b_x: Vec<f32>,
    pub(super) a_x: Vec<f32>,
    pub(super) b_y: Vec<f32>,
    pub(super) a_y: Vec<f32>,
}

/// Full simulation grid with precomputed medium, CPML, and attenuation fields.
#[derive(Clone, Debug)]
pub(super) struct AcousticGrid {
    pub(super) nx: usize,
    pub(super) ny: usize,
    pub(super) dx_m: f64,
    pub(super) dt_s: f64,
    pub(super) time_steps: usize,
    pub(super) source_cells: Vec<usize>,
    pub(super) receiver_cells: Vec<usize>,
    pub(super) source_delays_s: Vec<f64>,
    /// Per-cell amplitude decay factor per time step (dimensionless, ≥0).
    pub(super) alpha_np_per_step: Vec<f32>,
    /// CPML coefficient arrays (Komatitsch & Martin 2007, §2).
    pub(super) cpml: CpmlCoeffs,
}

/// Checkpoint schedule for memory-efficient adjoint (Griewank 1992).
#[derive(Clone, Debug)]
pub(super) struct CheckpointSchedule {
    /// Save a snapshot every `interval` steps.
    pub(super) interval: usize,
    /// Total forward time steps.
    pub(super) time_steps: usize,
}

impl CheckpointSchedule {
    pub(super) fn new(time_steps: usize) -> Self {
        let interval = (time_steps as f64).sqrt().ceil() as usize;
        let interval = interval.max(1);
        Self {
            interval,
            time_steps,
        }
    }

    pub(super) fn slot_count(&self) -> usize {
        self.time_steps / self.interval + 1
    }

    pub(super) fn is_checkpoint(&self, step: usize) -> bool {
        step.is_multiple_of(self.interval)
    }

    pub(super) fn slot_for(&self, step: usize) -> usize {
        step / self.interval
    }

    pub(super) fn preceding_checkpoint(&self, target: usize) -> usize {
        (target / self.interval) * self.interval
    }
}

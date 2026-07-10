//! Internal and public types for the 2-D acoustic waveform simulation.

use leto::Array2;

/// Padded 2-D simulation domain that encompasses both the body slice and the
/// transducer aperture, surrounded by coupling water and CPML on the outer ring.
///
/// The body slice is embedded centred in a larger grid; `body_offset` and
/// `body_dims` describe the rectangular sub-region that corresponds to the
/// caller-visible body grid. Both `speed_baseline` and `speed_true` have
/// shape `(grid.nx, grid.ny)` (padded). Cells outside the body region carry
/// `SOUND_SPEED_WATER_SIM` (coupling water).
///
/// References:
/// - Treeby & Cox (2010), J. Acoust. Soc. Am. 128:2741 — k-Wave padded domain.
/// - Komatitsch & Martin (2007), Geophysics 72:SM155 — CPML outer strip.
#[derive(Clone, Debug)]
pub(super) struct PaddedSimulation {
    pub(super) grid: AcousticGrid,
    pub(super) speed_baseline: Array2<f64>,
    pub(super) speed_true: Array2<f64>,
    /// Body sub-region offset (in REFINED padded-grid cell units).
    pub(super) body_offset: (usize, usize),
    /// Body sub-region dimensions in REFINED cells:
    /// `body_dims = (nx_body * refinement, ny_body * refinement)`.
    pub(super) body_dims: (usize, usize),
    /// Caller-visible body dimensions (matches `prepared.sound_speed_m_s.shape()`).
    pub(super) body_dims_coarse: (usize, usize),
    /// Internal grid-refinement factor (refined cells per body cell along
    /// each axis).  Chosen so `λ / dx_refined ≥ 4` at the highest configured
    /// transmit frequency, which is the minimum for the 4th-order FD stencil
    /// to propagate without destructive numerical dispersion.
    pub(super) refinement: usize,
}

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
    /// Checkpoint interval selected from a replay-work target and memory budget.
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
    /// Primary source frequency used for Ricker wavelet injection.
    pub(super) source_frequency_hz: f64,
    /// Source amplitude scale: `P₀ / √(N_src)` (Pa).
    ///
    /// Dividing by √N ensures the total injected energy is independent of
    /// the element count so the forward and adjoint replay are amplitude-matched.
    pub(super) source_scale: f32,
    /// Optional precomputed per-time-step source amplitude (unit-normalised).
    ///
    /// When `Some`, every source cell injects `source_scale · waveform[step]`
    /// *simultaneously* (zero electronic delay) — used for the broadband
    /// passive cavitation emission. When `None`, the per-cell Ricker wavelet
    /// with `source_delays_s` focal-law delays is injected instead.
    pub(super) source_waveform: Option<Vec<f32>>,
}

/// Checkpoint schedule for memory-efficient adjoint (Griewank 1992).
///
/// ## Memory layout
///
/// Each checkpoint slot stores a consecutive **pair** of pressure fields
/// `(previous, current)` at the checkpoint time `t = slot * interval`.
/// Storing the pair is mandatory for the second-order-in-time wave equation:
/// a single snapshot only fixes `p(t)`, leaving `p(t-1)` (the "velocity")
/// unknown, which forces `fwd_prev = fwd_curr` during replay and introduces
/// an O(|p|) initialization error on every replay outside slot 0.
///
/// Buffer layout: `[prev₀ | curr₀ | prev₁ | curr₁ | … | prevₛ | currₛ]`
/// where each `|prev_s|` or `|curr_s|` block has `N = nx * ny` elements.
/// Total size: `2 * slot_count * N`.
#[derive(Clone, Debug)]
pub(super) struct CheckpointSchedule {
    /// Save a snapshot every `interval` steps.
    pub(super) interval: usize,
    /// Total forward time steps.
    pub(super) time_steps: usize,
}

impl CheckpointSchedule {
    pub(super) fn new(time_steps: usize, cell_count: usize) -> Self {
        const TARGET_REPLAY_INTERVAL: usize = 8;
        const MAX_CHECKPOINT_BYTES: usize = 256 * 1024 * 1024;
        const BYTES_PER_SLOT_CELL: usize = 2 * std::mem::size_of::<f32>();

        let sqrt_interval = (time_steps as f64).sqrt().ceil() as usize;
        let replay_interval = sqrt_interval.clamp(1, TARGET_REPLAY_INTERVAL);
        let max_slots = if cell_count == 0 {
            1
        } else {
            MAX_CHECKPOINT_BYTES / (BYTES_PER_SLOT_CELL * cell_count)
        };
        let budget_interval = if max_slots <= 1 {
            time_steps.max(1)
        } else {
            time_steps.div_ceil(max_slots - 1).max(1)
        };
        let interval = replay_interval.max(budget_interval);
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

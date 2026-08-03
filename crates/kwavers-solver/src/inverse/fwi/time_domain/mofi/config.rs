use crate::inverse::reconstruction::seismic::MisfitType;

use super::{
    nonrigid::{FfdConfig, FfdField},
    transform::RigidTransform,
};

/// MOFI optimisation settings.
#[derive(Debug, Clone, Copy)]
pub struct MofiConfig {
    /// Maximum outer iterations.
    pub max_iterations: usize,
    /// Initial line-search step in the balanced parameter space \[m\].
    pub initial_step_m: f64,
    /// Armijo sufficient-decrease constant `c₁ ∈ (0, 1)`.
    pub armijo_c1: f64,
    /// Maximum Armijo backtracking halvings per iteration.
    pub max_line_search: usize,
    /// Sound speed assigned where the transformed template maps outside the grid
    /// (the background medium, e.g. water) [m/s].
    pub background_c: f64,
    /// Relative-misfit-change convergence tolerance.
    pub tolerance: f64,
}

impl Default for MofiConfig {
    fn default() -> Self {
        Self {
            max_iterations: 60,
            initial_step_m: 5e-3,
            armijo_c1: 1e-4,
            max_line_search: 15,
            background_c: 1500.0,
            tolerance: 1e-4,
        }
    }
}

/// Result of a MOFI alignment.
#[derive(Debug, Clone, Copy)]
pub struct MofiResult {
    /// Recovered rigid transform aligning the template to the data.
    pub transform: RigidTransform,
    /// Outer iterations performed.
    pub iterations: usize,
    /// Misfit at `φ = 0` (template untransformed).
    pub initial_misfit: f64,
    /// Misfit at the recovered transform.
    pub final_misfit: f64,
}

/// One stage of a MOFI misfit homotopy.
///
/// Each stage runs [`align_from`] with the data misfit set to `misfit_type` and
/// an optional zero-phase low-pass corner `band_limit_hz`, warm-started from the
/// previous stage's transform. Annealing from a convex, cycle-skip-robust misfit
/// (Wasserstein/optimal transport, envelope, correlation) toward L2 widens the
/// capture basin for large initial misalignments.
#[derive(Debug, Clone, Copy)]
pub struct MofiStage {
    /// Data misfit functional for this stage.
    pub misfit_type: MisfitType,
    /// Optional low-pass corner \[Hz\] applied to traces this stage (multiscale).
    pub band_limit_hz: Option<f64>,
    /// Optimisation settings for this stage.
    pub config: MofiConfig,
}

/// Coarse global pose-search settings.
///
/// The search brute-forces a regular grid over rotation and translation,
/// evaluating the data misfit at each candidate by a cheap sensor-only forward
/// solve. Being a global sample (not a gradient step), it is robust to arbitrarily
/// large misalignment and provides a starting pose inside the local basin for
/// [`align_from`] / [`align_homotopy`].
#[derive(Debug, Clone, Copy)]
pub struct CoarseSearchConfig {
    /// Half-range of the rotation sweep \[rad\]; candidates span `[−θ_max, θ_max]`.
    pub theta_max_rad: f64,
    /// Number of rotation samples (≥ 1).
    pub theta_steps: usize,
    /// Half-range of each translation axis \[m\]; candidates span `[−δ_max, δ_max]`.
    pub delta_max_m: f64,
    /// Number of samples per translation axis (≥ 1).
    pub delta_steps: usize,
    /// Background sound speed for out-of-domain template pixels [m/s].
    pub background_c: f64,
}

/// Result of a pose + sound-speed-calibration alignment.
#[derive(Debug, Clone, Copy)]
pub struct MofiCalibratedResult {
    /// Recovered rigid transform.
    pub transform: RigidTransform,
    /// Recovered global template sound-speed contrast scale `α`: the calibrated
    /// model is `c = c_bg + α·(c_template − c_bg)` (α = 1 leaves the template
    /// unchanged). Corrects a systematic CT→sound-speed mapping error that rigid
    /// alignment alone cannot.
    pub speed_scale: f64,
    /// Misfit at `φ = 0, α = 1`.
    pub initial_misfit: f64,
    /// Misfit at the recovered `(φ, α)`.
    pub final_misfit: f64,
    /// Outer block-coordinate iterations performed.
    pub outer_iterations: usize,
}

/// Full multi-pathway alignment pipeline configuration.
#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    /// Coarse global pose-search grid.
    pub coarse: CoarseSearchConfig,
    /// Misfit used for the coarse search (use an arrival-time-sensitive, cycle-
    /// skip-robust functional such as [`MisfitType::Wasserstein`]).
    pub search_misfit: MisfitType,
    /// Rigid/calibration optimisation settings.
    pub rigid: MofiConfig,
    /// Block-coordinate rounds for pose + speed calibration (0 ⇒ skip calibration,
    /// rigid pose only).
    pub calibration_outer: usize,
    /// Non-rigid FFD refinement settings (`n_ctrl_* < 2` ⇒ skip non-rigid).
    pub ffd: FfdConfig,
}

/// Result of the full alignment pipeline.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Recovered rigid pose.
    pub transform: RigidTransform,
    /// Recovered sound-speed contrast scale (1.0 if calibration skipped).
    pub speed_scale: f64,
    /// Recovered non-rigid deformation (zero lattice if non-rigid skipped).
    pub ffd: FfdField,
    /// Misfit at the unaligned template.
    pub initial_misfit: f64,
    /// Misfit at the fully aligned model.
    pub final_misfit: f64,
}

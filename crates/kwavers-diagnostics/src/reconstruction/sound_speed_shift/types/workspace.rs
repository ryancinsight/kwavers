//! Speed-shift solver workspace state.

/// Reusable scratch buffers for speed-of-sound shift reconstruction.
///
/// A workspace owns all PCG/ISTA work vectors used by dense and sparse
/// reconstructions. Reusing one workspace across calls with compatible or
/// smaller geometry preserves vector capacities and avoids repeated heap
/// allocation in iterative solves.
#[derive(Clone, Debug, Default)]
pub struct SoundSpeedShiftWorkspace {
    pub(in super::super) rhs: Vec<f64>,
    pub(in super::super) diagonal: Vec<f64>,
    pub(in super::super) solution: Vec<f64>,
    pub(in super::super) normal_solution: Vec<f64>,
    pub(in super::super) residual: Vec<f64>,
    pub(in super::super) preconditioned: Vec<f64>,
    pub(in super::super) direction: Vec<f64>,
    pub(in super::super) normal_direction: Vec<f64>,
    pub(in super::super) row: Vec<f64>,
    pub(in super::super) laplacian: Vec<f64>,
    pub(in super::super) prediction: Vec<f64>,
    pub(in super::super) previous_solution: Vec<f64>,
    pub(in super::super) power_vector: Vec<f64>,
    pub(in super::super) power_normal: Vec<f64>,
    pub(in super::super) objective_history: Vec<f64>,
}

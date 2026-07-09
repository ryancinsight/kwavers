use leto::{
    Array2,
    Array3,
};

/// Full output of one standing-wave suppression run.
///
/// `swi_history`, `focal_pressure_history`, and `objective_history` each
/// have length `n_iterations + 1` (index 0 = initial DAS state).
/// `snapshot_fields_re` and `snapshot_fields_im` have shape
/// `(n_snapshots, nx, ny)` and correspond to `snapshot_iterations`.
#[derive(Debug)]
pub struct StandingWaveOptResult {
    // Grid
    pub nx: usize,
    pub ny: usize,
    pub dx_m: f64,
    pub frequency_hz: f64,
    pub n_elements: usize,
    pub element_ys: Vec<usize>,
    pub source_x: usize,
    pub focus_x: usize,
    pub focus_y: usize,
    pub reflector_x_start: usize,
    pub reflector_x_end: usize,
    pub pml_cells: usize,

    // Medium
    pub sound_speed_map: Array2<f32>,

    // Iteration time series
    pub swi_history: Vec<f64>,
    pub focal_pressure_history: Vec<f64>,
    pub objective_history: Vec<f64>,

    // Phases
    pub initial_phases: Vec<f64>,
    pub final_phases: Vec<f64>,

    // Field snapshots for visualisation
    pub snapshot_iterations: Vec<usize>,
    pub snapshot_fields_re: Array3<f32>,
    pub snapshot_fields_im: Array3<f32>,

    // Initial and final fields (always included)
    pub initial_field_re: Array2<f32>,
    pub initial_field_im: Array2<f32>,
    pub final_field_re: Array2<f32>,
    pub final_field_im: Array2<f32>,

    // Scalar diagnostics
    pub swi_weight: f64,
    pub focal_weight: f64,
    pub focal_pressure_ref_pa: f64,
}

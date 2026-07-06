//! `run_standing_wave_suppression` pyfunction and result serialisation.

use kwavers_therapy::therapy::theranostic_guidance::{
    run_standing_wave_suppression, StandingWaveOptConfig,
};
use numpy::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction(name = "run_standing_wave_suppression")]
#[pyo3(signature = (
    nx = 128,
    ny = 64,
    dx_m = 7.5e-4,
    frequency_hz = 250_000.0,
    cfl = 0.25,
    pml_cells = 10,
    c_ref_m_s = 1540.0,
    c_layer_m_s = 2000.0,
    rho_ref_kg_m3 = 1000.0,
    rho_layer_kg_m3 = 1500.0,
    layer_x_start = 90,
    layer_x_end = 96,
    source_x = 11,
    focus_x = 68,
    focus_y = 32,
    n_elements = 12,
    element_y_min = 12,
    element_y_max = 52,
    focal_radius_cells = 3,
    burst_cycles = 5.0,
    accum_skip_cycles = 2.0,
    swi_axis_half_width = 2,
    n_opt_iter = 25,
    swi_weight = 0.70,
    focal_weight = 0.30,
    grad_delta_rad = 0.05,
    armijo_c1 = 0.01,
    line_search_alpha0 = 1.0,
    line_search_beta = 0.5,
    line_search_max = 12,
    n_snapshots = 5
))]
#[allow(clippy::too_many_arguments)]
pub fn run_standing_wave_suppression_py<'py>(
    py: Python<'py>,
    nx: usize,
    ny: usize,
    dx_m: f64,
    frequency_hz: f64,
    cfl: f64,
    pml_cells: usize,
    c_ref_m_s: f64,
    c_layer_m_s: f64,
    rho_ref_kg_m3: f64,
    rho_layer_kg_m3: f64,
    layer_x_start: usize,
    layer_x_end: usize,
    source_x: usize,
    focus_x: usize,
    focus_y: usize,
    n_elements: usize,
    element_y_min: usize,
    element_y_max: usize,
    focal_radius_cells: usize,
    burst_cycles: f64,
    accum_skip_cycles: f64,
    swi_axis_half_width: usize,
    n_opt_iter: usize,
    swi_weight: f64,
    focal_weight: f64,
    grad_delta_rad: f64,
    armijo_c1: f64,
    line_search_alpha0: f64,
    line_search_beta: f64,
    line_search_max: usize,
    n_snapshots: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let config = StandingWaveOptConfig {
        nx,
        ny,
        dx_m,
        frequency_hz,
        cfl,
        pml_cells,
        c_ref_m_s,
        rho_ref_kg_m3,
        c_layer_m_s,
        rho_layer_kg_m3,
        layer_x_start,
        layer_x_end,
        source_x,
        focus_x,
        focus_y,
        n_elements,
        element_y_min,
        element_y_max,
        focal_radius_cells,
        burst_cycles,
        accum_skip_cycles,
        swi_axis_half_width,
        n_opt_iter,
        swi_weight,
        focal_weight,
        grad_delta_rad,
        armijo_c1,
        line_search_alpha0,
        line_search_beta,
        line_search_max,
        n_snapshots,
    };

    let result = py.detach(|| run_standing_wave_suppression(&config));

    let dict = PyDict::new(py);

    // Grid / geometry scalars
    dict.set_item("nx", result.nx)?;
    dict.set_item("ny", result.ny)?;
    dict.set_item("dx_m", result.dx_m)?;
    dict.set_item("frequency_hz", result.frequency_hz)?;
    dict.set_item("n_elements", result.n_elements)?;
    dict.set_item("source_x", result.source_x)?;
    dict.set_item("focus_x", result.focus_x)?;
    dict.set_item("focus_y", result.focus_y)?;
    dict.set_item("reflector_x_start", result.reflector_x_start)?;
    dict.set_item("reflector_x_end", result.reflector_x_end)?;
    dict.set_item("pml_cells", result.pml_cells)?;

    // Medium
    dict.set_item("sound_speed_map", result.sound_speed_map.to_pyarray(py))?;

    // Element positions
    let eys: Vec<i64> = result.element_ys.iter().map(|&v| v as i64).collect();
    dict.set_item("element_ys", ndarray::Array1::from(eys).to_pyarray(py))?;

    // Time series
    dict.set_item(
        "swi_history",
        ndarray::Array1::from(result.swi_history).to_pyarray(py),
    )?;
    dict.set_item(
        "focal_pressure_history",
        ndarray::Array1::from(result.focal_pressure_history).to_pyarray(py),
    )?;
    dict.set_item(
        "objective_history",
        ndarray::Array1::from(result.objective_history).to_pyarray(py),
    )?;

    // Phases
    dict.set_item(
        "initial_phases",
        ndarray::Array1::from(result.initial_phases).to_pyarray(py),
    )?;
    dict.set_item(
        "final_phases",
        ndarray::Array1::from(result.final_phases).to_pyarray(py),
    )?;

    // Field snapshots
    let snap_iters: Vec<i64> = result
        .snapshot_iterations
        .iter()
        .map(|&v| v as i64)
        .collect();
    dict.set_item(
        "snapshot_iterations",
        ndarray::Array1::from(snap_iters).to_pyarray(py),
    )?;
    dict.set_item(
        "snapshot_fields_re",
        result.snapshot_fields_re.to_pyarray(py),
    )?;
    dict.set_item(
        "snapshot_fields_im",
        result.snapshot_fields_im.to_pyarray(py),
    )?;

    // Initial and final fields
    dict.set_item("initial_field_re", result.initial_field_re.to_pyarray(py))?;
    dict.set_item("initial_field_im", result.initial_field_im.to_pyarray(py))?;
    dict.set_item("final_field_re", result.final_field_re.to_pyarray(py))?;
    dict.set_item("final_field_im", result.final_field_im.to_pyarray(py))?;

    // Diagnostics
    dict.set_item("swi_weight", result.swi_weight)?;
    dict.set_item("focal_weight", result.focal_weight)?;
    dict.set_item("focal_pressure_ref_pa", result.focal_pressure_ref_pa)?;

    Ok(dict)
}


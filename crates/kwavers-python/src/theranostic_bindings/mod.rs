//! Python bindings for same-device therapy/imaging inverse simulations.

mod abdominal3d;
mod helpers;
mod inverse;
mod nonlinear3d;
mod standing_wave;
mod transcranial_benchmark;
mod transcranial_focused_bowl;
mod transcranial_fus;

pub use abdominal3d::plan_abdominal_array_placement_from_ritk_ct;
pub use inverse::run_theranostic_inverse_from_ritk;
pub use nonlinear3d::run_theranostic_nonlinear_3d_from_ritk;
pub use standing_wave::run_standing_wave_suppression_py;
pub use transcranial_benchmark::run_transcranial_skull_adaptive_benchmark_from_ritk_ct;
pub use transcranial_focused_bowl::plan_transcranial_focused_bowl_placement_from_ritk_ct;
pub use transcranial_fus::{
    bbb_opening_from_subspots_py, gbm_subspot_raster_py, run_transcranial_fus_planning_from_arrays,
    run_transcranial_fus_planning_from_ritk_ct, transcranial_pennes_thermal_dose_py,
};

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_theranostic_inverse_from_ritk, m)?)?;
    m.add_function(wrap_pyfunction!(run_theranostic_nonlinear_3d_from_ritk, m)?)?;
    m.add_function(wrap_pyfunction!(
        plan_transcranial_focused_bowl_placement_from_ritk_ct,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(run_standing_wave_suppression_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        plan_abdominal_array_placement_from_ritk_ct,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        run_transcranial_fus_planning_from_ritk_ct,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        run_transcranial_skull_adaptive_benchmark_from_ritk_ct,
        m
    )?)?;
    // Array-based entry points (accept pre-loaded numpy arrays).
    m.add_function(wrap_pyfunction!(
        run_transcranial_fus_planning_from_arrays,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(gbm_subspot_raster_py, m)?)?;
    m.add_function(wrap_pyfunction!(bbb_opening_from_subspots_py, m)?)?;
    m.add_function(wrap_pyfunction!(transcranial_pennes_thermal_dose_py, m)?)?;
    Ok(())
}

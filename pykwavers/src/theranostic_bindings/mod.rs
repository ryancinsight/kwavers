//! Python bindings for same-device therapy/imaging inverse simulations.

mod helpers;
mod helmet;
mod inverse;
mod nonlinear3d;
mod standing_wave;

pub use nonlinear3d::run_theranostic_nonlinear_3d_from_ritk;
pub use inverse::run_theranostic_inverse_from_ritk;
pub use helmet::plan_brain_helmet_placement_from_ritk_ct;
pub use standing_wave::run_standing_wave_suppression_py;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_theranostic_inverse_from_ritk, m)?)?;
    m.add_function(wrap_pyfunction!(run_theranostic_nonlinear_3d_from_ritk, m)?)?;
    m.add_function(wrap_pyfunction!(
        plan_brain_helmet_placement_from_ritk_ct,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(run_standing_wave_suppression_py, m)?)?;
    Ok(())
}

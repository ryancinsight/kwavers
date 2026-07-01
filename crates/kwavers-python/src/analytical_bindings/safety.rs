//! PyO3 bindings for `kwavers_physics::analytical::safety`.

mod damage;
mod mechanical;
mod thermal;

use kwavers_physics::analytical::safety;
use pyo3::prelude::*;

pub use damage::{
    arrhenius_cumulative, arrhenius_damage_integral, arrhenius_kill_probability,
    arrhenius_steady_kill_probability, combined_kill_probability,
};
pub use mechanical::{
    mechanical_index, mechanical_index_cavitation_risk, mechanical_index_field,
    mechanical_index_frequency_sweep,
};
pub use thermal::{
    cem43_cumulative, closed_loop_cem43_fixture, thermal_index_bone, thermal_index_cranial,
    thermal_index_soft_tissue,
};

/// Return the FDA ISPTA diagnostic-ultrasound limit (720 mW/cm²).
///
/// Returns:
///     ISPTA limit [mW/cm²].
#[pyfunction]
pub fn fda_ispta_limit_mw_cm2() -> PyResult<f64> {
    Ok(safety::fda_ispta_limit_mw_cm2())
}

/// Return the FDA ISPPA diagnostic-ultrasound limit (190 W/cm²).
///
/// Returns:
///     ISPPA limit [W/cm²].
#[pyfunction]
pub fn fda_isppa_limit_w_cm2() -> PyResult<f64> {
    Ok(safety::fda_isppa_limit_w_cm2())
}

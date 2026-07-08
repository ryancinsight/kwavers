//! PyO3 bindings for Ali et al. 2025 breast UST frequency-domain FWI.
//!
//! The binding layer performs only Python/Rust data conversion. Geometry and
//! paper identities stay in `physics`, numerical propagation stays in `solver`,
//! and clinical image metadata stays in `clinical::imaging::reconstruction`.
//!
//! Module topology:
//! - `array_config`   — `PyMultiRowRingArray` class
//! - `fwi_config`     — `PyFrequencyDomainFwiConfig` class
//! - `inversion`      — `PyFrequencyObservation` class + top-level FWI functions
//! - `helpers`        — private parsing/conversion utilities
//! - `dataset`, `diagnostics`, `direct_field`, `finite_window`,
//!   `operator_equivalence`, `phantom`, `reduction` — domain submodules

mod array_config;
pub mod complex_compat;
mod dataset;
mod diagnostics;
mod direct_field;
mod finite_window;
mod fwi_config;
mod helpers;
mod inversion;
mod operator_equivalence;
mod phantom;
mod reduction;

pub use array_config::PyMultiRowRingArray;
pub use dataset::{generate_breast_fwi_pstd_dataset, PyBreastFwiPstdDatasetConfig};
pub use fwi_config::PyFrequencyDomainFwiConfig;
pub use inversion::{
    ali_2025_breast_fwi_frequency_sweep_hz, invert_breast_fwi,
    simulate_breast_fwi_frequency_observation, snap_breast_fwi_array_to_grid,
    PyFrequencyObservation,
};
pub use phantom::load_ali_2025_breast_fwi_phantom;

use pyo3::prelude::*;
use pyo3::types::PyModule;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiRowRingArray>()?;
    m.add_class::<PyFrequencyDomainFwiConfig>()?;
    m.add_class::<PyFrequencyObservation>()?;
    m.add_class::<PyBreastFwiPstdDatasetConfig>()?;
    m.add_function(wrap_pyfunction!(ali_2025_breast_fwi_frequency_sweep_hz, m)?)?;
    m.add_function(wrap_pyfunction!(
        simulate_breast_fwi_frequency_observation,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(snap_breast_fwi_array_to_grid, m)?)?;
    m.add_function(wrap_pyfunction!(load_ali_2025_breast_fwi_phantom, m)?)?;
    m.add_function(wrap_pyfunction!(generate_breast_fwi_pstd_dataset, m)?)?;
    direct_field::register(m)?;
    diagnostics::register(m)?;
    finite_window::register(m)?;
    operator_equivalence::register(m)?;
    reduction::register(m)?;
    m.add_function(wrap_pyfunction!(invert_breast_fwi, m)?)?;
    Ok(())
}

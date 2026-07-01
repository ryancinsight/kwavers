//! Bubble dynamics and thermal dosimetry Python bindings.
//!
//! Provides PyO3 functions for:
//! - Rayleigh-Plesset ODE integration (incompressible bubble dynamics)
//! - Keller-Miksis ODE integration (compressible bubble dynamics)
//! - Keller-Herring ODE integration (typed KH wrapper, same solver family)
//! - CEM43 thermal dose (Sapareto & Dewey 1984)
//! - Arrhenius damage integral (Henriques & Moritz 1947)
//! - Hodgkin-Huxley-like neural response model (Yoo et al. 2022 temperature-coupling)
//!
//! All ODE integrators use fixed-step classical RK4. All physics executes in
//! Rust; Python receives numpy arrays only.
//!
//! # References
//!
//! - Rayleigh (1917) Phil. Mag. 34:94; Plesset (1949) J. Appl. Mech. 16:277
//! - Keller & Miksis (1980) J. Acoust. Soc. Am. 68(2):628
//! - Sapareto & Dewey (1984) Int. J. Radiat. Oncol. Biol. Phys. 10(6):787
//! - Henriques & Moritz (1947) Am. J. Pathol. 23:695
//! - Yoo et al. (2022) Nature Neuroscience 25:1557

mod arrhenius;
mod cem43;
mod gilmore;
mod hodgkin_huxley;
mod keller_miksis;
mod rayleigh_plesset;

pub use arrhenius::compute_arrhenius_damage;
pub use cem43::{cem43_at_temperatures, compute_cem43};
pub use gilmore::solve_gilmore;
pub use hodgkin_huxley::solve_hodgkin_huxley_like;
pub use keller_miksis::{solve_keller_herring, solve_keller_miksis};
pub use rayleigh_plesset::solve_rayleigh_plesset;

use pyo3::prelude::*;

pub fn register_bubble(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_rayleigh_plesset, m)?)?;
    m.add_function(wrap_pyfunction!(solve_keller_miksis, m)?)?;
    m.add_function(wrap_pyfunction!(solve_keller_herring, m)?)?;
    m.add_function(wrap_pyfunction!(solve_gilmore, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cem43, m)?)?;
    m.add_function(wrap_pyfunction!(cem43_at_temperatures, m)?)?;
    m.add_function(wrap_pyfunction!(compute_arrhenius_damage, m)?)?;
    m.add_function(wrap_pyfunction!(solve_hodgkin_huxley_like, m)?)?;
    Ok(())
}

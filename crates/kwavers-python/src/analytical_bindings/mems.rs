//! PyO3 bindings for `kwavers_transducer::mems` (CMUT / PMUT / IVUS comparison).
//!
//! Physics in Rust; these thin wrappers expose the scalar models so the
//! `ch33_cmut_vs_pmut.py` figure script can plot without re-implementing physics.

mod cmut;
mod comparison;
mod helpers;
mod plate;
mod pmut;

pub use cmut::{
    cmut_collapse_voltage, cmut_coupling_k2, cmut_flex_gap_derating, cmut_fractional_bandwidth,
    cmut_max_output_pressure, cmut_resonance_immersion, cmut_self_heating,
};
pub use comparison::{ivus_figure_of_merit, therapy_figure_of_merit};
pub use plate::{mems_clamped_plate_resonance, mems_immersion_resonance};
pub use pmut::{
    pmut_coupling_k2, pmut_fractional_bandwidth, pmut_max_output_pressure,
    pmut_resonance_immersion, pmut_self_heating,
};

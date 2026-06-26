//! DigiKey-sourced component database for HV ultrasound transducer drivers.
//!
//! Provides parametric data for modern in-stock HV pulser ICs, enabling the
//! co-optimiser to compare alternative component selections and optimise the
//! driver topology against cost, power, thermal, and routing constraints.
//!
//! # Component tiers
//!
//! | Part | Channels | Voltage | Topology | DigiKey status 2026 |
//! |---|---|---|---|---|
//! | HV7355 (Microchip) | 8 | ±150V | 3-level class-D | ✓ current design target |
//! | STHVUP32 (ST) | 32 | ±100V | 5-level + beamformer | ✓ stocked |
//! | MAX14815 (Analog) | 8 | 200Vpp | 5-level + T/R | ✓ stocked evaluation |
//! | STHV748S (ST) | 4 | ±90V | 3-level rugged | ✓ stocked |
//! | MD1715 (Microchip) | 2 | 200V | 5-level + ext FET | ✓ evaluation |
//! | HV7360 (Microchip) | 1 | ±100V | AC-coupled | ✓ stocked |
//!
//! Evidence tier: parametric data from manufacturer datasheets verified against
//! DigiKey stock availability; value-semantic unit tests ground the comparison
//! scores against hand-computed reference values.
//!
//! # Slice layout
//!
//! Carved by **role** (Phase 4i). Plain backticks name the slice-private submodules; the public
//! items each hosts stay clickable.
//! * `pulser_ic` — the [`PulserIc`] datasheet record, the [`StockStatus`] enum, and the derived
//!   per-IC property accessors.
//! * `catalog` — the static pulser-IC table behind [`available_pulsers`].
//! * `dcdc` — the [`DcDcModule`] HV-bias-rail converter record + its static table
//!   ([`available_dcdc_modules`]).
//! * `compare` — [`PulserComparison`], [`compare_pulsers`], and the 96-channel recommendation.
//!
//! The datasheet tables are compile-time `static` slices (every field is `&'static str` / scalar /
//! `&'static [f64]`), so `available_pulsers`/`available_dcdc_modules` return `&'static [_]` with **no
//! per-call heap allocation** — the const/zero-cost form for analytically-fixed data.

mod catalog;
mod compare;
mod dcdc;
mod pulser_ic;

#[cfg(test)]
mod tests;

pub use catalog::available_pulsers;
pub use compare::{compare_pulsers, recommend_96ch_architecture, PulserComparison};
pub use dcdc::{available_dcdc_modules, DcDcModule};
pub use pulser_ic::{
    board_area_per_n_channels_mm2, decoupling_per_ch_uf, output_pin_capacitance_pf, pkg_area_mm2,
    signal_pins_per_ch, supply_pins_per_ch, PulserIc, StockStatus,
};

//! Thermodynamic closures for the Keller-Miksis bubble model.
//!
//! Responsibilities are split by physical law:
//! - `phase`: water saturation pressure and latent heat laws.
//! - `eos`: Van der Waals pressure closure.
//! - `transfer`: vapor mass transfer.
//! - `temperature`: bubble temperature ODE update.

mod eos;
mod phase;
mod temperature;
mod transfer;

#[cfg(test)]
mod tests;

pub(crate) use eos::calculate_vdw_pressure;
pub use phase::{latent_heat_water_j_per_kg, p_sat_water_pa};
pub(crate) use temperature::update_temperature;
pub(crate) use transfer::update_mass_transfer;

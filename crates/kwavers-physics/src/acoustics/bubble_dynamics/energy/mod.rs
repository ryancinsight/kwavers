//! Energy Balance Model for Bubble Dynamics
//!
//! This module implements a comprehensive energy balance equation for bubble temperature
//! evolution, including work done by pressure-volume changes, heat transfer, and
//! latent heat from mass transfer.

mod calculator;
mod chemical_reaction;
mod heat_transfer;
mod plasma_ionization;
mod radiation;

#[cfg(test)]
mod tests;

pub use calculator::EnergyBalanceCalculator;
pub use heat_transfer::update_temperature_energy_balance;

//! Chemical and molecular constants

/// Oxygen diffusion coefficient in tissue [m²/s]
pub const OXYGEN_DIFFUSION_TISSUE: f64 = 2e-9;

/// Oxygen diffusion coefficient in water [m²/s]
pub const OXYGEN_DIFFUSION_WATER: f64 = 2.1e-9;

/// Henry's law constant for oxygen in water at 25°C [mol/(m³·Pa)]
pub const OXYGEN_HENRY_CONSTANT: f64 = 1.3e-5;

/// Oxygen saturation concentration in blood [mol/m³]
pub const OXYGEN_SATURATION_BLOOD: f64 = 8.8;

/// pH of blood
pub const BLOOD_PH: f64 = 7.4;

/// Ionic strength of physiological saline [mol/L]
pub const PHYSIOLOGICAL_IONIC_STRENGTH: f64 = 0.15;

/// Molecular weights [g/mol]
pub const HYDROXYL_RADICAL_WEIGHT: f64 = 17.008;
pub const HYDROGEN_PEROXIDE_WEIGHT: f64 = 34.014;
pub const SUPEROXIDE_WEIGHT: f64 = 32.00;
pub const SINGLET_OXYGEN_WEIGHT: f64 = 32.00;
pub const NITRIC_OXIDE_WEIGHT: f64 = 30.01;
pub const PEROXYNITRITE_WEIGHT: f64 = 62.00;
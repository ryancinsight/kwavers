//! High-intensity focused ultrasound physics.
//!
//! The module exposes two independent responsibilities:
//! - acoustic field synthesis for a focused aperture;
//! - CEM43 thermal-dose accumulation.

mod field;
mod thermal_dose;

#[cfg(test)]
mod tests;

pub use field::{compute_intensity_field, compute_pressure_field};
pub use thermal_dose::HifuThermalDose;

// Acoustic-Elastic Coupling Matrix Assembly
//
// FDTD discrete coupling at fluid-solid interfaces.
//
// ## References
//
// - Zienkiewicz et al. (2013). *The Finite Element Method*, 7th ed. §12.3.
// - de Hoop (1995). *Handbook of Radiation and Scattering of Waves*.

pub mod coupler;
pub mod terms;
#[cfg(test)]
mod tests;

pub use coupler::{stability_dt, AcousticElasticCoupler};
pub use terms::CouplingTerms;

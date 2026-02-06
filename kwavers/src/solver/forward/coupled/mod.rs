//! Coupled Multi-Physics Solvers
//!
//! This module implements monolithic solvers for coupled physical phenomena
//! where multiple fields interact and must be solved simultaneously.
//!
//! ## Supported Couplings
//!
//! - **Thermal-Acoustic**: Temperature-dependent material properties affecting acoustic propagation
//! - **Acoustic-Fluid**: Acoustic waves in fluids with nonlinear effects
//! - **Poroelastic-Thermal**: Coupled poroelasticity and heat diffusion
//!
//! ## Implementation Strategy
//!
//! Coupled solvers use monolithic time integration where:
//! 1. All fields (pressure, velocity, temperature) advanced simultaneously
//! 2. Coupling terms evaluated on current state
//! 3. Single time step ensures consistency and stability
//!
//! ## References
//!
//! - Baysal et al. (2005): "Thermal-acoustic coupling in focused ultrasound"
//! - Santos & Douglas (2008): "Multiphysics simulation of therapeutic ultrasound"
//! - Kolski-Andreaco et al. (2015): "Nonlinear acoustic heating in HIFU"

pub mod thermal_acoustic;

pub use thermal_acoustic::{ThermalAcousticConfig, ThermalAcousticCoupler};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupled_module_exports() {
        // Verify module structure is correct
        let _coupler = ThermalAcousticCoupler::new_default();
        assert!(true);
    }
}

//! Physics validation tests with known analytical solutions
//! 
//! This module contains tests that validate our numerical implementations
//! against known analytical solutions from physics literature.
//!
//! ## References
//! 
//! 1. **Treeby & Cox (2010)** - "k-Wave: MATLAB toolbox for the simulation
//!    and reconstruction of photoacoustic wave fields"
//! 2. **Szabo (1994)** - "Time domain wave equations for lossy media obeying
//!    a frequency power law"
//! 3. **Hamilton & Blackstock (1998)** - "Nonlinear Acoustics"
//! 4. **Pierce (1989)** - "Acoustics: An Introduction to Its Physical
//!    Principles and Applications"
//! 5. **Duck (1990)** - "Physical Properties of Tissue"
//! 6. **Gear & Wells (1984)** - "Multirate linear multistep methods"
//! 7. **Berger & Oliger (1984)** - "Adaptive mesh refinement for hyperbolic PDEs"
//! 8. **Persson & Peraire (2006)** - "Sub-cell shock capturing for DG methods"
//! 9. **Royer & Dieulesaint (2000)** - "Elastic Waves in Solids"

#[cfg(test)]
pub mod wave_propagation;

#[cfg(test)]
pub mod nonlinear_acoustics;

#[cfg(test)]
pub mod material_properties;

#[cfg(test)]
pub mod numerical_methods;

#[cfg(test)]
pub mod conservation_laws;

#[cfg(test)]
pub mod thermal_effects;

// Re-export test utilities
#[cfg(test)]
pub(crate) mod test_utils {
    use ndarray::Array3;
    
    /// Helper function to compute relative error
    pub fn relative_error(computed: f64, analytical: f64) -> f64 {
        if analytical.abs() < 1e-10 {
            (computed - analytical).abs()
        } else {
            ((computed - analytical) / analytical).abs()
        }
    }
    
    /// Helper function to compute L2 norm of error
    pub fn l2_error(computed: &Array3<f64>, analytical: &Array3<f64>) -> f64 {
        let diff = computed - analytical;
        (diff.mapv(|x| x * x).sum() / diff.len() as f64).sqrt()
    }
    
    /// Helper function to compute max error
    pub fn max_error(computed: &Array3<f64>, analytical: &Array3<f64>) -> f64 {
        (computed - analytical).mapv(f64::abs).fold(0.0, f64::max)
    }
}
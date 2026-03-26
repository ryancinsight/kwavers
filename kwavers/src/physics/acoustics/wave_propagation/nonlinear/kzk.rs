//! Khokhlov-Zabolotskaya-Kuznetsov (KZK) Equation
//!
//! # Description
//! Placeholder trait boundary and module for the 3D KZK implementation.
//!
//! The KZK equation models focused acoustic beams and accounts for:
//! 1. Nonlinearity (waveform distortion, harmonics)
//! 2. Diffraction (beam spreading, focusing)
//! 3. Absorption (thermoviscous losses)
//!
//! It is the gold standard for modeling HIFU fields.

pub trait KZKSolver {
    /// Advance the acoustic field by one spatial step down the propagation axis (z)
    fn step_forward(&mut self, dz: f64);
    
    /// Retrieve the acoustic pressure field at the current z-plane
    fn current_field(&self) -> ndarray::Array2<f64>;
}

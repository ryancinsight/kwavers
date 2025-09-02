//! KZK (Khokhlov-Zabolotskaya-Kuznetsov) Equation Implementation
//!
//! The KZK equation is a parabolic approximation for directional sound beams,
//! widely used in medical ultrasound for modeling focused transducers.
//!
//! References:
//! - Lee & Hamilton (1995) "Parametric array in air"
//! - Aanonsen et al. (1984) "Distortion and harmonic generation in the nearfield"
//! - Jing et al. (2007) "Evaluation of a wave-vector-frequency-domain method"
//!
//! The KZK equation in the time domain:
//! ∂²p/∂z∂τ = (c₀/2)∇⊥²p + (δ/2c₀³)∂³p/∂τ³ + (β/2ρ₀c₀³)∂²p²/∂τ²
//!
//! Where:
//! - z: axial coordinate (beam propagation direction)
//! - τ = t - z/c₀: retarded time
//! - ∇⊥²: transverse Laplacian (∂²/∂x² + ∂²/∂y²)
//! - δ: diffusivity of sound
//! - β: coefficient of nonlinearity

pub mod absorption;
pub mod diffraction;
pub mod diffraction_corrected;
pub mod nonlinearity;
pub mod solver;
pub mod validation;

pub use solver::KZKSolver;

/// KZK configuration parameters
#[derive(Debug, Clone)]
pub struct KZKConfig {
    /// Grid size in x direction (transverse)
    pub nx: usize,
    /// Grid size in y direction (transverse)
    pub ny: usize,
    /// Grid size in z direction (axial)
    pub nz: usize,
    /// Grid spacing in x and y (m)
    pub dx: f64,
    /// Grid spacing in z (m)
    pub dz: f64,
    /// Time step (s)
    pub dt: f64,
    /// Number of time steps
    pub nt: usize,
    /// Sound speed (m/s)
    pub c0: f64,
    /// Density (kg/m³)
    pub rho0: f64,
    /// Nonlinearity coefficient B/A
    pub beta: f64,
    /// Attenuation coefficient (Np/m/MHz^y)
    pub alpha0: f64,
    /// Attenuation power law exponent
    pub alpha_power: f64,
    /// Enable diffraction effects
    pub include_diffraction: bool,
    /// Enable absorption
    pub include_absorption: bool,
    /// Enable nonlinearity
    pub include_nonlinearity: bool,
    /// Operating frequency (Hz)
    pub frequency: f64,
}

impl Default for KZKConfig {
    fn default() -> Self {
        Self {
            nx: 128,
            ny: 128,
            nz: 256,
            dx: 0.5e-3, // 0.5 mm
            dz: 0.5e-3, // 0.5 mm
            dt: 10e-9,  // 10 ns
            nt: 1000,
            c0: 1540.0, // water/tissue
            rho0: 1000.0,
            beta: 3.5,   // B/A for water
            alpha0: 0.5, // dB/cm/MHz
            alpha_power: 1.1,
            include_diffraction: true,
            include_absorption: true,
            include_nonlinearity: true,
            frequency: 1e6, // Default 1 MHz
        }
    }
}

/// Validate KZK configuration
pub fn validate_config(config: &KZKConfig) -> Result<(), String> {
    // Check grid sizes
    if config.nx < 2 || config.ny < 2 || config.nz < 2 {
        return Err("Grid dimensions must be at least 2".to_string());
    }

    // Check physical parameters
    if config.c0 <= 0.0 {
        return Err("Sound speed must be positive".to_string());
    }
    if config.rho0 <= 0.0 {
        return Err("Density must be positive".to_string());
    }

    // Check CFL condition for parabolic approximation
    let cfl = config.c0 * config.dt / config.dz;
    if cfl > 0.5 {
        return Err(format!("CFL number {} exceeds 0.5 for stability", cfl));
    }

    // Check parabolic approximation validity
    let theta_max = (config.nx as f64 * config.dx / (2.0 * config.nz as f64 * config.dz)).atan();
    if theta_max > 0.3 {
        // ~17 degrees
        return Err(format!(
            "Maximum angle {:.1}° exceeds parabolic approximation limit",
            theta_max.to_degrees()
        ));
    }

    Ok(())
}

//! KZK (Khokhlov-Zabolotskaya-Kuznetsov) Equation Implementation
//!
//! The KZK equation is a parabolic approximation for directional sound beams,
//! widely used in medical ultrasound for modeling focused transducers.
//!
//! # KZK Equation
//!
//! ```text
//! ∂²p/∂z∂τ = (c₀/2)∇⊥²p + (δ/2c₀³)∂³p/∂τ³ + (β/2ρ₀c₀³)∂²(p²)/∂τ²
//! ```
//!
//! - z: axial coordinate (beam propagation direction)
//! - τ = t − z/c₀: retarded time
//! - ∇⊥²: transverse Laplacian (∂²/∂x² + ∂²/∂y²)
//! - δ: diffusivity of sound [m²/s]
//! - β = 1 + B/(2A): nonlinearity coefficient (dimensionless)
//! - ρ₀, c₀: ambient density [kg/m³] and speed (m/s)
//!
//! # Operator Splitting
//!
//! Strang splitting (Strang 1968) achieves second-order accuracy in Δz:
//!
//! ```text
//! U(Δz) ≈ D(Δz/2) · A(Δz/2) · N(Δz) · A(Δz/2) · D(Δz/2)
//! ```
//!
//! where D = diffraction, A = absorption, N = nonlinearity.
//!
//! # References
//!
//! - Zabolotskaya EA, Khokhlov RV (1969). Sov. Phys. Acoust. 15, 35–40.
//! - Kuznetsov VP (1971). Sov. Phys. Acoust. 16, 467–470.
//! - Aanonsen SI et al. (1984). J. Acoust. Soc. Am. 75(3), 749–768. DOI:10.1121/1.390585
//! - Lee Y-S, Hamilton MF (1995). J. Acoust. Soc. Am. 97(2), 906–917. DOI:10.1121/1.412000
//! - Strang G (1968). SIAM J. Numer. Anal. 5(3), 506–517. DOI:10.1137/0705041
//! - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics. Academic Press.

pub mod absorption;
pub mod angular_spectrum_2d;
pub mod beam_debug;
pub mod complex_parabolic_diffraction;
pub mod constants;
pub mod finite_difference_diffraction;
pub mod harmonic_tracking;
pub mod nonlinearity;
pub mod parabolic_diffraction;
pub mod plane_wave_test;
pub mod shock_capturing;
pub mod solver;
pub mod validation;

pub use harmonic_tracking::{HarmonicAnalysis, HarmonicConfig, HarmonicTracker, PredictionModel};
pub use shock_capturing::{ShockCapture, ShockCapturingConfig, ShockDetectionResult};
pub use solver::KZKSolver;

/// Re-export the physics-layer trait under an ergonomic alias.
///
/// Consumers can use either:
/// - `use kwavers::solver::forward::nonlinear::kzk::KZKSolverTrait;`
/// - `use kwavers::physics::acoustics::wave_propagation::nonlinear::kzk::KZKSolver;`
pub use crate::physics::acoustics::wave_propagation::nonlinear::kzk::KZKSolver as KZKSolverTrait;

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
    /// Medium nonlinearity ratio B/A (dimensionless).
    ///
    /// # Theorem (B/A vs β)
    ///
    /// The equation of state is expanded as a Taylor series in density:
    ///   p = ρ₀c₀²(ρ'/ρ₀) + (B/A)/2 · ρ₀c₀²(ρ'/ρ₀)² + O(ρ'³)
    ///
    /// The nonlinearity coefficient used in the KZK equation is
    ///   β = 1 + B/(2A)
    ///
    /// `b_over_a` stores the raw ratio B/A; β is computed internally by
    /// `NonlinearOperator::new` as `1.0 + b_over_a / 2.0`.
    ///
    /// Typical values: water ≈ 5.0, soft tissue ≈ 6.0–7.5.
    ///
    /// Reference: Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics.
    ///   Academic Press. §2.3.2, eq. (2.3.10).
    pub b_over_a: f64,
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
            b_over_a: 5.0, // B/A for water at 25°C (Beyer 1960)
            alpha0: 0.5,   // dB/cm/MHz
            alpha_power: 1.1,
            include_diffraction: true,
            include_absorption: true,
            include_nonlinearity: true,
            frequency: 1e6, // Default 1 MHz
        }
    }
}

/// Validate KZK configuration
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn validate_config(config: &KZKConfig) -> Result<(), String> {
    // Check grid sizes
    if config.nx < 2 || config.ny < 2 || config.nz < 2 {
        return Err("Grid dimensions must be at least 2".to_owned());
    }

    // Check physical parameters
    if config.c0 <= 0.0 {
        return Err("Sound speed must be positive".to_owned());
    }
    if config.rho0 <= 0.0 {
        return Err("Density must be positive".to_owned());
    }

    // Check CFL condition for parabolic approximation
    let cfl = config.c0 * config.dt / config.dz;
    if cfl > 0.5 {
        return Err(format!("CFL number {cfl} exceeds 0.5 for stability"));
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

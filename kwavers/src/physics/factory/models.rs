//! Physics model type definitions
//!
//! Comprehensive type system for physics models following domain principles

use serde::{Deserialize, Serialize};

/// Physics model configuration with strong typing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsModelConfig {
    pub model_type: PhysicsModelType,
    pub enabled: bool,
    pub parameters: std::collections::HashMap<String, f64>,
}

/// Strongly-typed physics model variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhysicsModelType {
    /// Linear acoustics with wave propagation
    LinearAcoustics {
        solver_type: AcousticSolver,
        boundary_conditions: BoundaryType,
    },
    /// Nonlinear acoustics with harmonic generation
    NonlinearAcoustics {
        equation_type: NonlinearEquation,
        harmonics: usize,
    },
    /// Bubble dynamics with cavitation
    BubbleDynamics {
        model: BubbleModel,
        nucleation: bool,
    },
    /// Thermal diffusion and heating
    ThermalDiffusion { bioheat: bool, perfusion: bool },
    /// Optical propagation and absorption
    OpticalPropagation { scattering: bool, anisotropy: f64 },
    /// Mechanical stress and strain
    MechanicalStress {
        elastic_modulus: f64,
        poisson_ratio: f64,
    },
}

/// Acoustic solver types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcousticSolver {
    FDTD { order: u8 },
    PSTD { spectral_accuracy: bool },
    DG { polynomial_order: u8 },
}

/// Boundary condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryType {
    Absorbing { pml_layers: u8 },
    Reflecting { impedance: Option<f64> },
    Periodic,
    Transparent,
}

/// Nonlinear equation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NonlinearEquation {
    Westervelt,
    Kuznetsov,
    KZK,
}

/// Bubble dynamics models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BubbleModel {
    RayleighPlesset,
    KellerMiksis,
    Gilmore,
}

impl PhysicsModelConfig {
    /// Create linear acoustics configuration
    pub fn linear_acoustics(solver: AcousticSolver) -> Self {
        Self {
            model_type: PhysicsModelType::LinearAcoustics {
                solver_type: solver,
                boundary_conditions: BoundaryType::Absorbing { pml_layers: 10 },
            },
            enabled: true,
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Create nonlinear acoustics configuration
    pub fn nonlinear_acoustics(equation: NonlinearEquation, harmonics: usize) -> Self {
        Self {
            model_type: PhysicsModelType::NonlinearAcoustics {
                equation_type: equation,
                harmonics,
            },
            enabled: true,
            parameters: std::collections::HashMap::new(),
        }
    }
}

impl Default for PhysicsModelConfig {
    fn default() -> Self {
        Self::linear_acoustics(AcousticSolver::FDTD { order: 2 })
    }
}

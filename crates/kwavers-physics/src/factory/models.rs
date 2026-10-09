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
        boundary_conditions: PhysicsBoundaryCondition,
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
    /// Optical propagation and absorption via the diffusion approximation.
    OpticalPropagation { scattering: bool, anisotropy: f64 },
    /// Linear acoustic wave propagation with exact power-law absorption via
    /// a fractional-Laplacian spectral filter (Treeby & Cox 2010).
    ///
    /// More accurate than relaxation-arm memory variables for broadband
    /// power-law media such as soft tissue.
    FractionalViscoacoustic {
        /// Absorption coefficient α₀ in Np/m/(rad/s)^y.
        alpha0: f64,
        /// Power-law exponent y (typical tissue: 1.0–1.5).
        exponent: f64,
    },
    /// Elastic (mechanical) stress–velocity propagation in a solid (λ, μ).
    ///
    /// Distinct from [`LinearAcoustics`](Self::LinearAcoustics): the shear
    /// modulus `μ > 0` supports shear waves, which the acoustic-fluid path
    /// cannot represent. See ADR 021.
    MechanicalStress { wave_kind: ElasticWaveKind },
    /// Photoacoustic time-reversal reconstruction. Re-injects measured
    /// photoacoustic time series backward to focus at the absorber origin.
    ///
    /// Requires the solver-side time-reversal feature wiring for the selected
    /// equation family.
    PhotoacousticTimeReversal { equation: NonlinearEquation },
}

/// Elastic-wave propagation mode for [`PhysicsModelType::MechanicalStress`].
///
/// Additive-by-design: new modes (anisotropic, nonlinear) extend this enum
/// without a breaking change to the acoustic capability surface.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ElasticWaveKind {
    /// Isotropic linear elastic stress–velocity propagation (Lamé `λ`, `μ`).
    Isotropic,
    /// Vertical Transverse Isotropic (VTI) elastic propagation with five
    /// independent stiffness parameters.
    ///
    /// Reduces to isotropic when
    /// `c11 = c33 = λ + 2μ`, `c13 = λ`, and `c44 = c66 = μ`.
    Vti {
        c11: f64,
        c13: f64,
        c33: f64,
        c44: f64,
        c66: f64,
    },
}

/// Acoustic solver types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcousticSolver {
    FDTD { order: u8 },
    PSTD { spectral_accuracy: bool },
    DG { polynomial_order: u8 },
}

/// Wave propagation boundary treatment options for physics factory configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhysicsBoundaryCondition {
    Absorbing { pml_layers: u8 },
    Reflecting { impedance: Option<f64> },
    Periodic,
    Transparent,
}

/// Nonlinear equation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NonlinearEquation {
    /// Spectral (k-space) Westervelt equation solver. Default Westervelt path.
    Westervelt,
    /// Explicit FDTD Westervelt solver. Supports 2nd/4th/6th order stencils,
    /// artificial viscosity, and pointwise heterogeneous media (c₀, ρ₀, β).
    /// Prefer over `Westervelt` when heterogeneous media or explicit time-stepping
    /// is required; prefer `Westervelt` for high accuracy in homogeneous media.
    WesterveltFdtd,
    /// Hybrid Angular Spectrum (HAS) nonlinear propagation.
    ///
    /// Operator-splitting with FFT-based angular-spectrum diffraction and
    /// time-domain nonlinearity (Christopher & Parker 1991, JASA 90, 507–521).
    /// Efficient for moderately nonlinear focused beams at moderate angles.
    HybridAngularSpectrum,
    /// Full Kuznetsov equation.
    Kuznetsov,
    /// KZK (parabolic beam equation, multiple diffraction schemes).
    KZK,
}

/// Bubble dynamics models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BubbleModel {
    RayleighPlesset,
    KellerMiksis,
    KellerHerring,
    Gilmore,
}

impl PhysicsModelConfig {
    /// Create linear acoustics configuration
    #[must_use]
    pub fn linear_acoustics(solver: AcousticSolver) -> Self {
        Self {
            model_type: PhysicsModelType::LinearAcoustics {
                solver_type: solver,
                boundary_conditions: PhysicsBoundaryCondition::Absorbing { pml_layers: 10 },
            },
            enabled: true,
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Create nonlinear acoustics configuration
    #[must_use]
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

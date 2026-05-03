//! PDE type enumeration for meta-learning task classification.

/// Types of partial differential equations supported in meta-learning
///
/// Each PDE type represents a different physics domain with characteristic
/// mathematical structure and computational challenges.
///
/// # Complexity Ordering (for curriculum learning)
///
/// From simplest to most complex:
/// 1. **Wave**: Linear, second-order hyperbolic PDE
/// 2. **Diffusion**: Linear, second-order parabolic PDE
/// 3. **Acoustic**: Linear wave equation with medium heterogeneity
/// 4. **Elastic**: Coupled vector-valued wave equations
/// 5. **Electromagnetic**: Coupled Maxwell's equations (vector calculus)
/// 6. **Navier-Stokes**: Nonlinear, coupled momentum and continuity equations
///
/// # Literature
///
/// - Evans, L. C. (2010). *Partial Differential Equations* (Vol. 19). AMS.
/// - Karniadakis, G. E., et al. (2021). "Physics-informed machine learning",
///   *Nature Reviews Physics*, 3(6), 422-440.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PdeType {
    /// Linear wave equation: ∂²u/∂t² - c²∇²u = 0
    ///
    /// Complexity: Low
    /// Applications: Acoustics, seismology, electromagnetic waves
    Wave,

    /// Heat/diffusion equation: ∂u/∂t - α∇²u = 0
    ///
    /// Complexity: Low-Medium
    /// Applications: Heat transfer, mass diffusion, option pricing
    Diffusion,

    /// Navier-Stokes equations: ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u
    ///
    /// Complexity: Very High (nonlinear, coupled)
    /// Applications: Fluid dynamics, aerodynamics, weather prediction
    NavierStokes,

    /// Maxwell's equations: ∇×E = -∂B/∂t, ∇×B = μ₀(J + ε₀∂E/∂t)
    ///
    /// Complexity: High (coupled vector equations)
    /// Applications: Electromagnetic wave propagation, antenna design
    Electromagnetic,

    /// Acoustic wave equation with heterogeneous medium
    ///
    /// Complexity: Medium
    /// Applications: Medical ultrasound, sonar, room acoustics
    Acoustic,

    /// Elastic wave equations: ρ∂²u/∂t² = ∇·σ
    ///
    /// Complexity: High (coupled vector equations, tensor operations)
    /// Applications: Seismology, structural mechanics, elastography
    Elastic,
}

impl PdeType {
    /// Get the relative computational complexity (0.0 = easiest, 1.0 = hardest)
    ///
    /// Used for curriculum learning strategies to progressively increase
    /// task difficulty during meta-training.
    pub fn complexity(&self) -> f64 {
        match self {
            PdeType::Wave => 0.2,
            PdeType::Diffusion => 0.3,
            PdeType::Acoustic => 0.4,
            PdeType::Elastic => 0.6,
            PdeType::Electromagnetic => 0.7,
            PdeType::NavierStokes => 1.0,
        }
    }

    /// Get the typical number of coupled equations
    pub fn num_equations(&self) -> usize {
        match self {
            PdeType::Wave | PdeType::Diffusion | PdeType::Acoustic => 1,
            PdeType::Elastic => 2,         // 2D elasticity: (u_x, u_y)
            PdeType::Electromagnetic => 6, // E_x, E_y, E_z, B_x, B_y, B_z
            PdeType::NavierStokes => 4,    // u, v, w, p (or u, v, p in 2D)
        }
    }

    /// Check if the PDE is linear
    pub fn is_linear(&self) -> bool {
        matches!(
            self,
            PdeType::Wave | PdeType::Diffusion | PdeType::Acoustic | PdeType::Elastic
        )
    }
}

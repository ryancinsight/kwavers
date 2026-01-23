//! Nonlinear Shear Wave Elastography (NL-SWE) Solver
//!
//! Numerical solver for nonlinear elastic wave propagation in soft tissue.
//!
//! ## Architectural Note
//!
//! This module was moved from `physics/acoustics/imaging/modalities/elastography/nonlinear/`
//! to enforce proper separation of concerns:
//! - **Solver layer** (HERE): Numerical methods, discretization, time integration
//! - **Physics layer**: Material models, constitutive equations, physics specifications
//!
//! Implements nonlinear elastic wave propagation for advanced tissue characterization.
//!
//! ## Overview
//!
//! Nonlinear SWE extends linear elasticity to capture:
//! 1. Hyperelastic material behavior (finite strain theory)
//! 2. Higher-order elastic constants (nonlinear stress-strain)
//! 3. Harmonic generation and detection
//! 4. Advanced parameter estimation for tissue nonlinearity
//!
//! ## Governing Equations
//!
//! ### Hyperelastic Constitutive Models
//!
//! TODO_AUDIT: P2 - Advanced Nonlinear Elasticity - Implement complete finite strain hyperelasticity with higher-order elastic constants and viscoelastic coupling
//! DEPENDS ON: solver/forward/elastic/nonlinear/hyperelastic.rs, solver/forward/elastic/nonlinear/acoustoelastic.rs, solver/forward/elastic/nonlinear/viscoelastic.rs
//! MISSING: Ogden hyperelastic model: W = Σ (μᵢ/αᵢ) (λ₁^αⁱ + λ₂^αⁱ + λ₃^αⁱ - 3) for compressible materials
//! MISSING: Acoustoelastic tensor: cᵢⱼₖₗ + cᵢⱼₖₗ^(AE) with prestress dependence
//! MISSING: Fractional viscoelasticity: σ(t) = ∫ G(t-τ) dε(τ)/dτ dτ with fractional derivatives
//! MISSING: Higher-order elastic constants M, N for third-order nonlinearity
//! MISSING: Wave mixing and harmonic generation in prestressed media
//! THEOREM: Piola-Kirchhoff stress: P = J σ F^{-T} for finite deformation
//! THEOREM: Acoustoelastic effect: Δc/c = (1/2μ) σ (l² + m² - 2λ/μ) for isotropic solids
//! REFERENCES: Brugger (1964) Phys Rev; Thurston & Brugger (1964) Phys Rev; Man & Ogilvy (1997) J Phys D
//!
//! **Neo-Hookean Model:**
//! W = C₁(I₁ - 3) + D₁(J - 1)²
//!
//! **Mooney-Rivlin Model:**
//! W = C₁(I₁ - 3) + C₂(I₂ - 3) + D₁(J - 1)²
//!
//! **Ogden Model:**
//! W = Σᵢ (μᵢ/αᵢ) [(λ₁^αⁱ + λ₂^αⁱ + λ₃^αⁱ - 3)]
//!
//! Where:
//! - W = strain energy density function
//! - I₁, I₂ = strain invariants
//! - J = determinant of deformation gradient
//! - C₁, C₂, μᵢ, αᵢ = material parameters
//!
//! ### Nonlinear Wave Equation
//!
//! ∂²u/∂t² = ∇·σ(u, ∇u) + nonlinear terms
//!
//! Where nonlinear terms include:
//! - Geometric nonlinearity: ∇·(∇u ⊗ ∇u)
//! - Material nonlinearity: higher-order elastic constants
//! - Harmonic generation: frequency doubling, sum/difference frequencies
//!
//! ## Architecture
//!
//! This module follows a deep vertical hierarchy with clear separation of concerns:
//!
//! - `config` - Configuration parameters and simulation settings
//! - `material` - Hyperelastic material models and constitutive relations
//! - `wave_field` - Wave field state representation and operations
//! - `numerics` - Numerical operators (Laplacian, divergence, gradient)
//! - `solver` - Main solver orchestration and integration
//!
//! ## Literature References
//!
//! - Destrade, M., et al. (2010). "Finite amplitude waves in Mooney-Rivlin hyperelastic
//!   materials." *Journal of the Acoustical Society of America*, 127(6), 3336-3342.
//! - Bruus, H. (2012). "Acoustofluidics 7: The acoustic radiation force on small
//!   particles." *Lab on a Chip*, 12(6), 1014-1021.
//! - Chen, S., et al. (2013). "Harmonic motion detection in ultrasound elastography."
//!   *IEEE Transactions on Medical Imaging*, 32(5), 863-874.
//! - Parker, K. J., et al. (2011). "Sonoelasticity of organs: Shear waves ring a bell."
//!   *Journal of Ultrasound in Medicine*, 30(4), 507-515.
//! - Nightingale, K. R., et al. (2015). "Acoustic Radiation Force Impulse (ARFI)
//!   imaging: A review." *Current Medical Imaging Reviews*, 11(1), 22-32.

mod config;
mod material;
mod numerics;
mod solver;
mod wave_field;

// Public API exports
pub use config::NonlinearSWEConfig;
pub use material::HyperelasticModel;
pub use numerics::NumericsOperators;
pub use solver::NonlinearElasticWaveSolver;
pub use wave_field::NonlinearElasticWaveField;

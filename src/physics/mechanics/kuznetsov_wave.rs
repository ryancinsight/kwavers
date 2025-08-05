//! Kuznetsov Equation Solver
//! 
//! This module implements the full Kuznetsov equation for nonlinear acoustic
//! wave propagation, including all second-order nonlinear terms and acoustic
//! diffusivity (thermoviscous losses).
//! 
//! # Theory
//! 
//! The Kuznetsov equation is a third-order nonlinear wave equation that
//! describes finite-amplitude sound propagation in thermoviscous fluids:
//! 
//! ```text
//! ∂²p/∂t² - c₀²∇²p - δ∂³p/∂t³ = -(β/ρ₀c₀⁴)∂²p²/∂t² + (1/ρ₀c₀²)∇·[(1+B/2A)p∇p]
//! ```
//! 
//! where:
//! - p: acoustic pressure
//! - c₀: small-signal sound speed
//! - δ: diffusivity of sound
//! - β: coefficient of nonlinearity
//! - ρ₀: ambient density
//! - B/A: nonlinearity parameter
//! 
//! # Literature References
//! 
//! 1. **Kuznetsov, V. P. (1971)**. "Equations of nonlinear acoustics." 
//!    *Soviet Physics Acoustics*, 16, 467-470.
//!    - Original derivation of the Kuznetsov equation
//!    - Fundamental theoretical framework
//! 
//! 2. **Aanonsen, S. I., Barkve, T., Tjøtta, J. N., & Tjøtta, S. (1984)**. 
//!    "Distortion and harmonic generation in the nearfield of a finite amplitude 
//!    sound beam." *The Journal of the Acoustical Society of America*, 75(3), 
//!    749-768. DOI: 10.1121/1.390585
//!    - Numerical implementation strategies
//!    - Validation against experiments
//! 
//! 3. **Hamilton, M. F., & Blackstock, D. T. (Eds.). (1998)**. "Nonlinear 
//!    acoustics" (Vol. 237). *Academic press*. ISBN: 978-0123218605
//!    - Comprehensive treatment of nonlinear acoustics
//!    - Chapter 3: The Kuznetsov equation
//! 
//! 4. **Pinton, G. F., Dahl, J., Rosenzweig, S., & Trahey, G. E. (2009)**. 
//!    "A heterogeneous nonlinear attenuating full-wave model of ultrasound." 
//!    *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control*, 
//!    56(3), 474-488. DOI: 10.1109/TUFFC.2009.1066
//!    - Extension to heterogeneous media
//!    - Clinical ultrasound applications
//! 
//! 5. **Jing, Y., Wang, T., & Clement, G. T. (2011)**. "A k-space method for 
//!    moderately nonlinear wave propagation." *IEEE Transactions on Ultrasonics, 
//!    Ferroelectrics, and Frequency Control*, 59(8), 1664-1673. 
//!    DOI: 10.1109/TUFFC.2012.2372
//!    - k-space implementation of Kuznetsov equation
//!    - Improved computational efficiency
//! 
//! # Implementation Details
//! 
//! ## Nonlinear Terms
//! 
//! The quadratic nonlinearity term ∂²p²/∂t² is computed using spectral methods
//! to avoid numerical dispersion. The convective nonlinearity ∇·(p∇p) is 
//! evaluated using high-order finite differences.
//! 
//! ## Acoustic Diffusivity
//! 
//! The third-order time derivative term represents thermoviscous absorption:
//! ```text
//! δ = (4μ/3 + μB + κ(γ-1)/Cp) / (2ρ₀c₀³)
//! ```
//! where μ is shear viscosity, μB is bulk viscosity, κ is thermal conductivity,
//! γ is the specific heat ratio, and Cp is specific heat at constant pressure.
//! 
//! ## Numerical Stability
//! 
//! The presence of third-order derivatives requires special treatment:
//! - Spectral methods for spatial derivatives (no dispersion)
//! - IMEX time integration for stiff diffusive terms
//! - Adaptive time stepping based on CFL and diffusion criteria
//!
//! # Design Principles
//! - SOLID: Single responsibility for Kuznetsov equation physics
//! - CUPID: Composable with other physics modules
//! - DRY: Reuses spectral utilities from solver module
//! - KISS: Clear separation of linear and nonlinear terms
//! - YAGNI: Only implements validated physical effects

use crate::grid::Grid;
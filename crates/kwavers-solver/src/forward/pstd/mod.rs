//! Pseudospectral Time-Domain (PSTD) Solver
//!
//! Implements the k-space pseudospectral method for acoustic wave propagation
//! in heterogeneous media with power-law frequency-dependent absorption.
//!
//! # Governing Equations
//!
//! The first-order coupled acoustic equations (Treeby & Cox 2010, Eqs. 5–7):
//! ```text
//! ∂u/∂t = −(1/ρ₀) ∇p                                    (momentum)
//! ∂ρ/∂t = −ρ₀ ∇·u − 2ρ₀α₀cᵧ ∇·(−∇²)^{(y/2−1)} u     (mass, absorption)
//! p     = c₀² (ρ + d·ρ₀(∂ρ/∂t) − Lρ)                   (equation of state)
//! ```
//!
//! where `y` is the power-law exponent, `α₀` is the absorption coefficient,
//! `cᵧ = −2α₀c₀^{y−1}`, and `L` is a loss operator.
//!
//! # Theorem: Spectral Convergence (Boyd 2001, Theorem 2.1)
//!
//! For a function u ∈ Cˢ(Ω) with s continuous derivatives on a periodic domain,
//! the pseudospectral derivative approximation satisfies:
//! ```text
//! ‖∂u/∂x − D_N u‖_∞ ≤ C · N^{−s}
//! ```
//! where D_N is the N-point spectral derivative operator and C depends on ‖u^(s+1)‖.
//! For analytic (C^∞) functions, convergence is **exponential**: error ~ O(exp(−αN)).
//!
//! Compared to FDTD: 2nd-order O(Δx²) requires ~10 points per wavelength (PPW),
//! while PSTD achieves the same accuracy at **2 PPW** (Nyquist limit).
//!
//! # K-Space Correction (Tabei et al. 2002)
//!
//! The standard PSTD scheme accumulates temporal dispersion error. The k-space
//! correction replaces the temporal finite difference with an exact spectral
//! propagator:
//! ```text
//! û^{n+1} = û^n · exp(−i·c₀·|k|·Δt)     (exact propagation in k-space)
//! ```
//! This eliminates temporal dispersion entirely for homogeneous media, allowing
//! CFL numbers up to 1.0 (vs. 0.577 for 3D FDTD).
//!
//! # Gibbs Phenomenon and Spectral Filtering
//!
//! At material discontinuities, the Fourier series exhibits O(1) oscillations
//! (Gibbs phenomenon) that do not vanish with increasing N. This is mitigated by:
//!
//! 1. **Smooth medium interfaces**: Use gradual transitions (sigmoid) over 2–4 grid points
//! 2. **Spectral filtering**: Apply a low-pass filter H(k) to suppress aliasing:
//!    ```text
//!    H(k) = { 1                           |k| ≤ k_c
//!           { exp(−α(|k|−k_c)^β / (k_max−k_c)^β)  |k| > k_c
//!    ```
//!    where k_c ≈ 2/3 · k_max (Hou & Li 2007).
//! 3. **DG coupling**: For domains with sharp interfaces, use the DG sub-module
//!    (`dg/`) which couples PSTD with a Discontinuous Galerkin shock-capturing
//!    scheme at material boundaries.
//!
//! # Boundary Conditions
//!
//! PSTD with FFT assumes periodicity. Non-periodic boundaries require:
//! - **CPML** (Convolutional PML): Roden & Gedney (2000), implemented in
//!   `domain/boundary/cpml/`. The recursive-convolution formulation is compatible
//!   with spectral derivatives.
//! - **PML**: Classical split-field PML (Berenger 1994) as fallback.
//!
//! # References
//!
//! - Treeby BE, Cox BT (2010). "k-Wave: MATLAB toolbox for the simulation
//!   and reconstruction of photoacoustic wave fields." J Biomed Opt 15(2):021314.
//! - Tabei M, Mast TD, Waag RC (2002). "A k-space method for coupled
//!   first-order acoustic propagation equations." JASA 111(1):53–63.
//! - Boyd JP (2001). *Chebyshev and Fourier Spectral Methods*, 2nd ed. Dover.
//! - Hou TY, Li R (2007). "Computing nearly singular solutions using
//!   pseudo-spectral methods." J Comput Phys 226(1):379–397.
//! - Roden JA, Gedney SD (2000). "Convolution PML (CPML): An efficient
//!   FDTD implementation." Microw Opt Technol Lett 27(5):334–339.

pub mod checkpoint;
pub mod config;
pub mod data;
pub mod derivatives; // Spectral derivatives (NEW)
pub mod dg;
pub mod extensions;
pub mod implementation;
pub mod numerics;
pub mod physics;
pub mod plugin;
pub mod propagator;
pub mod utils;

pub use config::PSTDConfig;
pub use derivatives::SpectralDerivativeOperator;
pub use extensions::{PstdElasticPlugin, SpectralElasticConfig};
pub use implementation::core::orchestrator::PSTDSolver;
pub use plugin::PSTDPlugin;

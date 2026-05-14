//! Westervelt Nonlinear Wave Equation — FDTD Solver
//!
//! This module implements the Westervelt equation using finite-difference
//! time-domain (FDTD) methods, suitable for heterogeneous media with
//! thermoviscous absorption and cumulative nonlinear distortion.
//!
//! ---
//!
//! ## Background
//!
//! Linear acoustics describes wave propagation in the limit of infinitesimally
//! small perturbations. At finite amplitudes two physical effects alter the
//! waveform progressively as it travels:
//!
//! 1. **Thermoviscous absorption** — classical viscous and thermal relaxation
//!    processes convert acoustic energy to heat. For a monochromatic wave at
//!    angular frequency ω the absorbed power scales as ω².
//!
//! 2. **Cumulative nonlinear distortion** — the local sound speed depends on
//!    the instantaneous particle velocity: rarefactions travel slightly slower
//!    and compressions slightly faster, causing progressive steepening and
//!    harmonic generation over the propagation path.
//!
//! The Westervelt equation captures both effects within a single scalar PDE
//! for the acoustic pressure p. It is the preferred model for therapeutic and
//! diagnostic ultrasound (HIFU, lithotripsy, harmonic imaging) and is
//! equivalent to the KZK equation in paraxial geometries.
//!
//! ---
//!
//! ## Theorem — Westervelt Equation
//!
//! In a quiescent, homogeneous medium with ambient density ρ₀ and small-
//! amplitude sound speed c₀, the acoustic pressure p(x, t) satisfies:
//!
//! ```text
//! ∇²p - (1/c₀²) ∂²p/∂t² + (δ/c₀⁴) ∂³p/∂t³ + (β/ρ₀c₀⁴) ∂²p²/∂t² = 0
//! ```
//!
//! **Variable glossary**
//!
//! | Symbol | Meaning | SI unit |
//! |--------|---------|---------|
//! | p      | Acoustic pressure | Pa |
//! | c₀     | Small-amplitude sound speed | m s⁻¹ |
//! | ρ₀     | Ambient (equilibrium) density | kg m⁻³ |
//! | δ      | Diffusivity of sound = (4μ/3 + μ_B)/ρ₀ + κ(1/c_v − 1/c_p)/ρ₀ | m² s⁻¹ |
//! | β      | Coefficient of nonlinearity = 1 + B/2A | dimensionless |
//! | B/A    | Parameter of nonlinearity (medium property) | dimensionless |
//!
//! **Term-by-term interpretation**
//!
//! * `∇²p − (1/c₀²) ∂²p/∂t²` — lossless linear wave operator (d'Alembertian
//!   applied to p).
//! * `+(δ/c₀⁴) ∂³p/∂t³` — **thermoviscous absorption**: odd time-derivative
//!   introduces dissipation proportional to ω² in the frequency domain,
//!   consistent with classical Stokes-Kirchhoff theory.
//! * `+(β/ρ₀c₀⁴) ∂²p²/∂t²` — **cumulative nonlinear distortion**: the
//!   quadratic pressure term generates harmonics and shock precursors. The
//!   sign convention places it on the left-hand side (equals zero), matching
//!   Westervelt's original 1963 formulation.
//!
//! **Lemma — reduction to KZK in paraxial limit**
//!
//! Applying the parabolic (paraxial) approximation `∂²/∂z² ≈ 2/c₀ ∂²/∂z∂t`
//! to the Westervelt equation recovers the Khokhlov-Zabolotskaya-Kuznetsov
//! (KZK) equation, which is the standard model for focused beam propagation.
//!
//! **Lemma — linear limit**
//!
//! Setting β = 0 reduces the Westervelt equation to the lossy linear wave
//! equation. Setting additionally δ = 0 recovers the lossless wave equation.
//!
//! ---
//!
//! ## Discretization
//!
//! ### Explicit FDTD Leapfrog
//!
//! The equation is advanced in time using a second-order explicit scheme.
//! Denoting `p^n ≡ p(x, nΔt)` and `Δt` as the time step:
//!
//! ```text
//! p^{n+1} = 2p^n - p^{n-1} + (c₀Δt)² ∇²p^n
//!           - (δΔt/c₀²)(p^n - 2p^{n-1} + p^{n-2})/Δt²
//!           + (βΔt²/ρ₀c₀²) ∂²(p²)/∂t²|^n
//! ```
//!
//! where the nonlinear term is evaluated via the product rule:
//!
//! ```text
//! ∂²(p²)/∂t²|^n ≈ 2p^n (p^n - 2p^{n-1} + p^{n-2})/Δt²
//!                  + 2[(p^n - p^{n-1})/Δt]²
//! ```
//!
//! ### Sign convention
//!
//! The nonlinear contribution to `p^{n+1}` is **positive** (the explicit
//! update adds `+ q·∂²(p²)/∂t²` with `q = β·Δt² / (ρ₀·c₀²) > 0`). Equivalently,
//! solving the Westervelt equation
//! `∇²p − (1/c²)·p_tt + (β/(ρ₀c⁴))·∂²(p²)/∂t² = 0` for `p_tt` produces
//! `p_tt = c²·∇²p + (β/(ρ₀c²))·∂²(p²)/∂t²` with the nonlinear coefficient
//! positive on the right-hand side. This is the sign required for physical
//! forward steepening — compressions travel faster than rarefactions and
//! peaks at a fixed downstream receiver arrive earlier than the linear
//! prediction.
//!
//! ### Spatial Laplacian
//!
//! The Laplacian ∇²p is approximated by a compact finite-difference stencil.
//! This module supports:
//!
//! * **2nd-order** — standard 7-point stencil (coefficients ±1, −2).
//! * **4th-order** — 13-point stencil (coefficients −1/12, 4/3, −5/2, …),
//!   halving the leading truncation error for a given spatial resolution.
//!
//! ### Stability (CFL Condition)
//!
//! Stability requires:
//!
//! ```text
//! Δt ≤ cfl_safety × Δx_min / (c_max √3)
//! ```
//!
//! where `Δx_min = min(Δx, Δy, Δz)`, `c_max` is the peak sound speed in the
//! domain, and `cfl_safety < 1` is a user-configurable safety margin
//! (default 0.95).
//!
//! ### Operator Splitting (Alternative)
//!
//! For pseudospectral solvers the Westervelt equation may be split into:
//!
//! 1. **Propagation sub-step** — linear wave equation advanced via FFT-based
//!    k-space methods (see the `pstd` module).
//! 2. **Absorption sub-step** — thermoviscous term applied as a frequency-
//!    domain filter.
//! 3. **Nonlinear sub-step** — explicit update of the ∂²p²/∂t² term.
//!
//! ---
//!
//! ## Implementation Notes
//!
//! * **Heterogeneous media**: c₀, ρ₀, and β are evaluated pointwise from
//!   the `Medium` trait at each grid node. The absorption coefficient α
//!   (Np m⁻¹) is converted to the diffusivity δ via
//!   `δ = 2αc₀³/(2πf_ref)²` at a 1 MHz reference frequency.
//!
//! * **Power-law absorption** (Treeby & Cox 2010): biological tissue exhibits
//!   frequency-dependent absorption α ∝ fʸ with y ≈ 1–2, which cannot be
//!   reproduced exactly by the `δ ω²` classical model. The
//!   `absorb_tau`/`absorb_eta` fractional-Laplacian approach implemented in
//!   the `pstd` module replaces the diffusivity term for power-law media,
//!   providing physically accurate attenuation across the full bandwidth.
//!
//! * **Artificial viscosity**: a small numerical viscosity term
//!   `ν_art Δt ∇²p^n` is added to suppress spurious high-frequency
//!   oscillations near shocks. The default coefficient is 0.01 and may be
//!   set to zero for energy-conservation tests.
//!
//! * **Conservation diagnostics**: optional tracking of total acoustic energy
//!   `E = ∫ p²/(2ρ₀c₀²) dV`, momentum, and mass is provided via the
//!   `ConservationDiagnostics` trait. Critical violations trigger `warn!`
//!   log entries via the tracing infrastructure.
//!
//! * **Source injection**: point sources are added after the field update
//!   by incrementing the nearest grid node by `amplitude × Δt`. Distributed
//!   source masks (k-Wave style) are handled in the parent `nonlinear` module.
//!
//! ## Module layout
//!
//! - `laplacian`: finite-difference Laplacian (O2/O4) into a pre-allocated
//!   workspace.
//! - `nonlinear`: ∂²p²/∂t² product-rule kernel.
//! - `update`: full Westervelt time-step (linear + nonlinear + absorption +
//!   artificial viscosity), source injection, history rotation, and
//!   conservation-check pipeline.
//! - `conservation`: `ConservationDiagnostics` trait impl.
//!
//! ---
//!
//! ## References
//!
//! 1. Westervelt, P. J. (1963). Parametric acoustic array. *Journal of the
//!    Acoustical Society of America*, **35**(4), 535–537.
//!    <https://doi.org/10.1121/1.1918525>
//!
//! 2. Hamilton, M. F., & Blackstock, D. T. (Eds.). (1998). *Nonlinear
//!    Acoustics* (Vol. 237). Academic Press. (Chapter 3: Westervelt equation.)
//!
//! 3. Treeby, B. E., & Cox, B. T. (2010). Modeling power law absorption and
//!    dispersion for acoustic propagation using the fractional Laplacian.
//!    *Journal of the Acoustical Society of America*, **127**(5), 2741–2748.
//!    <https://doi.org/10.1121/1.3377056>
//!
//! 4. Treeby, B. E., & Cox, B. T. (2010). k-Wave: MATLAB toolbox for the
//!    simulation and reconstruction of photoacoustic wave fields. *Journal of
//!    Biomedical Optics*, **15**(2), 021314.
//!    <https://doi.org/10.1117/1.3360308>
//!
//! 5. Aanonsen, S. I., Barkve, T., Tjøtta, J. N., & Tjøtta, S. (1984).
//!    Distortion and harmonic generation in the nearfield of a finite amplitude
//!    sound beam. *Journal of the Acoustical Society of America*, **75**(3),
//!    749–768. <https://doi.org/10.1121/1.390585>

mod conservation;
mod laplacian;
mod nonlinear;
mod update;

#[cfg(test)]
mod tests;

use ndarray::Array3;

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::solver::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances, ConservationTracker,
};

/// Configuration for Westervelt FDTD solver
#[derive(Debug, Clone)]
pub struct WesterveltFdtdConfig {
    /// Spatial discretization order (2, 4, or 6)
    pub spatial_order: usize,
    /// Enable absorption/attenuation term
    pub enable_absorption: bool,
    /// CFL safety factor (< 1.0)
    pub cfl_safety: f64,
    /// Artificial viscosity coefficient for stability (dimensionless, 0.0-1.0)
    pub artificial_viscosity: f64,
}

impl Default for WesterveltFdtdConfig {
    fn default() -> Self {
        Self {
            spatial_order: 4,
            enable_absorption: true,
            cfl_safety: 0.95,
            artificial_viscosity: 0.01, // Small artificial viscosity for stability
        }
    }
}

/// Westervelt equation solver using FDTD
#[derive(Debug)]
pub struct WesterveltFdtd {
    pub(super) config: WesterveltFdtdConfig,
    /// Current pressure field p^n
    pub(super) pressure: Array3<f64>,
    /// Previous pressure field p^{n-1}
    pub(super) pressure_prev: Array3<f64>,
    /// Two steps back p^{n-2} (for absorption term)
    pub(super) pressure_prev2: Option<Array3<f64>>,
    /// Workspace for Laplacian calculation
    pub(super) laplacian: Array3<f64>,
    /// Conservation diagnostics tracker
    pub(super) conservation_tracker: Option<ConservationTracker>,
    /// Current time step counter
    pub(super) current_step: usize,
    /// Current simulation time
    pub(super) current_time: f64,
    /// Grid reference for conservation calculations
    pub(super) grid: Grid,
    /// Medium reference for conservation calculations
    pub(super) medium_properties: MediumProperties,
}

/// Cached medium properties for conservation calculations
#[derive(Debug, Clone)]
pub(super) struct MediumProperties {
    pub(super) rho0: f64,
    pub(super) c0: f64,
}

impl WesterveltFdtd {
    /// Create a new Westervelt FDTD solver
    pub fn new(config: WesterveltFdtdConfig, grid: &Grid, medium: &dyn Medium) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);

        // Get representative medium properties (center point)
        let center_x = grid.dx * (grid.nx as f64) / 2.0;
        let center_y = grid.dy * (grid.ny as f64) / 2.0;
        let center_z = grid.dz * (grid.nz as f64) / 2.0;
        let rho0 = crate::domain::medium::density_at(medium, center_x, center_y, center_z, grid);
        let c0 = crate::domain::medium::sound_speed_at(medium, center_x, center_y, center_z, grid);

        Self {
            config,
            pressure: Array3::zeros(shape),
            pressure_prev: Array3::zeros(shape),
            pressure_prev2: None,
            laplacian: Array3::zeros(shape),
            conservation_tracker: None,
            current_step: 0,
            current_time: 0.0,
            grid: grid.clone(),
            medium_properties: MediumProperties { rho0, c0 },
        }
    }

    /// Enable conservation diagnostics with specified tolerances
    ///
    /// # Arguments
    ///
    /// * `tolerances` - Conservation tolerance parameters (absolute/relative/check_interval)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kwavers::solver::forward::nonlinear::conservation::ConservationTolerances;
    ///
    /// let mut solver = WesterveltFdtd::new(config, &grid, &medium);
    /// solver.enable_conservation_diagnostics(ConservationTolerances::default());
    /// ```
    pub fn enable_conservation_diagnostics(&mut self, tolerances: ConservationTolerances) {
        let initial_energy = self.calculate_total_energy();
        let initial_momentum = self.calculate_total_momentum();
        let initial_mass = self.calculate_total_mass();

        self.conservation_tracker = Some(ConservationTracker::new(
            initial_energy,
            initial_momentum,
            initial_mass,
            tolerances,
        ));
    }

    /// Disable conservation diagnostics
    pub fn disable_conservation_diagnostics(&mut self) {
        self.conservation_tracker = None;
    }

    /// Get conservation diagnostic summary
    ///
    /// Returns a summary of all conservation checks performed,
    /// including maximum severity and error magnitudes.
    pub fn get_conservation_summary(&self) -> Option<String> {
        self.conservation_tracker
            .as_ref()
            .map(|tracker| tracker.summary().to_string())
    }

    /// Check if solution satisfies conservation constraints
    ///
    /// Returns `true` if all conservation violations are within acceptable limits.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn is_solution_valid(&self) -> bool {
        self.conservation_tracker
            .as_ref()
            .is_none_or(|tracker| tracker.is_solution_valid())
    }

    /// Get the current pressure field
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn pressure(&self) -> &Array3<f64> {
        &self.pressure
    }

    /// Calculate the CFL-limited time step
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn calculate_dt(&self, medium: &dyn Medium, grid: &Grid) -> KwaversResult<f64> {
        // Get maximum sound speed
        let c_max = medium
            .sound_speed_array()
            .iter()
            .fold(0.0f64, |acc, &x| acc.max(x));

        // CFL condition for 3D FDTD
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let dt_cfl = self.config.cfl_safety * dx_min / (c_max * 3.0f64.sqrt());

        Ok(dt_cfl)
    }
}

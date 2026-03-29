//! Keller-Miksis Equation Solver
//!
//! Implementation of the compressible Keller-Miksis equation for high-speed
//! bubble dynamics accounting for liquid compressibility effects.
//!
//! ---
//!
//! ## Background
//!
//! The Rayleigh-Plesset (R-P) equation describes the radial motion of a
//! spherical bubble in an incompressible liquid. At large oscillation
//! amplitudes the bubble wall velocity Ṙ approaches an appreciable fraction
//! of the liquid sound speed c_L (acoustic Mach number Ṙ/c_L ~ 0.1), and
//! the incompressible assumption breaks down. The Keller-Miksis (K-M) model
//! corrects for first-order liquid compressibility by retaining leading-order
//! radiation-damping terms. It is the standard ODE used in single-bubble
//! sonoluminescence, therapeutic ultrasound, and cavitation-erosion modelling.
//!
//! ---
//!
//! ## Theorem — Keller-Miksis ODE
//!
//! For a spherical bubble of instantaneous radius R(t) driven by an external
//! pressure field p_∞(t) in a compressible Newtonian liquid with density
//! ρ_L and small-amplitude sound speed c_L, the bubble-wall motion satisfies:
//!
//! ```text
//! ρ_L [ (1 - Ṙ/c_L) R R̈ + 3/2 (1 - Ṙ/3c_L) Ṙ² ] =
//!     (1 + Ṙ/c_L)(p_B - p_∞) + (R/c_L) d/dt(p_B - p_∞)
//! ```
//!
//! **Variable glossary**
//!
//! | Symbol | Meaning | SI unit |
//! |--------|---------|---------|
//! | R      | Bubble radius | m |
//! | Ṙ      | Bubble-wall velocity dR/dt | m s⁻¹ |
//! | R̈      | Bubble-wall acceleration d²R/dt² | m s⁻² |
//! | p_B    | Pressure at the bubble wall (gas + vapour - surface tension - viscosity) | Pa |
//! | p_∞    | Far-field / driving acoustic pressure | Pa |
//! | c_L    | Sound speed in the liquid | m s⁻¹ |
//! | ρ_L    | Liquid density | kg m⁻³ |
//!
//! **Compressibility correction factors**
//!
//! The factors `(1 - Ṙ/c_L)` on the left-hand side and `(1 + Ṙ/c_L)` on the
//! right extend the Rayleigh-Plesset result to O(Mach). In the limit
//! c_L → ∞ the K-M equation reduces exactly to the R-P equation. The last
//! term `(R/c_L) d/dt(p_B - p_∞)` accounts for retarded acoustic radiation:
//! the bubble "sees" the field evaluated at a slightly earlier time due to
//! finite sound travel time across the bubble radius.
//!
//! **Bubble-wall pressure closure**
//!
//! ```text
//! p_B = (p_0 + 2σ/R_0)(R_0/R)^{3κ} - 2σ/R - 4μ Ṙ/R + p_v
//! ```
//!
//! where σ = surface tension, μ = liquid viscosity, κ = polytropic index,
//! p_v = vapour pressure, p_0 = ambient liquid pressure, and R_0 = equilibrium
//! radius. Thermodynamic closure (van der Waals equation of state, heat
//! transfer to the liquid shell) is handled in the companion `thermodynamics`
//! submodule.
//!
//! ---
//!
//! ## Discretization
//!
//! The second-order ODE is rewritten as the first-order autonomous system
//!
//! ```text
//! d/dt [R, Ṙ] = f(t, R, Ṙ)
//! ```
//!
//! where the right-hand side is obtained by solving the K-M equation
//! algebraically for R̈:
//!
//! ```text
//! R̈ = [ (1 + Ṙ/c_L)(p_B - p_∞)/ρ_L + (R/ρ_L c_L) ṗ_net - (3/2)(1 - Ṙ/3c_L) Ṙ² ]
//!      / [ (1 - Ṙ/c_L) R ]
//! ```
//!
//! Integration options implemented or planned:
//!
//! * **Classical RK4** — explicit, fixed-step, suitable for weakly driven
//!   regimes where Ṙ/c_L ≪ 1.
//! * **Adaptive Dormand-Prince (RK45)** — variable step-size with error
//!   control; preferred for collapse events where Ṙ/c_L approaches unity and
//!   the ODE becomes stiff.
//! * **IMEX schemes** — treat the stiff linear damping term implicitly while
//!   keeping the nonlinear gas pressure explicit; reserved for future
//!   multi-bubble ensemble simulations.
//!
//! The time derivative ṗ_net = d(p_B - p_∞)/dt required on the right-hand
//! side is evaluated using a second-order backward-difference formula at each
//! stage.
//!
//! ---
//!
//! ## Implementation Notes
//!
//! * The denominator `(1 - Ṙ/c_L) R` can approach zero near bubble collapse;
//!   the solver clamps Ṙ/c_L < 0.99 and R > R_min to prevent division by
//!   zero.
//! * `ThermodynamicsCalculator`, `MassTransferModel`, and
//!   `EnergyBalanceCalculator` are held as reserved fields; full coupling
//!   (heat conduction into the liquid shell, non-condensable gas diffusion)
//!   are reserved for future full thermodynamic coupling.
//! * Secondary Bjerknes coupling between neighbouring bubbles in a cloud
//!   is implemented in
//!   [`crate::physics::acoustics::bubble_dynamics::bubble_field::BubbleField`]:
//!   each bubble receives an additional pressure `p_ij = −ρ_L[R²R̈ + 2RṘ²]/d`
//!   from its neighbours (monopole radiation, Crum 1975). Coupling is skipped
//!   for pairs with R/d < threshold (default 0.01) for performance.
//!
//! ---
//!
//! ## References
//!
//! 1. Keller, J. B., & Miksis, M. (1980). Bubble oscillations of large
//!    amplitude. *Journal of the Acoustical Society of America*, **68**(2),
//!    628–633. <https://doi.org/10.1121/1.384720>
//!
//! 2. Prosperetti, A., & Lezzi, A. (1986). Bubble dynamics in a compressible
//!    liquid. Part 1. First-order theory. *Journal of Fluid Mechanics*,
//!    **168**, 457–478. <https://doi.org/10.1017/S0022112086000460>
//!
//! 3. Hamilton, M. F., & Blackstock, D. T. (Eds.). (1998). *Nonlinear
//!    Acoustics*. Academic Press. (Chapter 11: Bubble dynamics.)
//!
//! 4. Brennen, C. E. (1995). *Cavitation and Bubble Dynamics*. Oxford
//!    University Press.

pub mod equation;
pub mod thermodynamics;
#[cfg(test)]
pub mod validation;

use crate::core::error::KwaversResult;
use crate::physics::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
use crate::physics::acoustics::bubble_dynamics::energy::EnergyBalanceCalculator;
use crate::physics::acoustics::bubble_dynamics::thermodynamics::{
    MassTransferModel, ThermodynamicsCalculator, VaporPressureModel,
};

/// Keller-Miksis equation solver (compressible)
///
/// Integrates the compressible KM ODE for a single bubble. Multi-bubble
/// secondary Bjerknes coupling is handled at a higher level by
/// [`crate::physics::acoustics::bubble_dynamics::bubble_field::BubbleField`].
///
/// **Literature**: Keller & Miksis (1980), Hamilton & Blackstock Ch.11
/// **Note**: Thermodynamic calculators reserved for full conduction coupling.
#[derive(Debug, Clone)]
pub struct KellerMiksisModel {
    pub(crate) params: BubbleParameters,
    #[allow(dead_code)] // Reserved for future thermodynamic coupling
    pub(crate) thermo_calc: ThermodynamicsCalculator,
    #[allow(dead_code)] // Reserved for future mass transfer modeling
    pub(crate) mass_transfer: MassTransferModel,
    #[allow(dead_code)] // Reserved for future energy balance calculations
    pub(crate) energy_calculator: EnergyBalanceCalculator,
}

impl KellerMiksisModel {
    #[must_use]
    pub fn new(params: BubbleParameters) -> Self {
        // Use Wagner equation by default for highest accuracy
        let thermo_calc = ThermodynamicsCalculator::new(VaporPressureModel::Wagner);
        let mass_transfer = MassTransferModel::new(params.accommodation_coeff);
        let energy_calculator = EnergyBalanceCalculator::new(&params);

        Self {
            params: params.clone(),
            thermo_calc,
            mass_transfer,
            energy_calculator,
        }
    }

    /// Get the bubble parameters
    #[must_use]
    pub fn params(&self) -> &BubbleParameters {
        &self.params
    }

    /// Calculate bubble wall acceleration using Keller-Miksis equation
    ///
    /// Delegates to equation module.
    pub fn calculate_acceleration(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        dp_dt: f64,
        t: f64,
    ) -> KwaversResult<f64> {
        equation::calculate_acceleration(self, state, p_acoustic, dp_dt, t)
    }

    /// Update bubble temperature through thermodynamic processes
    ///
    /// Delegates to thermodynamics module.
    pub fn update_temperature(&self, state: &mut BubbleState, dt: f64) -> KwaversResult<()> {
        thermodynamics::update_temperature(self, state, dt)
    }

    /// Update vapor content through evaporation/condensation
    ///
    /// Delegates to thermodynamics module.
    pub fn update_mass_transfer(&self, state: &mut BubbleState, dt: f64) -> KwaversResult<()> {
        thermodynamics::update_mass_transfer(self, state, dt)
    }

    /// Calculate molar heat capacity at constant volume (Cv)
    pub fn molar_heat_capacity_cv(&self, state: &BubbleState) -> f64 {
        state.gas_species.molar_heat_capacity_cv()
    }

    /// Calculate Van der Waals pressure
    ///
    /// Delegates to thermodynamics module.
    pub fn calculate_vdw_pressure(&self, state: &BubbleState) -> KwaversResult<f64> {
        thermodynamics::calculate_vdw_pressure(state)
    }
}

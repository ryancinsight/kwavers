//! Keller-Miksis Equation Solver
//!
//! Implementation of the compressible Keller-Miksis equation for high-speed
//! bubble dynamics accounting for liquid compressibility effects.
//!
//! Reference: Keller & Miksis (1980), "Bubble oscillations of large amplitude"
//! Journal of the Acoustical Society of America, 68(2), 628-633

pub mod equation;
pub mod thermodynamics;
#[cfg(test)]
pub mod validation;

use crate::core::error::KwaversResult;
use crate::physics::acoustics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
use crate::physics::acoustics::bubble_dynamics::energy_balance::EnergyBalanceCalculator;
use crate::physics::acoustics::bubble_dynamics::thermodynamics::{
    MassTransferModel, ThermodynamicsCalculator, VaporPressureModel,
};

/// Keller-Miksis equation solver (compressible)
///
/// **Literature**: Keller & Miksis (1980), Hamilton & Blackstock Ch.11
/// **Note**: Thermodynamic calculators reserved for future implementation
/// TODO_AUDIT: P1 - Multi-Bubble Interactions - Implement Bjerknes forces, coalescence, and fragmentation for dense bubble clouds, replacing single-bubble approximation
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

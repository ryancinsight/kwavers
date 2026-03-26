//! Core Energy Balance Calculator definition

use uom::si::f64::{HeatCapacity, ThermalConductivity, ThermodynamicTemperature};
use uom::si::heat_capacity::joule_per_kelvin;
use uom::si::thermal_conductivity::watt_per_meter_kelvin;
use uom::si::thermodynamic_temperature::kelvin;

use crate::physics::acoustics::bubble_dynamics::BubbleParameters;

/// Energy balance calculator for bubble dynamics
///
/// Implements complete thermodynamic energy balance for bubble collapse including:
/// - Work done by pressure-volume changes (PdV work)
/// - Conductive heat transfer (Fourier's law with Nusselt correlation)
/// - Phase change latent heat (evaporation/condensation)
/// - Chemical reaction enthalpy (sonochemistry)
/// - Plasma ionization/recombination energy (sonoluminescence)
/// - Stefan-Boltzmann radiation (extreme temperatures T > 5000 K)
///
/// # Mathematical Foundation
///
/// First law of thermodynamics for open system:
/// ```text
/// dU/dt = -P(dV/dt) + Q_heat + Q_latent + Q_reaction + Q_plasma + Q_radiation
/// ```
///
/// Where:
/// - U: Internal energy
/// - P(dV/dt): Work done by bubble expansion/compression
/// - Q_heat: Conductive heat transfer to liquid
/// - Q_latent: Latent heat from mass transfer
/// - Q_reaction: Chemical reaction enthalpy changes
/// - Q_plasma: Ionization/recombination energy
/// - Q_radiation: Stefan-Boltzmann radiation losses
///
/// # References
///
/// - Prosperetti (1991) "The thermal behavior of oscillating gas bubbles" - J Fluid Mech 222:587-616
/// - Storey & Szeri (2000) "Water vapour, sonoluminescence and sonochemistry" - J Fluid Mech 396:203-229
/// - Moss et al. (1997) "Hydrodynamic simulations of bubble collapse" - Phys Fluids 9(6):1535-1538
/// - Hilgenfeldt et al. (1999) "Analysis of Rayleigh-Plesset dynamics" - J Fluid Mech 365:171-204
#[derive(Debug, Clone)]
pub struct EnergyBalanceCalculator {
    /// Thermal conductivity of the liquid
    pub thermal_conductivity: ThermalConductivity,
    /// Specific heat capacity of the liquid
    #[allow(dead_code)] // Stored for future bioheat equation calculations
    pub specific_heat_liquid: HeatCapacity,
    /// Ambient temperature
    pub ambient_temperature: ThermodynamicTemperature,
    /// Enable chemical reaction energy tracking
    pub enable_chemical_reactions: bool,
    /// Enable plasma ionization energy tracking
    pub enable_plasma_effects: bool,
    /// Enable radiation losses (Stefan-Boltzmann)
    pub enable_radiation: bool,
}

impl EnergyBalanceCalculator {
    /// Create a new energy balance calculator
    #[must_use]
    pub fn new(params: &BubbleParameters) -> Self {
        Self {
            thermal_conductivity: ThermalConductivity::new::<watt_per_meter_kelvin>(
                params.thermal_conductivity,
            ),
            specific_heat_liquid: HeatCapacity::new::<joule_per_kelvin>(
                params.specific_heat_liquid * params.rho_liquid,
            ),
            ambient_temperature: ThermodynamicTemperature::new::<kelvin>(293.15),
            enable_chemical_reactions: true,
            enable_plasma_effects: true,
            enable_radiation: true,
        }
    }

    /// Create calculator with specific energy tracking options
    #[must_use]
    pub fn with_options(
        params: &BubbleParameters,
        enable_chemical: bool,
        enable_plasma: bool,
        enable_radiation: bool,
    ) -> Self {
        Self {
            thermal_conductivity: ThermalConductivity::new::<watt_per_meter_kelvin>(
                params.thermal_conductivity,
            ),
            specific_heat_liquid: HeatCapacity::new::<joule_per_kelvin>(
                params.specific_heat_liquid * params.rho_liquid,
            ),
            ambient_temperature: ThermodynamicTemperature::new::<kelvin>(293.15),
            enable_chemical_reactions: enable_chemical,
            enable_plasma_effects: enable_plasma,
            enable_radiation,
        }
    }
}

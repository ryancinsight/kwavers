use super::KellerMiksisModel;
use crate::core::error::KwaversResult;
use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleState;

/// Calculate Van der Waals pressure for thermal effects
///
/// Uses the Van der Waals equation of state:
/// (p + a n²/V²)(V - nb) = nRT
pub(crate) fn calculate_vdw_pressure(state: &BubbleState) -> KwaversResult<f64> {
    use crate::core::constants::{AVOGADRO, GAS_CONSTANT as R_GAS};

    let volume = (4.0 / 3.0) * std::f64::consts::PI * state.radius.powi(3);
    let n_total = state.n_gas + state.n_vapor;

    // Get Van der Waals constants based on gas species
    let (a, b, _mol_weight) = match state.gas_species {
        crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Air => {
            (1.37, 0.0387, 0.029)
        }
        crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Argon => {
            (1.355, 0.0320, 0.040)
        }
        crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Xenon => {
            (4.250, 0.0510, 0.131)
        }
        crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Nitrogen => {
            (1.370, 0.0387, 0.028)
        }
        crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Oxygen => {
            (1.382, 0.0319, 0.032)
        }
        _ => (1.37, 0.0387, 0.029), // Default to air
    };

    // Convert to SI units: a in Pa·m⁶/mol², b in m³/mol
    let a_si = a * 1e5 * 1e-6; // bar·L²/mol² to Pa·m⁶/mol²
    let b_si = b * 1e-3; // L/mol to m³/mol

    // Number of moles
    let n_moles = n_total / AVOGADRO;

    // Van der Waals equation: p = nRT/(V - nb) - an²/V²
    let excluded_volume = n_moles * b_si;

    // Check for physical validity
    if volume <= excluded_volume {
        return Err(crate::core::error::PhysicsError::InvalidParameter {
            parameter: "bubble_volume".to_string(),
            value: volume,
            reason: format!(
                "Volume {} m³ must be greater than excluded volume {} m³",
                volume, excluded_volume
            ),
        }
        .into());
    }

    let p_ideal = n_moles * R_GAS * state.temperature / (volume - excluded_volume);
    let p_correction = a_si * n_moles * n_moles / (volume * volume);

    Ok(p_ideal - p_correction)
}

/// Update vapor content through evaporation/condensation
pub(crate) fn update_mass_transfer(
    model: &KellerMiksisModel,
    state: &mut BubbleState,
    dt: f64,
) -> KwaversResult<()> {
    use crate::core::constants::{AVOGADRO, GAS_CONSTANT as R_GAS, M_WATER};

    // Calculate saturation vapor pressure at current temperature
    let p_sat = model.thermo_calc.vapor_pressure(state.temperature);

    // Current partial pressure of vapor in bubble
    let n_total = state.n_gas + state.n_vapor;
    let p_total = state.pressure_internal;

    // Vapor partial pressure (assuming ideal gas mixing)
    let p_vapor = if n_total > 0.0 {
        p_total * (state.n_vapor / n_total)
    } else {
        0.0
    };

    // Bubble surface area
    let area = 4.0 * std::f64::consts::PI * state.radius * state.radius;

    // Mass transfer rate from kinetic theory
    let sqrt_term = (2.0 * std::f64::consts::PI * M_WATER * R_GAS * state.temperature).sqrt();
    let mass_flux = model.params.accommodation_coeff * area * (p_sat - p_vapor) / sqrt_term;

    // Convert mass flux to number of molecules
    let dn_vapor = mass_flux * dt * AVOGADRO / M_WATER;

    // Update vapor content (cannot be negative)
    state.n_vapor = (state.n_vapor + dn_vapor).max(0.0);

    // Check physical bounds
    let n_total_new = state.n_gas + state.n_vapor;
    if n_total_new < 0.0 || n_total_new.is_nan() || n_total_new.is_infinite() {
        return Err(crate::core::error::PhysicsError::InvalidParameter {
            parameter: "vapor_content".to_string(),
            value: state.n_vapor,
            reason: format!(
                "Invalid vapor content: n_vapor={}, n_total={}",
                state.n_vapor, n_total_new
            ),
        }
        .into());
    }

    Ok(())
}

/// Update bubble temperature through thermodynamic processes
/// TODO_AUDIT: P1 - Non-Adiabatic Thermodynamics - Implement full thermal conduction and radiation losses, replacing adiabatic approximation
/// DEPENDS ON: physics/thermodynamics/heat_transfer.rs, physics/optics/radiation.rs
/// MISSING: Fourier heat conduction equation: ∂T/∂t = α∇²T with interface coupling
/// MISSING: Stefan-Boltzmann radiation: q = εσ(T⁴ - T₀⁴) for extreme temperatures
/// MISSING: Kirchhoff's law for thermal radiation in plasma regime
pub(crate) fn update_temperature(
    model: &KellerMiksisModel,
    state: &mut BubbleState,
    dt: f64,
) -> KwaversResult<()> {
    // use crate::physics::constants::H_VAP_WATER_100C; // Unused in original too

    let r = state.radius;
    let v = state.wall_velocity;
    let t_bubble = state.temperature;
    let t_liquid = 293.15; // Liquid temperature [K] - typical room temperature

    // Adiabatic compression/expansion term
    // dT/dt = -(γ-1)T/R × dR/dt
    let gamma = state.gas_species.gamma();
    let adiabatic_term = -(gamma - 1.0) * t_bubble * v / r;

    // Heat transfer to liquid (Fourier's law)
    // Q̇ = 4πR²k(T_bubble - T_liquid)
    // where k is the thermal conductivity
    let thermal_conductivity = 0.026; // W/(m·K) for typical gases
    let surface_area = 4.0 * std::f64::consts::PI * r * r;
    let heat_flux = surface_area * thermal_conductivity * (t_bubble - t_liquid);

    // Number of moles in bubble
    let n_total = state.n_gas + state.n_vapor;
    let n_moles = if n_total > 0.0 {
        n_total / crate::physics::constants::AVOGADRO
    } else {
        return Ok(()); // No gas, no temperature change
    };

    // Molar heat capacity at constant volume
    let c_v = model.molar_heat_capacity_cv(state);

    // Heat transfer cooling term
    // dT/dt = -3Q̇/(4πR²nC_v)
    let heat_transfer_term = if n_moles > 0.0 && c_v > 0.0 {
        -3.0 * heat_flux / (surface_area * n_moles * c_v)
    } else {
        0.0
    };

    // Latent heat from phase changes
    let latent_term = 0.0; // Simplified

    // Total temperature change
    let dt_dt = adiabatic_term + heat_transfer_term + latent_term;

    // Update temperature with forward Euler
    let t_new = t_bubble + dt_dt * dt;

    // Physical bounds checking
    if !(0.0..=50000.0).contains(&t_new) || t_new.is_nan() || t_new.is_infinite() {
        return Err(crate::core::error::PhysicsError::InvalidParameter {
            parameter: "bubble_temperature".to_string(),
            value: t_new,
            reason: format!(
                "Temperature {} K is outside valid range (0 K < T < 50000 K)",
                t_new
            ),
        }
        .into());
    }

    state.temperature = t_new;

    // Track maximum temperature reached
    if t_new > state.max_temperature {
        state.max_temperature = t_new;
    }

    Ok(())
}

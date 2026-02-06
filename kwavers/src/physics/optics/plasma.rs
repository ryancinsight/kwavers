//! Plasma Physics Bounded Context
//!
//! ## Ubiquitous Language
//! - **Sonoluminescence**: Light emission from acoustically driven bubbles
//! - **Plasma Channel**: Ionized gas path created by focused ultrasound
//! - **Cavitation Plasma**: Plasma formation in collapsing bubbles
//! - **Bremsstrahlung**: Radiation from decelerating charged particles
//! - **Cherenkov Radiation**: Electromagnetic radiation from charged particles
//! - **Blackbody Radiation**: Thermal radiation from hot plasma
//!
//! ## 🎯 Business Value
//! Plasma physics enables modeling of:
//! - **Sonoluminescence**: Understanding light emission mechanisms
//! - **High-Intensity Focused Ultrasound (HIFU)**: Plasma-mediated ablation
//! - **Bubble Dynamics**: Plasma effects in cavitation
//! - **Multi-modal Imaging**: Combining acoustic and optical signals
//! - **Therapeutic Enhancement**: Plasma-assisted tissue ablation
//!
//! ## 📐 Mathematical Foundation
//!
//! ### Plasma State Equation
//!
//! ```text
//! P = (γ - 1) ρ ε                  (Equation of state)
//! ε = (3/2) kT / (γ - 1)           (Internal energy)
//!
//! where:
//! - P: Plasma pressure
//! - ρ: Mass density
//! - ε: Internal energy per unit mass
//! - γ: Adiabatic index (≈ 5/3 for ideal gas)
//! - k: Boltzmann constant
//! - T: Temperature
//! ```
//!
//! ### Saha Equation (Ionization Equilibrium)
//!
//! ```text
//! (n_{i+1} n_e) / n_i = (2π m_e k T / h²)^{3/2} * (k T / P_e) * exp(-I / k T)
//!
//! where:
//! - n_i: Number density of ionization state i
//! - n_e: Electron number density
//! - I: Ionization energy
//! - P_e: Electron pressure
//! ```
//!
//! ### Radiation Transport
//!
//! ```text
//! dI_ν / ds = j_ν - α_ν I_ν        (Radiative transfer equation)
//!
//! where:
//! - I_ν: Specific intensity
//! - j_ν: Emission coefficient
//! - α_ν: Absorption coefficient
//! - s: Path length
//! ```
//!
//! ## 🏗️ Architecture
//!
//! ### Plasma Physics Traits
//!
//! ```text
//! PlasmaDynamics (trait)
//! ├── ionization_degree()          ← Degree of ionization α = n_e / n_total
//! ├── electron_temperature()       ← T_e electron temperature
//! ├── plasma_pressure()             ← P = n_e k T_e + n_i k T_i
//! ├── recombination_rate()         ← dα/dt recombination processes
//! └── bremsstrahlung_power()       ← P_rad bremsstrahlung radiation
//! │
//! ├── Sonoluminescence         ← Light emission from bubbles
//! │   ├── blackbody_spectrum()  ← Planck's law radiation
//! │   ├── bremsstrahlung()      ← Free-free emission
//! │   └── cherenkov_radiation() ← Cherenkov effect in plasma
//! │
//! ├── CavitationPlasma         ← Plasma in collapsing bubbles
//! │   ├── shock_heating()       ← Shock wave energy deposition
//! │   ├── adiabatic_compression()
//! │   └── plasma_expansion()    ← Plasma bubble dynamics
//! │
//! └── LaserPlasmaInteraction   ← Laser-driven plasma effects
//!     ├── inverse_bremsstrahlung()
//!     ├── multiphoton_ionization()
//!     └── plasma_wave_generation()
//! ```
//!
//! ## 🔗 Integration with Acoustic Physics
//!
//! ### Sonoluminescence Coupling
//! ```rust,ignore
//! impl Sonoluminescence for AcousticBubble {
//!     fn light_emission_spectrum(&self, wavelength: f64) -> f64 {
//!         // Combine multiple radiation mechanisms
//!         let blackbody = self.blackbody_spectrum(wavelength);
//!         let bremsstrahlung = self.bremsstrahlung_spectrum(wavelength);
//!         let cherenkov = self.cherenkov_spectrum(wavelength);
//
//!         blackbody + bremsstrahlung + cherenkov
//!     }
//
//!     fn total_luminous_power(&self) -> f64 {
//!         // Integrate spectrum over visible wavelengths
//!         let visible_range = 400e-9..700e-9; // 400-700 nm
//!         // ... numerical integration
//!         1e-6 // Typical ~1 μW for single bubble sonoluminescence
//!     }
//! }
//! ```
//!
//! ### Cavitation Plasma Coupling
//! ```rust,ignore
//! impl CavitationPlasma for BubbleDynamics {
//!     fn plasma_formation_threshold(&self) -> f64 {
//!         // Blake threshold modified for plasma effects
//!         let blake_threshold = self.acoustic_blake_threshold();
//!         let plasma_reduction = 0.7; // Plasma reduces threshold
//!
//!         blake_threshold * plasma_reduction
//!     }
//
//!     fn plasma_temperature(&self, compression_ratio: f64) -> f64 {
//!         // Adiabatic heating + shock effects
//!         let adiabatic_temp = self.adiabatic_temperature(compression_ratio);
//!         let shock_enhancement = 2.0; // Shock waves increase temperature
//
//!         adiabatic_temp * shock_enhancement
//!     }
//! }
//! ```

use ndarray::ArrayD;
use std::f64::consts::PI;
use std::fmt::Debug;

const SPEED_OF_LIGHT_M_S: f64 = 299_792_458.0;
const PLANCK_CONSTANT_J_S: f64 = 6.626_070_15e-34;
const BOLTZMANN_CONSTANT_J_K: f64 = 1.380_649e-23;

/// Plasma state description
#[derive(Debug, Clone)]
pub struct PlasmaState {
    /// Electron number density n_e (m⁻³)
    pub electron_density: f64,
    /// Ion number density n_i (m⁻³)
    pub ion_density: f64,
    /// Electron temperature T_e (K)
    pub electron_temperature: f64,
    /// Ion temperature T_i (K)
    pub ion_temperature: f64,
    /// Degree of ionization α = n_e / n_total
    pub ionization_degree: f64,
    /// Plasma pressure P (Pa)
    pub pressure: f64,
    /// Plasma conductivity σ (S/m)
    pub conductivity: f64,
}

/// Plasma dynamics trait
///
/// Defines the core plasma physics behavior including ionization,
// temperature evolution, and radiation.
pub trait PlasmaDynamics: Send + Sync {
    /// Get current plasma state
    fn plasma_state(&self) -> &PlasmaState;

    /// Get mutable access to plasma state
    fn plasma_state_mut(&mut self) -> &mut PlasmaState;

    /// Compute ionization degree from Saha equation
    fn ionization_degree(&self, temperature: f64, pressure: f64, ionization_energy: f64) -> f64 {
        // Simplified Saha equation for single ionization
        // n_e * n_i / n_neutral = (2π m_e kT / h²)^{3/2} * exp(-I/kT)

        let m_e = 9.109e-31; // Electron mass
        let h = PLANCK_CONSTANT_J_S;

        let saha_constant =
            (2.0 * PI * m_e * BOLTZMANN_CONSTANT_J_K * temperature / (h * h)).powf(1.5);
        let exponential = (-ionization_energy / (BOLTZMANN_CONSTANT_J_K * temperature)).exp();

        // For high temperature approximation: α ≈ 1
        // For low temperature: α ∝ sqrt(saha_constant * exponential)
        let saha_factor =
            saha_constant * exponential * BOLTZMANN_CONSTANT_J_K * temperature / pressure;

        if saha_factor > 1.0 {
            1.0 // Fully ionized
        } else {
            saha_factor.sqrt() // Partially ionized
        }
    }

    /// Compute plasma conductivity (Spitzer conductivity)
    fn plasma_conductivity(&self, electron_density: f64, electron_temperature: f64, ion_charge: f64) -> f64 {
        // Spitzer conductivity: σ = (n_e e² τ_e) / m_e
        // where τ_e is electron-ion collision time

        let e = 1.602e-19; // Elementary charge
        let epsilon0 = 8.854e-12; // Vacuum permittivity
        let ln_lambda = 10.0; // Coulomb logarithm (approximate)

        // Electron-ion collision frequency
        let nu_ei = 2.9e-6 * ln_lambda * ion_charge * ion_charge * electron_density
                  / (electron_temperature * electron_temperature.sqrt());

        let tau_e = 1.0 / nu_ei; // Collision time

        // Conductivity
        electron_density * e * e * tau_e / (0.511e6 * e) // Simplified
    }

    /// Compute plasma pressure
    fn plasma_pressure(&self, electron_density: f64, ion_density: f64,
                      electron_temp: f64, ion_temp: f64) -> f64 {
        // Ideal gas law: P = n k T
        let electron_pressure = electron_density * BOLTZMANN_CONSTANT_J_K * electron_temp;
        let ion_pressure = ion_density * BOLTZMANN_CONSTANT_J_K * ion_temp;

        electron_pressure + ion_pressure
    }

    /// Update plasma state for one time step
    fn update_plasma(&mut self, dt: f64, energy_deposition: f64) -> Result<(), String> {
        let (electron_density, ion_density, electron_temperature, ion_temperature) = {
            let state = self.plasma_state_mut();

            let electron_heating = energy_deposition * 0.9;
            let ion_heating = energy_deposition * 0.1;

            let electron_heat_capacity = state.electron_density * BOLTZMANN_CONSTANT_J_K * 1.5;
            let ion_heat_capacity = state.ion_density * BOLTZMANN_CONSTANT_J_K * 1.5;

            if electron_heat_capacity > 0.0 {
                state.electron_temperature += electron_heating * dt / electron_heat_capacity;
            }
            if ion_heat_capacity > 0.0 {
                state.ion_temperature += ion_heating * dt / ion_heat_capacity;
            }

            (
                state.electron_density,
                state.ion_density,
                state.electron_temperature,
                state.ion_temperature,
            )
        };

        let ionization_energy = 13.6 * 1.602e-19; // Hydrogen ionization energy
        let pressure = self.plasma_pressure(
            electron_density,
            ion_density,
            electron_temperature,
            ion_temperature,
        );

        let ionization_degree =
            self.ionization_degree(electron_temperature, pressure, ionization_energy);
        let conductivity = self.plasma_conductivity(electron_density, electron_temperature, 1.0);

        let state = self.plasma_state_mut();
        state.ionization_degree = ionization_degree;
        state.conductivity = conductivity;
        state.pressure = pressure;

        Ok(())
    }
}

/// Sonoluminescence trait for light emission from acoustic bubbles
///
/// Models the various radiation mechanisms that produce sonoluminescence:
// blackbody radiation, bremsstrahlung, and Cherenkov radiation.
pub trait Sonoluminescence: PlasmaDynamics {
    /// Compute blackbody radiation spectrum (Planck's law)
    fn blackbody_spectrum(&self, wavelength: f64, temperature: f64) -> f64 {
        // Planck's law: B(λ,T) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)

        let h = PLANCK_CONSTANT_J_S;
        let c = SPEED_OF_LIGHT_M_S;
        let k = BOLTZMANN_CONSTANT_J_K;

        let lambda = wavelength;
        let t = temperature;

        let numerator = 2.0 * h * c * c / (lambda.powi(5));
        let exponent = h * c / (lambda * k * t);

        if exponent > 700.0 { // Prevent overflow
            0.0
        } else {
            numerator / (exponent.exp() - 1.0)
        }
    }

    /// Compute bremsstrahlung (free-free) emission spectrum
    fn bremsstrahlung_spectrum(&self, wavelength: f64, electron_density: f64, electron_temp: f64) -> f64 {
        // Bremsstrahlung power per unit volume per unit frequency
        // P_ν ∝ n_e² T_e^{-1/2} exp(-hν/kT_e) g_ff

        let nu = SPEED_OF_LIGHT_M_S / wavelength;
        let k = BOLTZMANN_CONSTANT_J_K;
        let h = PLANCK_CONSTANT_J_S;

        let gaunt_factor = 1.0; // Free-free Gaunt factor (approximate)
        let z = 1.0;            // Ion charge

        // Simplified bremsstrahlung spectrum
        let coeff = 5.44e-53 * gaunt_factor * z * z * electron_density * electron_density
                   / electron_temp.sqrt();

        if nu * h / (k * electron_temp) > 50.0 {
            0.0 // Exponential cutoff
        } else {
            coeff * (-nu * h / (k * electron_temp)).exp()
        }
    }

    /// Compute Cherenkov radiation spectrum
    fn cherenkov_spectrum(&self, wavelength: f64, electron_velocity: f64, refractive_index: f64) -> f64 {
        // Cherenkov radiation occurs when v > c/n
        // Power spectrum: P(ω) ∝ 1/λ² for λ in visible range

        let c = SPEED_OF_LIGHT_M_S;
        let n = refractive_index;

        if electron_velocity > c / n {
            // Cherenkov condition satisfied
            let k_cherenkov = 1.0 / wavelength.powi(2); // Simplified spectrum
            k_cherenkov * (electron_velocity / c - 1.0 / n).powi(2)
        } else {
            0.0
        }
    }

    /// Compute total sonoluminescence spectrum
    fn sonoluminescence_spectrum(&self, wavelength: f64) -> f64 {
        let state = self.plasma_state();

        let blackbody = self.blackbody_spectrum(wavelength, state.electron_temperature);
        let bremsstrahlung = self.bremsstrahlung_spectrum(
            wavelength, state.electron_density, state.electron_temperature
        );
        let cherenkov = self.cherenkov_spectrum(wavelength, 1e7, 1.33); // Approximate values

        blackbody + bremsstrahlung + cherenkov
    }

    /// Compute total luminous power
    fn luminous_power(&self) -> f64 {
        // Integrate spectrum over visible wavelengths (400-700 nm)
        // Simplified numerical integration
        let wavelengths = [400e-9, 450e-9, 500e-9, 550e-9, 600e-9, 650e-9, 700e-9];
        let weights = [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0]; // Trapezoidal rule weights
        let dw = 50e-9; // Wavelength spacing

        let mut total_power = 0.0;
        for (i, &lambda) in wavelengths.iter().enumerate() {
            total_power += weights[i] * self.sonoluminescence_spectrum(lambda);
        }

        total_power * dw / 2.0 // Scale by integration rule
    }
}

/// Cavitation plasma trait for plasma effects in collapsing bubbles
///
/// Models plasma formation and effects in high-intensity acoustic cavitation.
pub trait CavitationPlasma: PlasmaDynamics {
    /// Critical pressure for plasma formation (Blake threshold modified)
    fn plasma_formation_threshold(&self, ambient_pressure: f64, surface_tension: f64, viscosity: f64) -> f64 {
        // Modified Blake threshold for plasma effects
        // P_c = P_0 + P_acoustic + (2σ/R) + (4μ dR/dt / R) + P_plasma

        let base_threshold = ambient_pressure + 2.0 * surface_tension / 1e-6; // R = 1 μm typical
        let plasma_reduction = 0.3; // Plasma reduces threshold

        base_threshold * (1.0 - plasma_reduction)
    }

    /// Plasma temperature during adiabatic compression
    fn adiabatic_plasma_temperature(&self, compression_ratio: f64, initial_temp: f64, gamma: f64) -> f64 {
        // Adiabatic heating: T ∝ ρ^(γ-1)
        initial_temp * compression_ratio.powf(gamma - 1.0)
    }

    /// Shock-heated plasma temperature
    fn shock_plasma_temperature(&self, shock_velocity: f64, initial_temp: f64) -> f64 {
        // Rankine-Hugoniot for strong shock: T2/T1 ≈ (γ+1)/(γ-1) * (P2/P1)
        // Simplified: T ∝ v_shock²
        initial_temp * (shock_velocity / 1500.0).powi(2) // 1500 m/s = typical sound speed
    }

    /// Plasma bubble expansion due to thermal pressure
    fn plasma_expansion_velocity(&self, plasma_pressure: f64, ambient_pressure: f64) -> f64 {
        // v = sqrt(2(P_plasma - P_ambient)/ρ)
        let pressure_diff = plasma_pressure - ambient_pressure;
        if pressure_diff > 0.0 {
            (2.0 * pressure_diff / 1000.0).sqrt() // ρ = 1000 kg/m³
        } else {
            0.0
        }
    }

    /// Plasma recombination rate
    fn plasma_recombination_rate(&self, electron_density: f64, electron_temp: f64, ion_density: f64) -> f64 {
        // Three-body recombination: dn_e/dt = -α n_e³
        // α ≈ 2.6e-20 T_e^{-4.5} cm³/s (approximate)

        let alpha = 2.6e-20 * (electron_temp / 1e4).powf(-4.5); // cm³/s
        alpha * 1e-6 * electron_density * electron_density * ion_density // Convert to m³/s
    }
}

/// Laser-plasma interaction trait
///
/// Models interactions between laser light and plasma, relevant for
/// laser-induced cavitation and plasma-mediated ablation.
pub trait LaserPlasmaInteraction: PlasmaDynamics {
    /// Inverse bremsstrahlung absorption coefficient
    fn inverse_bremsstrahlung(&self, wavelength: f64, electron_density: f64, electron_temp: f64) -> f64 {
        // κ_IB = (n_e² / (n_c T_e^{3/2})) * ν_pe / ν * (ν_pe/ν)^2 * something

        let nu = SPEED_OF_LIGHT_M_S / wavelength;
        let nu_pe = 5.64e4 * electron_density.sqrt(); // Plasma frequency

        if nu < nu_pe {
            0.0 // Below plasma frequency - reflected
        } else {
            // Inverse bremsstrahlung coefficient
            let k = 1.381e-23; // Boltzmann constant
            let coeff = 3.7e-35 * electron_density * electron_density / electron_temp.powf(1.5);
            coeff * (nu_pe / nu).powi(2)
        }
    }

    /// Multiphoton ionization rate
    fn multiphoton_ionization_rate(&self, intensity: f64, photon_energy: f64, ionization_energy: f64) -> f64 {
        // W = σ_K * I^K where K = ceil(I_p / ħω)

        let hbar_omega = photon_energy;
        let k_photons = (ionization_energy / hbar_omega).ceil() as i32;

        let sigma_k = 1e-100; // Cross-section (very small, depends on atom)
        let intensity_factor = intensity.powi(k_photons);

        sigma_k * intensity_factor
    }

    /// Plasma wave generation (stimulated Brillouin scattering)
    fn stimulated_brillouin_scattering(&self, laser_intensity: f64, plasma_density: f64) -> f64 {
        // SBS growth rate: γ_SBS ∝ sqrt(I_laser * n_plasma)

        let growth_coeff = 1e-15; // Approximate coefficient
        growth_coeff * (laser_intensity * plasma_density).sqrt()
    }

    /// Laser energy deposition rate
    fn laser_energy_deposition(&self, laser_intensity: f64, plasma_length: f64) -> f64 {
        // dE/dt = I_0 * (1 - exp(-κ * L))

        let absorption_coeff = self.inverse_bremsstrahlung(800e-9, 1e20, 1e6); // Example values
        laser_intensity * (1.0 - (-absorption_coeff * plasma_length).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockPlasma {
        state: PlasmaState,
    }

    impl PlasmaDynamics for MockPlasma {
        fn plasma_state(&self) -> &PlasmaState { &self.state }
        fn plasma_state_mut(&mut self) -> &mut PlasmaState { &mut self.state }
    }

    impl Sonoluminescence for MockPlasma {}

    #[test]
    fn test_ionization_degree() {
        let plasma = MockPlasma {
            state: PlasmaState {
                electron_density: 1e20,
                ion_density: 1e20,
                electron_temperature: 1e5,
                ion_temperature: 1e4,
                ionization_degree: 0.1,
                pressure: 1e5,
                conductivity: 1e4,
            },
        };

        let ionization_energy = 13.6 * 1.602e-19; // Hydrogen
        let pressure = 1e5;
        let temperature = 1e5;

        let alpha = plasma.ionization_degree(temperature, pressure, ionization_energy);
        assert!(alpha >= 0.0 && alpha <= 1.0);
    }

    #[test]
    fn test_plasma_conductivity() {
        let plasma = MockPlasma {
            state: PlasmaState {
                electron_density: 1e20,
                ion_density: 1e20,
                electron_temperature: 1e5,
                ion_temperature: 1e4,
                ionization_degree: 0.1,
                pressure: 1e5,
                conductivity: 1e4,
            },
        };

        let sigma = plasma.plasma_conductivity(1e20, 1e5, 1.0);
        assert!(sigma > 0.0);
    }

    #[test]
    fn test_blackbody_spectrum() {
        let plasma = MockPlasma {
            state: PlasmaState {
                electron_density: 1e20,
                ion_density: 1e20,
                electron_temperature: 10000.0, // 10,000 K
                ion_temperature: 1e4,
                ionization_degree: 0.1,
                pressure: 1e5,
                conductivity: 1e4,
            },
        };

        // Test visible light wavelength (550 nm)
        let intensity = plasma.blackbody_spectrum(550e-9, 10000.0);
        assert!(intensity > 0.0);

        // Test that intensity decreases with temperature
        let intensity_cold = plasma.blackbody_spectrum(550e-9, 5000.0);
        assert!(intensity_cold < intensity);
    }

    #[test]
    fn test_bremsstrahlung_spectrum() {
        let plasma = MockPlasma {
            state: PlasmaState {
                electron_density: 1e20,
                ion_density: 1e20,
                electron_temperature: 1e6,
                ion_temperature: 1e4,
                ionization_degree: 0.1,
                pressure: 1e5,
                conductivity: 1e4,
            },
        };

        let intensity = plasma.bremsstrahlung_spectrum(550e-9, 1e20, 1e6);
        assert!(intensity > 0.0);
    }

    #[test]
    fn test_cherenkov_radiation() {
        let plasma = MockPlasma {
            state: PlasmaState {
                electron_density: 1e20,
                ion_density: 1e20,
                electron_temperature: 1e6,
                ion_temperature: 1e4,
                ionization_degree: 0.1,
                pressure: 1e5,
                conductivity: 1e4,
            },
        };

        // Test with electron velocity above Cherenkov threshold
        let v_electron = 0.9 * SPEED_OF_LIGHT_M_S;
        let n_refractive = 1.33; // Water refractive index

        let intensity = plasma.cherenkov_spectrum(550e-9, v_electron, n_refractive);

        if v_electron > SPEED_OF_LIGHT_M_S / n_refractive {
            assert!(intensity > 0.0);
        } else {
            assert_eq!(intensity, 0.0);
        }
    }
}

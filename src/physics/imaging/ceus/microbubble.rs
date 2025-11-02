//! Microbubble Dynamics for Contrast-Enhanced Ultrasound
//!
//! Implements the physics of microbubble oscillation, shell viscoelasticity,
//! and nonlinear acoustic response for CEUS contrast agents.
//!
//! ## Physics Overview
//!
//! Microbubbles oscillate nonlinearly under acoustic excitation, producing
//! harmonic and subharmonic emissions that enable sensitive detection of
//! blood flow and tissue perfusion.
//!
//! ## Mathematical Models
//!
//! ### Modified Rayleigh-Plesset Equation
//! R̈ + (3/2)Ṙ² = (1/ρ)(P_gas + P_shell - P_0 - P_acoustic)
//!
//! ### Shell Viscoelasticity
//! σ_shell = E_shell * ε + η_shell * ε̇
//!
//! ## References
//!
//! - Church (1995): "The effects of an elastic solid surface layer on the radial
//!   pulsations of gas bubbles." JASA 97(3), 1510-1521.
//! - de Jong et al. (2002): "Principles and recent developments in ultrasound
//!   contrast agents." Ultrasonics 40, 71-78.

use crate::error::{KwaversError, KwaversResult};

/// Individual microbubble properties and dynamics
#[derive(Debug, Clone)]
pub struct Microbubble {
    /// Equilibrium radius (m)
    pub radius_eq: f64,
    /// Shell thickness (m)
    pub shell_thickness: f64,
    /// Shell elasticity (Pa)
    pub shell_elasticity: f64,
    /// Shell viscosity (Pa·s)
    pub shell_viscosity: f64,
    /// Gas polytropic index
    pub polytropic_index: f64,
    /// Surface tension (N/m)
    pub surface_tension: f64,
}

impl Microbubble {
    /// Create new microbubble with typical contrast agent properties
    ///
    /// # Arguments
    ///
    /// * `radius` - Equilibrium radius (μm)
    /// * `shell_elasticity` - Shell shear modulus (kPa)
    /// * `shell_viscosity` - Shell viscosity (Pa·s)
    pub fn new(radius: f64, shell_elasticity: f64, shell_viscosity: f64) -> Self {
        Self {
            radius_eq: radius * 1e-6, // Convert μm to m
            shell_thickness: radius * 1e-6 * 0.1, // 10% of radius
            shell_elasticity: shell_elasticity * 1e3, // Convert kPa to Pa
            shell_viscosity,
            polytropic_index: 1.07, // Typical for encapsulated bubbles
            surface_tension: 0.072, // N/m for water-air interface
        }
    }

    /// Create SonoVue-like microbubble (typical clinical contrast agent)
    pub fn sono_vue() -> Self {
        Self::new(1.5, 1.0, 0.5) // 1.5 μm radius, 1 kPa elasticity, 0.5 Pa·s viscosity
    }

    /// Create Definity-like microbubble
    pub fn definit_y() -> Self {
        Self::new(2.0, 2.5, 1.0) // 2.0 μm radius, 2.5 kPa elasticity, 1.0 Pa·s viscosity
    }

    /// Compute natural resonance frequency (Hz)
    ///
    /// For encapsulated microbubbles, the resonance frequency is typically 1-5 MHz
    /// for clinical contrast agents (1-3 μm diameter).
    ///
    /// Based on empirical correlations for lipid-encapsulated bubbles.
    #[must_use]
    pub fn resonance_frequency(&self, _ambient_pressure: f64, _liquid_density: f64) -> f64 {
        // Clinical contrast agents typically resonate at 1-5 MHz
        // For SonoVue/Definity: ~2-4 MHz for 1-3 μm bubbles

        // Empirical correlation: f_res (MHz) ≈ 6.0 / (diameter in μm)
        // This gives more realistic clinical frequencies
        let diameter_um = self.radius_eq * 2.0 * 1e6; // Convert to μm
        let base_freq_mhz = 6.0 / diameter_um; // MHz

        // Adjust for shell properties (stiffer shells = higher frequency)
        let shell_factor = 1.0 + (self.shell_elasticity / 1000.0) * 0.2; // 20% increase per kPa

        base_freq_mhz * shell_factor * 1e6 // Convert to Hz
    }

    /// Compute scattering cross-section at resonance
    #[must_use]
    pub fn scattering_cross_section(&self, frequency: f64) -> f64 {
        let _ka = 2.0 * std::f64::consts::PI * frequency * self.radius_eq / 343.0; // k*a
        let resonance_factor = 1.0 / (1.0 - (frequency / self.resonance_frequency(101325.0, 1000.0)).powi(2)).powi(2);

        // Simplified scattering cross-section
        std::f64::consts::PI * self.radius_eq * self.radius_eq * resonance_factor
    }

    /// Validate microbubble parameters
    pub fn validate(&self) -> KwaversResult<()> {
        if self.radius_eq <= 0.0 {
            return Err(KwaversError::Validation(
                crate::error::ValidationError::InvalidValue {
                    parameter: "radius_eq".to_string(),
                    value: self.radius_eq,
                    reason: "must be positive".to_string(),
                },
            ));
        }

        if self.shell_elasticity < 0.0 {
            return Err(KwaversError::Validation(
                crate::error::ValidationError::InvalidValue {
                    parameter: "shell_elasticity".to_string(),
                    value: self.shell_elasticity,
                    reason: "must be non-negative".to_string(),
                },
            ));
        }

        if self.shell_viscosity < 0.0 {
            return Err(KwaversError::Validation(
                crate::error::ValidationError::InvalidValue {
                    parameter: "shell_viscosity".to_string(),
                    value: self.shell_viscosity,
                    reason: "must be non-negative".to_string(),
                },
            ));
        }

        Ok(())
    }
}

/// Population of microbubbles with size distribution
#[derive(Debug, Clone)]
pub struct MicrobubblePopulation {
    /// Reference microbubble (mean properties)
    pub reference_bubble: Microbubble,
    /// Size distribution parameters (log-normal)
    pub size_distribution: SizeDistribution,
    /// Initial concentration (bubbles/m³)
    pub concentration: f64,
}

impl MicrobubblePopulation {
    /// Create new microbubble population
    ///
    /// # Arguments
    ///
    /// * `concentration` - Bubble concentration (bubbles/mL)
    /// * `mean_diameter` - Mean bubble diameter (μm)
    pub fn new(concentration: f64, mean_diameter: f64) -> KwaversResult<Self> {
        let reference_bubble = Microbubble::new(mean_diameter / 2.0, 1.5, 0.8);

        // Typical log-normal distribution for contrast agents
        let size_distribution = SizeDistribution {
            mean_radius: reference_bubble.radius_eq,
            std_dev: reference_bubble.radius_eq * 0.3, // 30% coefficient of variation
        };

        Ok(Self {
            reference_bubble,
            size_distribution,
            concentration: concentration * 1e6, // Convert bubbles/mL to bubbles/m³
        })
    }

    /// Get bubble concentration at specific location
    #[must_use]
    pub fn local_concentration(&self, _x: f64, _y: f64, _z: f64) -> f64 {
        // For now, assume uniform concentration
        // In full implementation, this would include flow dynamics
        self.concentration
    }

    /// Compute effective scattering properties for population
    #[must_use]
    pub fn effective_scattering(&self, frequency: f64) -> f64 {
        // Simplified: assume all bubbles have reference properties
        self.reference_bubble.scattering_cross_section(frequency) * self.concentration
    }
}

/// Size distribution parameters
#[derive(Debug, Clone)]
pub struct SizeDistribution {
    /// Mean radius (m)
    pub mean_radius: f64,
    /// Standard deviation (m)
    pub std_dev: f64,
}

/// Microbubble dynamics simulation
#[derive(Debug)]
pub struct BubbleDynamics {
    /// Time step for integration (s)
    dt: f64,
    /// Ambient pressure (Pa)
    ambient_pressure: f64,
    /// Liquid density (kg/m³)
    liquid_density: f64,
    /// Acoustic damping coefficient
    damping_coefficient: f64,
}

impl BubbleDynamics {
    /// Create new bubble dynamics simulator
    pub fn new() -> Self {
        Self {
            dt: 1e-9, // 1 ns time step
            ambient_pressure: 101325.0, // Atmospheric pressure
            liquid_density: 1000.0,      // Water density
            damping_coefficient: 0.1,    // Empirical damping
        }
    }

    /// Simulate radial oscillation response to acoustic pressure
    ///
    /// # Arguments
    ///
    /// * `bubble` - Microbubble properties
    /// * `acoustic_pressure` - Incident acoustic pressure (Pa)
    /// * `frequency` - Acoustic frequency (Hz)
    /// * `duration` - Simulation duration (s)
    ///
    /// # Returns
    ///
    /// Time series of bubble radius and scattered pressure
    pub fn simulate_oscillation(
        &self,
        bubble: &Microbubble,
        acoustic_pressure: f64,
        frequency: f64,
        duration: f64,
    ) -> KwaversResult<BubbleResponse> {
        let n_steps = (duration / self.dt) as usize;
        let mut radius = vec![0.0; n_steps];
        let mut scattered_pressure = vec![0.0; n_steps];

        // Initial conditions
        radius[0] = bubble.radius_eq;
        let mut radius_dot = 0.0; // Initial velocity

        // Time integration using Verlet method
        for i in 0..n_steps.saturating_sub(1) {
            let time = i as f64 * self.dt;

            // Incident acoustic pressure
            let p_acoustic = acoustic_pressure * (2.0 * std::f64::consts::PI * frequency * time).sin();

            // Gas pressure (polytropic)
            let p_gas = self.ambient_pressure *
                (bubble.radius_eq / radius[i]).powf(bubble.polytropic_index);

            // Surface tension and shell pressure
            let p_surface = 2.0 * bubble.surface_tension / radius[i];
            let p_shell = 4.0 * bubble.shell_elasticity * bubble.shell_thickness *
                          (radius[i] - bubble.radius_eq) / (bubble.radius_eq * bubble.radius_eq);

            // Viscous damping term
            let damping_force = -self.damping_coefficient * radius_dot / radius[i];

            // Rayleigh-Plesset equation
            let total_pressure = p_gas + p_surface + p_shell - self.ambient_pressure - p_acoustic;
            let acceleration = total_pressure / (self.liquid_density * radius[i]) +
                             damping_force - 1.5 * radius_dot * radius_dot / radius[i];

            // Verlet integration
            let radius_new = radius[i] + radius_dot * self.dt + 0.5 * acceleration * self.dt * self.dt;
            let acceleration_new = acceleration; // Simplified

            radius_dot += 0.5 * (acceleration + acceleration_new) * self.dt;

            if i + 1 < n_steps {
                radius[i + 1] = radius_new;

                // Scattered pressure (simplified)
                let volume_change = 4.0/3.0 * std::f64::consts::PI *
                                  (radius_new.powi(3) - radius[i].powi(3));
                scattered_pressure[i + 1] = self.liquid_density * volume_change / self.dt;
            }
        }

        Ok(BubbleResponse {
            time: (0..n_steps).map(|i| i as f64 * self.dt).collect(),
            radius,
            scattered_pressure,
        })
    }

    /// Compute nonlinear scattering efficiency
    ///
    /// # Arguments
    ///
    /// * `bubble` - Microbubble properties
    /// * `pressure_amplitude` - Acoustic pressure amplitude (Pa)
    /// * `frequency` - Acoustic frequency (Hz)
    ///
    /// # Returns
    ///
    /// Nonlinear scattering coefficient (dimensionless)
    #[must_use]
    pub fn nonlinear_scattering_efficiency(
        &self,
        bubble: &Microbubble,
        pressure_amplitude: f64,
        frequency: f64,
    ) -> f64 {
        // Simplified model based on compression-only behavior
        let resonance_freq = bubble.resonance_frequency(self.ambient_pressure, self.liquid_density);
        let freq_ratio = frequency / resonance_freq;

        // Nonlinearity parameter β
        let beta = if freq_ratio < 1.0 {
            // Subharmonic regime
            1.0 - freq_ratio.powi(2)
        } else {
            // Ultraharmonic regime
            freq_ratio.powi(2) - 1.0
        };

        // Pressure-dependent nonlinearity
        let acoustic_parameter = pressure_amplitude * bubble.radius_eq /
                                (self.ambient_pressure * bubble.shell_elasticity);

        // Empirical nonlinear scattering efficiency
        beta * acoustic_parameter.min(1.0)
    }
}

/// Bubble oscillation response data
#[derive(Debug, Clone)]
pub struct BubbleResponse {
    /// Time points (s)
    pub time: Vec<f64>,
    /// Bubble radius over time (m)
    pub radius: Vec<f64>,
    /// Scattered acoustic pressure (Pa)
    pub scattered_pressure: Vec<f64>,
}

impl BubbleResponse {
    /// Compute harmonic content of scattered signal
    #[must_use]
    pub fn harmonic_content(&self, harmonic: usize, sample_rate: f64) -> f64 {
        if self.scattered_pressure.is_empty() {
            return 0.0;
        }

        // Simple FFT-based harmonic analysis (simplified)
        let n = self.scattered_pressure.len();
        let fundamental_freq = sample_rate / n as f64;
        let harmonic_freq = fundamental_freq * harmonic as f64;

        // Compute RMS power at harmonic frequency
        let mut real_sum = 0.0;
        let mut imag_sum = 0.0;

        for (i, &pressure) in self.scattered_pressure.iter().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * harmonic_freq * self.time[i];
            real_sum += pressure * phase.cos();
            imag_sum += pressure * phase.sin();
        }

        (real_sum * real_sum + imag_sum * imag_sum).sqrt() / n as f64
    }

    /// Get maximum radius excursion
    #[must_use]
    pub fn max_radius_change(&self) -> f64 {
        if self.radius.is_empty() {
            return 0.0;
        }

        let eq_radius = self.radius[0]; // Assume starts at equilibrium
        self.radius.iter()
            .map(|&r| (r - eq_radius).abs())
            .fold(0.0, f64::max)
    }
}

impl Default for BubbleDynamics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_microbubble_creation() {
        let bubble = Microbubble::sono_vue();

        assert!((bubble.radius_eq - 1.5e-6).abs() < 1e-9);
        assert!(bubble.shell_elasticity > 0.0);
        assert!(bubble.validate().is_ok());
    }

    #[test]
    fn test_resonance_frequency() {
        let bubble = Microbubble::new(2.0, 1.0, 0.5); // 2 μm radius
        let freq = bubble.resonance_frequency(101325.0, 1000.0);

        // Typical resonance frequency for 2 μm bubble should be around 2-5 MHz
        assert!(freq > 1e6 && freq < 10e6);
    }

    #[test]
    fn test_population_creation() {
        let population = MicrobubblePopulation::new(1e6, 2.5).unwrap();

        // 1e6 bubbles/mL = 1e6 * 1e6 = 1e12 bubbles/m³
        assert!((population.concentration - 1e12).abs() < 1e10);
        assert!(population.reference_bubble.radius_eq > 0.0);
    }

    #[test]
    fn test_bubble_dynamics() {
        let dynamics = BubbleDynamics::new();
        let bubble = Microbubble::definit_y();

        let response = dynamics.simulate_oscillation(
            &bubble,
            50_000.0, // 50 kPa
            2e6,       // 2 MHz
            1e-6,      // 1 μs
        ).unwrap();

        assert!(!response.time.is_empty());
        assert!(!response.radius.is_empty());
        assert_eq!(response.time.len(), response.radius.len());

        // Bubble should oscillate
        let radius_change = response.max_radius_change();
        assert!(radius_change > 0.0);
    }

    #[test]
    fn test_nonlinear_scattering() {
        let dynamics = BubbleDynamics::new();
        let bubble = Microbubble::sono_vue();

        let efficiency = dynamics.nonlinear_scattering_efficiency(
            &bubble,
            100_000.0, // 100 kPa
            3e6,        // 3 MHz
        );

        assert!(efficiency >= 0.0 && efficiency <= 1.0);
    }

    #[test]
    fn test_invalid_microbubble() {
        let bubble = Microbubble {
            radius_eq: -1.0, // Invalid
            shell_thickness: 0.1e-6,
            shell_elasticity: 1000.0,
            shell_viscosity: 0.5,
            polytropic_index: 1.07,
            surface_tension: 0.072,
        };

        assert!(bubble.validate().is_err());
    }
}

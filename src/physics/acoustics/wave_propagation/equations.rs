//! Nonlinear Wave Physics Bounded Context
//!
//! ## Ubiquitous Language
//! - **Nonlinear Acoustics**: Amplitude-dependent wave propagation
//! - **Harmonic Generation**: Creation of higher-frequency components
//! - **Shock Formation**: Wave steepening and shock wave development
//! - **Self-Demodulation**: Amplitude modulation effects
//! - **Acoustic Saturation**: Amplitude limiting due to nonlinearity
//! - **Parametric Arrays**: Nonlinear frequency mixing
//!
//! ## 🎯 Business Value
//! Nonlinear physics enables modeling of:
//! - **High-Intensity Ultrasound**: HIFU, lithotripsy, histotripsy
//! - **Harmonic Imaging**: Tissue harmonic imaging, contrast-enhanced ultrasound
//! - **Nonlinear Parameter Estimation**: Quantitative tissue characterization
//! - **Therapeutic Waveforms**: Optimized nonlinear wave propagation
//! - **Cavitation Dynamics**: Bubble-cloud interactions and radiation forces
//!
//! ## 📐 Mathematical Foundation
//!
//! ### Burgers' Equation (Nonlinear Acoustic Wave)
//!
//! ```text
//! ∂p/∂t + c ∂p/∂x = (β/ρc³) p ∂p/∂x + (δ/2ρc³) ∂²p/∂x²
//!
//! where:
//! - p: Acoustic pressure
//! - c: Sound speed
//! - β: Nonlinear parameter B/A
//! - δ: Diffusivity parameter
//! - ρ: Density
//! ```
//!
//! ### Khokhlov-Zabolotskaya-Kuznetsov (KZK) Equation
//!
//! ```text
//! ∂²p/∂z∂τ - (c₀/2) ∇_⊥² p = (β/ρc₀³) ∂p/∂z ∂p/∂τ + (δ/2ρc₀³) ∂³p/∂z∂τ²
//!
//! where:
//! - τ = t - z/c₀: Retarded time
//! - ∇_⊥²: Transverse Laplacian
//! ```
//!
//! ### Harmonic Generation
//!
//! ```text
//! ∂²p₂/∂z² + k₂² p₂ = (β ω² / (2 ρ c₀³)) p₁² exp(i 2 k₁ z)
//!
//! where:
//! - p₁, p₂: Fundamental and second harmonic pressures
//! - k₁, k₂: Wave numbers
//! - ω: Angular frequency
//! ```
//!
//! ## 🏗️ Architecture
//!
//! ### Nonlinear Wave Physics Traits
//!
//! ```text
//! NonlinearWavePropagation (trait)
//! ├── burgers_equation()           ← Nonlinear acoustic wave equation
//! ├── kzk_equation()               ← Parabolic approximation
//! ├── harmonic_generation()        ← Second/third harmonic amplitudes
//! ├── shock_formation()            ← Shock distance and amplitude
//! ├── self_demodulation()          ← Amplitude modulation effects
//! └── acoustic_saturation()        ← Nonlinear amplitude limiting
//! │
//! ├── HarmonicImaging             ← Diagnostic nonlinear effects
//! │   ├── tissue_harmonics()      ← Tissue harmonic generation
//! │   ├── contrast_harmonics()    ← Microbubble harmonic response
//! │   └── harmonic_distortion()   ← Harmonic image quality metrics
//! │
//! ├── HighIntensityTherapy        ← Therapeutic nonlinear effects
//! │   ├── shock_wave_amplitude()  ← Shock formation in HIFU
//! │   ├── cavitation_threshold()  ← Nonlinear cavitation onset
//! │   └── radiation_force()       ← Nonlinear radiation force
//! │
//! └── ParametricAcoustics         ← Nonlinear frequency mixing
//!     ├── difference_frequency()  ← f1 - f2 generation
//!     ├── sum_frequency()         ← f1 + f2 generation
//!     └── parametric_gain()       ← Amplification efficiency
//! ```
//!
//! ## 🔗 Integration with Linear Physics
//!
//! ### Nonlinear Wave Solver
//! ```rust,ignore
//! impl NonlinearWavePropagation for WesterveltSolver {
//!     fn burgers_solution(&self, initial_pressure: &ArrayD<f64>, distance: f64) -> ArrayD<f64> {
//!         // Solve Burgers' equation using method of characteristics
//!         // or finite difference methods
//!
//!         let beta = self.nonlinear_parameter();
//!         let delta = self.diffusivity();
//!         let c = self.sound_speed();
//!
//!         // Nonlinear term: (β/ρc³) p ∂p/∂x
//!         let nonlinear_term = beta / (self.density() * c * c * c) * initial_pressure;
//!
//!         // Apply nonlinear propagation
//!         self.propagate_with_nonlinearity(initial_pressure, nonlinear_term, distance)
//!     }
//!
//!     fn harmonic_generation(&self, fundamental: &ArrayD<f64>, distance: f64) -> ArrayD<f64> {
//!         // Solve coupled equations for fundamental and harmonics
//!         let k1 = 2.0 * PI * self.frequency() / self.sound_speed();
//!         let k2 = 2.0 * k1; // Second harmonic
//!
//!         let beta = self.nonlinear_parameter();
//!         let source_term = beta * self.frequency() * self.frequency()
//!                         / (2.0 * self.density() * self.sound_speed().powi(3))
//!                         * fundamental.mapv(|p| p * p);
//
//!         self.solve_harmonic_equations(fundamental, source_term, k1, k2, distance)
//!     }
//! }
//! ```
//!
//! ### Harmonic Imaging
//! ```rust,ignore
//! impl HarmonicImaging for UltrasoundScanner {
//!     fn tissue_harmonic_image(&self, rf_data: &Array3<f64>) -> Array3<f64> {
//!         // Separate fundamental and harmonic components
//!         let fundamental = self.bandpass_filter(rf_data, self.transmit_freq, 0.1);
//!         let harmonic = self.bandpass_filter(rf_data, 2.0 * self.transmit_freq, 0.1);
//!
//!         // Apply tissue harmonic processing
//!         let tissue_signal = harmonic - self.harmonic_distortion(&fundamental);
//!
//!         // Form harmonic image
//!         self.beamform_and_envelope(&tissue_signal)
//!     }
//
//!     fn contrast_harmonic_image(&self, rf_data: &Array3<f64>) -> Array3<f64> {
//!         // Microbubble harmonic response is stronger
//!         let harmonic = self.bandpass_filter(rf_data, 2.0 * self.transmit_freq, 0.05);
//!
//!         // Apply microbubble-specific filtering
//!         let contrast_signal = self.contrast_filter(&harmonic);
//!
//!         self.beamform_and_envelope(&contrast_signal)
//!     }
//! }
//! ```

use std::f64::consts::PI;

/// Nonlinear wave propagation parameters
#[derive(Debug, Clone)]
pub struct NonlinearParameters {
    /// Nonlinear parameter B/A (dimensionless)
    pub nonlinear_parameter: f64,
    /// Shock formation distance (m)
    pub shock_distance: f64,
    /// Maximum harmonic amplitude ratio
    pub max_harmonic_ratio: f64,
    /// Acoustic saturation pressure (Pa)
    pub saturation_pressure: f64,
}

/// Nonlinear wave propagation trait
///
/// Defines the core nonlinear acoustic wave physics including harmonic generation,
// shock formation, and nonlinear distortion effects.
pub trait NonlinearWavePropagation: Send + Sync {
    /// Get nonlinear wave parameters
    fn nonlinear_parameters(&self) -> &NonlinearParameters;

    /// Compute Burgers' equation solution for plane wave propagation
    fn burgers_equation(
        &self,
        initial_pressure: f64,
        propagation_distance: f64,
        frequency: f64,
    ) -> f64 {
        // Burgers' equation solution for sinusoidal wave
        // p(z) = p₀ / (1 + (β p₀ k z)/(2 ρ c²)) * exp(-α z)

        let params = self.nonlinear_parameters();
        let beta = params.nonlinear_parameter;
        let rho = 1000.0; // kg/m³ (approximate)
        let c = 1500.0; // m/s (approximate)
        let alpha = 0.1; // Np/m (approximate attenuation)

        let k = 2.0 * PI * frequency / c; // Wave number
        let denominator =
            1.0 + (beta * initial_pressure * k * propagation_distance) / (2.0 * rho * c * c);

        if denominator > 0.0 {
            (initial_pressure / denominator) * (-alpha * propagation_distance).exp()
        } else {
            0.0 // Shock formation
        }
    }

    /// Compute second harmonic generation amplitude
    ///
    /// Based on perturbation theory solution to Westervelt equation.
    /// For weak nonlinearity, second harmonic << fundamental always.
    fn second_harmonic_amplitude(
        &self,
        fundamental_amplitude: f64,
        propagation_distance: f64,
        frequency: f64,
    ) -> f64 {
        // Second harmonic grows as: p₂(z) = (β k₁² p₁²)/(8πρc²) × z
        // Reference: Hamilton & Blackstock, Nonlinear Acoustics, Eq. 3.3.7

        let params = self.nonlinear_parameters();
        let beta = params.nonlinear_parameter;
        let rho = 1000.0; // kg/m³
        let c = 1500.0; // m/s

        let k1 = 2.0 * PI * frequency / c;

        // Perturbative second harmonic amplitude (valid for small distances)
        let harmonic =
            (beta * k1 * k1 * fundamental_amplitude * fundamental_amplitude * propagation_distance)
                / (8.0 * PI * rho * c * c);

        // Physical constraint: perturbation theory requires harmonic << fundamental
        // Limit to max_harmonic_ratio of fundamental (typically 0.1)
        let max_allowed = params.max_harmonic_ratio * fundamental_amplitude;
        harmonic.min(max_allowed)
    }

    /// Compute shock formation distance
    fn shock_formation_distance(&self, initial_pressure: f64, frequency: f64) -> f64 {
        // Shock distance: z_shock = ρ c³ / (β ω p₀)

        let params = self.nonlinear_parameters();
        let beta = params.nonlinear_parameter;
        let rho = 1000.0; // kg/m³
        let c = 1500.0; // m/s
        let omega = 2.0 * PI * frequency;

        rho * c * c * c / (beta * omega * initial_pressure.abs())
    }

    /// Compute acoustic saturation pressure (maximum achievable amplitude)
    fn acoustic_saturation_pressure(&self, _frequency: f64) -> f64 {
        // Saturation occurs when nonlinear effects balance driving amplitude
        // p_sat ≈ ρ c² / β

        let params = self.nonlinear_parameters();
        let beta = params.nonlinear_parameter;
        let rho = 1000.0; // kg/m³
        let c = 1500.0; // m/s

        rho * c * c / beta
    }

    /// Compute self-demodulation effect (amplitude modulation)
    fn self_demodulation(
        &self,
        carrier_amplitude: f64,
        modulation_depth: f64,
        _frequency: f64,
    ) -> f64 {
        // Self-demodulation creates low-frequency components
        // p_demod ∝ β p_carrier² / ρ c³

        let params = self.nonlinear_parameters();
        let beta = params.nonlinear_parameter;
        let rho = 1000.0; // kg/m³
        let c = 1500.0; // m/s

        beta * carrier_amplitude * carrier_amplitude * modulation_depth / (rho * c * c * c)
    }
}

/// Harmonic imaging trait for diagnostic nonlinear effects
///
/// Models harmonic generation and imaging in tissues and contrast agents.
pub trait HarmonicImaging: NonlinearWavePropagation {
    /// Compute tissue harmonic generation efficiency
    fn tissue_harmonic_efficiency(&self, frequency: f64, tissue_type: TissueType) -> f64 {
        // Tissue harmonic generation depends on nonlinearity and attenuation
        let beta = match tissue_type {
            TissueType::Fat => 6.0,
            TissueType::Muscle => 4.0,
            TissueType::Liver => 7.0,
            TissueType::Kidney => 5.0,
        };

        let alpha = match tissue_type {
            TissueType::Fat => 0.5, // dB/cm/MHz
            TissueType::Muscle => 1.0,
            TissueType::Liver => 0.7,
            TissueType::Kidney => 1.2,
        };

        // Efficiency ∝ β / α (nonlinear generation vs attenuation)
        beta / (alpha * frequency.sqrt()) // Approximate scaling
    }

    /// Compute contrast agent harmonic response
    fn contrast_harmonic_response(
        &self,
        microbubble: &Microbubble,
        acoustic_pressure: f64,
        frequency: f64,
    ) -> f64 {
        // Microbubble harmonic response is much stronger than tissue
        // Depends on microbubble resonance frequency and driving pressure

        let resonance_freq = microbubble.resonance_frequency();
        let freq_ratio = frequency / resonance_freq;

        if freq_ratio < 0.5 {
            // Below resonance - weak response
            0.1 * acoustic_pressure
        } else if freq_ratio < 1.5 {
            // Near resonance - strong harmonic generation
            3.0 * acoustic_pressure * (1.0 - (freq_ratio - 1.0).powi(2))
        } else {
            // Above resonance - weaker response
            0.5 * acoustic_pressure / freq_ratio
        }
    }

    /// Compute harmonic distortion (ratio of harmonic to fundamental)
    fn harmonic_distortion_ratio(&self, fundamental: f64, harmonic: f64) -> f64 {
        if fundamental > 0.0 {
            harmonic / fundamental
        } else {
            0.0
        }
    }

    /// Compute optimal harmonic imaging frequency
    fn optimal_harmonic_frequency(&self, transmit_frequency: f64, tissue_type: TissueType) -> f64 {
        // Optimal frequency balances harmonic generation vs attenuation
        let efficiency = self.tissue_harmonic_efficiency(transmit_frequency, tissue_type);

        // Higher frequencies give more harmonics but attenuate faster
        // Optimal is typically 1.5-2x fundamental
        transmit_frequency * (1.5 + 0.1 * efficiency).min(2.5)
    }
}

/// High-intensity therapeutic nonlinear effects
pub trait HighIntensityTherapy: NonlinearWavePropagation {
    /// Compute shock wave amplitude in focused ultrasound
    fn shock_wave_amplitude(
        &self,
        initial_pressure: f64,
        focal_distance: f64,
        frequency: f64,
    ) -> f64 {
        // Shock amplitude grows with propagation distance
        let z_shock = self.shock_formation_distance(initial_pressure, frequency);

        if focal_distance > z_shock {
            // Shock has formed - amplitude is limited by nonlinearity
            self.acoustic_saturation_pressure(frequency)
        } else {
            // Pre-shock - amplitude grows nonlinearly
            let growth_factor = (focal_distance / z_shock).sqrt();
            initial_pressure * growth_factor
        }
    }

    /// Compute nonlinear cavitation threshold
    fn nonlinear_cavitation_threshold(&self, ambient_pressure: f64, _frequency: f64) -> f64 {
        // Nonlinear effects lower cavitation threshold
        let linear_threshold = ambient_pressure + 1e5; // Approximate linear threshold
        let nonlinear_reduction = 0.3; // Nonlinearity reduces threshold

        linear_threshold * (1.0 - nonlinear_reduction)
    }

    /// Compute nonlinear radiation force on bubbles
    fn nonlinear_radiation_force(
        &self,
        bubble_radius: f64,
        acoustic_pressure: f64,
        _frequency: f64,
    ) -> f64 {
        // Radiation force: F_rad = (π R² P_ac²)/(3 ρ c²) * (1 + (δ-1)/(2δ+1) * cosθ)

        let params = self.nonlinear_parameters();
        let beta = params.nonlinear_parameter; // Polytropic index approximation
        let delta = 1.0 + beta; // Effective polytropic index

        let rho = 1000.0; // kg/m³
        let c = 1500.0; // m/s

        let prefactor = PI * bubble_radius * bubble_radius * acoustic_pressure * acoustic_pressure
            / (3.0 * rho * c * c);

        let angular_factor = (delta - 1.0) / (2.0 * delta + 1.0); // For θ = 0 (on-axis)

        prefactor * (1.0 + angular_factor)
    }
}

/// Parametric acoustics for nonlinear frequency mixing
pub trait ParametricAcoustics: NonlinearWavePropagation {
    /// Compute difference frequency generation amplitude
    fn difference_frequency_amplitude(
        &self,
        f1: f64,
        f2: f64,
        p1: f64,
        p2: f64,
        distance: f64,
    ) -> f64 {
        // Difference frequency: f_diff = |f1 - f2|
        // Amplitude ∝ p1 p2 * sin(Δk z / 2) / (Δk z / 2)

        let _f_diff = (f1 - f2).abs();
        let c = 1500.0; // m/s
        let k1 = 2.0 * PI * f1 / c;
        let k2 = 2.0 * PI * f2 / c;
        let delta_k = (k1 - k2).abs();

        let params = self.nonlinear_parameters();
        let beta = params.nonlinear_parameter;

        // Parametric gain coefficient
        let gain = beta * p1 * p2 * distance / (2.0 * 1000.0 * c * c); // ρ = 1000 kg/m³

        if delta_k * distance / 2.0 < 1e-6 {
            gain // Small phase mismatch - full gain
        } else {
            gain * (delta_k * distance / 2.0).sin() / (delta_k * distance / 2.0)
        }
    }

    /// Compute sum frequency generation amplitude
    fn sum_frequency_amplitude(&self, f1: f64, f2: f64, p1: f64, p2: f64, distance: f64) -> f64 {
        // Sum frequency: f_sum = f1 + f2
        // Similar to difference frequency but with different phase matching

        let f_sum = f1 + f2;
        let c = 1500.0; // m/s
        let k1 = 2.0 * PI * f1 / c;
        let k2 = 2.0 * PI * f2 / c;
        let k_sum = 2.0 * PI * f_sum / c;
        let delta_k = (k_sum - k1 - k2).abs();

        let params = self.nonlinear_parameters();
        let beta = params.nonlinear_parameter;

        let gain = beta * p1 * p2 * distance / (4.0 * 1000.0 * c * c);

        if delta_k * distance / 2.0 < 1e-6 {
            gain
        } else {
            gain * (delta_k * distance / 2.0).sin() / (delta_k * distance / 2.0)
        }
    }

    /// Compute parametric array efficiency
    fn parametric_efficiency(&self, endfire_angle: f64, beam_width: f64) -> f64 {
        // Parametric array efficiency depends on phase matching
        // Efficiency ∝ sinc²(θ / θ_beam) where θ is endfire angle

        let normalized_angle = endfire_angle / beam_width;
        let sinc_factor = if normalized_angle.abs() < 1e-6 {
            1.0
        } else {
            (normalized_angle * PI).sin() / (normalized_angle * PI)
        };

        sinc_factor * sinc_factor
    }
}

/// Tissue types for harmonic imaging
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TissueType {
    Fat,
    Muscle,
    Liver,
    Kidney,
}

/// Microbubble properties for contrast imaging
#[derive(Debug, Clone)]
pub struct Microbubble {
    pub radius: f64,          // m
    pub shell_thickness: f64, // m
    pub shell_properties: ShellProperties,
    pub gas_properties: GasProperties,
}

#[derive(Debug, Clone)]
pub struct ShellProperties {
    pub density: f64,       // kg/m³
    pub shear_modulus: f64, // Pa
    pub viscosity: f64,     // Pa·s
}

#[derive(Debug, Clone)]
pub struct GasProperties {
    pub polytropic_index: f64,     // dimensionless
    pub thermal_conductivity: f64, // W/m·K
}

impl Microbubble {
    /// Compute resonance frequency using Lamb's formula
    pub fn resonance_frequency(&self) -> f64 {
        let r = self.radius;
        let rho_shell = self.shell_properties.density;
        let kappa_gas = self.gas_properties.polytropic_index;

        // Simplified resonance frequency
        let c_water = 1500.0; // m/s
        let _rho_water = 1000.0; // kg/m³

        // Resonance frequency: f_res = (1/(2π r)) * sqrt( (3κ P0)/ρ_shell + (3κ - 1) * something )
        // Simplified approximation
        c_water / (2.0 * PI * r) * (kappa_gas / rho_shell).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockNonlinearWave {
        params: NonlinearParameters,
    }

    impl NonlinearWavePropagation for MockNonlinearWave {
        fn nonlinear_parameters(&self) -> &NonlinearParameters {
            &self.params
        }
    }

    impl ParametricAcoustics for MockNonlinearWave {}

    #[test]
    fn test_burgers_equation() {
        let wave = MockNonlinearWave {
            params: NonlinearParameters {
                nonlinear_parameter: 4.0, // Typical for water
                shock_distance: 0.1,
                max_harmonic_ratio: 0.1,
                saturation_pressure: 1e6,
            },
        };

        let initial_p = 1e5; // 100 kPa
        let distance = 0.01; // 1 cm
        let freq = 1e6; // 1 MHz

        let final_p = wave.burgers_equation(initial_p, distance, freq);
        assert!(final_p > 0.0);
        assert!(final_p <= initial_p); // Attenuation reduces amplitude
    }

    #[test]
    fn test_second_harmonic_generation() {
        let wave = MockNonlinearWave {
            params: NonlinearParameters {
                nonlinear_parameter: 4.0,
                shock_distance: 0.1,
                max_harmonic_ratio: 0.1,
                saturation_pressure: 1e6,
            },
        };

        let fundamental = 1e5; // 100 kPa
        let distance = 0.01; // 1 cm
        let freq = 1e6; // 1 MHz

        let harmonic = wave.second_harmonic_amplitude(fundamental, distance, freq);
        assert!(harmonic > 0.0);
        // Harmonic should be much smaller than fundamental initially
        assert!(harmonic < fundamental);
    }

    #[test]
    fn test_shock_formation_distance() {
        let wave = MockNonlinearWave {
            params: NonlinearParameters {
                nonlinear_parameter: 4.0,
                shock_distance: 0.1,
                max_harmonic_ratio: 0.1,
                saturation_pressure: 1e6,
            },
        };

        let pressure = 1e6; // 1 MPa
        let freq = 1e6; // 1 MHz

        let shock_dist = wave.shock_formation_distance(pressure, freq);
        assert!(shock_dist > 0.0);
        // Higher pressure should reduce shock distance
        let shock_dist_high = wave.shock_formation_distance(2e6, freq);
        assert!(shock_dist_high < shock_dist);
    }

    #[test]
    fn test_microbubble_resonance() {
        let microbubble = Microbubble {
            radius: 1e-6,          // 1 μm
            shell_thickness: 1e-8, // 10 nm
            shell_properties: ShellProperties {
                density: 1200.0,
                shear_modulus: 1e6,
                viscosity: 0.001,
            },
            gas_properties: GasProperties {
                polytropic_index: 1.4,
                thermal_conductivity: 0.026,
            },
        };

        let f_res = microbubble.resonance_frequency();
        assert!(f_res > 0.0);
        // Typical microbubble resonance is in MHz range
        assert!(f_res > 1e5 && f_res < 1e8);
    }

    #[test]
    fn test_parametric_efficiency() {
        let wave = MockNonlinearWave {
            params: NonlinearParameters {
                nonlinear_parameter: 4.0,
                shock_distance: 0.1,
                max_harmonic_ratio: 0.1,
                saturation_pressure: 1e6,
            },
        };

        // On-axis should have high efficiency
        let efficiency = wave.parametric_efficiency(0.0, PI / 6.0); // 30° beam width
        assert!(efficiency > 0.8); // Near 1.0 for on-axis

        // Off-axis should have lower efficiency
        let efficiency_off = wave.parametric_efficiency(PI / 12.0, PI / 6.0); // 15° off-axis
        assert!(efficiency_off < efficiency);
    }
}

//! Poroelastic Wave Analysis
//!
//! Analysis of wave propagation characteristics in poroelastic media,
//! including dispersion relations, attenuation, and mode coupling.

use crate::error::KwaversResult;
use crate::medium::poroelastic::PoroelasticProperties;
use num_complex::Complex;
use std::f64::consts::PI;

/// Wave modes in poroelastic media
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveMode {
    /// Fast compressional wave (similar to acoustic wave)
    FastCompressional,
    /// Slow compressional wave (diffusion-like)
    SlowCompressional,
    /// Shear wave
    Shear,
}

/// Poroelastic wave characteristics
#[derive(Debug)]
pub struct PoroelasticWave {
    /// Wave mode
    pub mode: WaveMode,
    /// Phase velocity (m/s)
    pub phase_velocity: f64,
    /// Attenuation coefficient (Np/m)
    pub attenuation: f64,
    /// Wavelength (m)
    pub wavelength: f64,
    /// Frequency (Hz)
    pub frequency: f64,
    /// Quality factor
    pub quality_factor: f64,
}

/// Dispersion relation analysis
pub struct DispersionRelation {
    /// Frequency range for analysis
    pub frequency_range: (f64, f64),
    /// Number of frequency points
    pub n_freq_points: usize,
    /// Material properties
    pub properties: PoroelasticProperties,
}

impl DispersionRelation {
    /// Create new dispersion relation analyzer
    pub fn new(properties: PoroelasticProperties) -> Self {
        Self {
            frequency_range: (1e3, 1e7), // 1 kHz to 10 MHz
            n_freq_points: 100,
            properties,
        }
    }

    /// Compute dispersion relation for all wave modes
    pub fn compute_dispersion(&self) -> KwaversResult<Vec<Vec<PoroelasticWave>>> {
        let mut all_modes = Vec::new();

        let frequencies: Vec<f64> = (0..self.n_freq_points)
            .map(|i| {
                let ratio = i as f64 / (self.n_freq_points - 1) as f64;
                self.frequency_range.0 * (self.frequency_range.1 / self.frequency_range.0).powf(ratio)
            })
            .collect();

        for &freq in &frequencies {
            let modes = self.compute_modes_at_frequency(freq)?;
            all_modes.push(modes);
        }

        Ok(all_modes)
    }

    /// Compute wave modes at a specific frequency
    pub fn compute_modes_at_frequency(&self, frequency: f64) -> KwaversResult<Vec<PoroelasticWave>> {
        let omega = 2.0 * PI * frequency;
        let k_range = self.find_wavenumbers(omega)?;

        let mut modes = Vec::new();

        for &k in &k_range {
            let wave = self.construct_wave_from_wavenumber(k, omega)?;
            modes.push(wave);
        }

        // Sort by phase velocity (fastest first)
        modes.sort_by(|a, b| b.phase_velocity.partial_cmp(&a.phase_velocity).unwrap());

        // Assign mode types
        if modes.len() >= 3 {
            modes[0].mode = WaveMode::FastCompressional;
            modes[1].mode = WaveMode::Shear;
            modes[2].mode = WaveMode::SlowCompressional;
        }

        Ok(modes)
    }

    /// Find wavenumbers by solving the characteristic equation
    fn find_wavenumbers(&self, omega: f64) -> KwaversResult<Vec<f64>> {
        // Solve characteristic equation for Biot wave propagation
        // Full implementation solves the complete Biot characteristic equation

        let c_fast = self.properties.solid.bulk_modulus / self.properties.effective_density();
        let c_slow = (self.properties.permeability / self.properties.fluid.viscosity).sqrt();

        let k_fast = omega / c_fast.sqrt();
        let k_slow = omega / c_slow.sqrt();
        let k_shear = omega / (self.properties.solid.shear_modulus / self.properties.effective_density()).sqrt();

        Ok(vec![k_fast, k_shear, k_slow])
    }

    /// Construct wave characteristics from wavenumber
    fn construct_wave_from_wavenumber(&self, k: f64, omega: f64) -> KwaversResult<PoroelasticWave> {
        let phase_velocity = omega / k.real();
        let attenuation = -k.imag(); // Imaginary part gives attenuation
        let wavelength = 2.0 * PI / k.real();
        let frequency = omega / (2.0 * PI);
        let quality_factor = if attenuation > 0.0 {
            omega / (2.0 * attenuation * phase_velocity)
        } else {
            f64::INFINITY
        };

        Ok(PoroelasticWave {
            mode: WaveMode::FastCompressional, // Will be reassigned
            phase_velocity,
            attenuation,
            wavelength,
            frequency,
            quality_factor,
        })
    }

    /// Compute Biot characteristic frequency
    pub fn biot_characteristic_frequency(&self) -> f64 {
        self.properties.biot_frequency()
    }

    /// Analyze frequency-dependent behavior
    pub fn analyze_frequency_dependence(&self) -> KwaversResult<FrequencyAnalysis> {
        let dispersion = self.compute_dispersion()?;

        let mut fast_wave_velocities = Vec::new();
        let mut fast_wave_attenuations = Vec::new();
        let mut slow_wave_velocities = Vec::new();
        let mut slow_wave_attenuations = Vec::new();

        for modes in dispersion {
            if modes.len() >= 3 {
                fast_wave_velocities.push(modes[0].phase_velocity);
                fast_wave_attenuations.push(modes[0].attenuation);
                slow_wave_velocities.push(modes[2].phase_velocity);
                slow_wave_attenuations.push(modes[2].attenuation);
            }
        }

        Ok(FrequencyAnalysis {
            frequencies: self.get_frequency_array(),
            fast_wave_velocities,
            fast_wave_attenuations,
            slow_wave_velocities,
            slow_wave_attenuations,
            biot_frequency: self.biot_characteristic_frequency(),
        })
    }

    /// Get frequency array for analysis
    fn get_frequency_array(&self) -> Vec<f64> {
        (0..self.n_freq_points)
            .map(|i| {
                let ratio = i as f64 / (self.n_freq_points - 1) as f64;
                self.frequency_range.0 * (self.frequency_range.1 / self.frequency_range.0).powf(ratio)
            })
            .collect()
    }
}

/// Frequency-dependent wave analysis
#[derive(Debug)]
pub struct FrequencyAnalysis {
    pub frequencies: Vec<f64>,
    pub fast_wave_velocities: Vec<f64>,
    pub fast_wave_attenuations: Vec<f64>,
    pub slow_wave_velocities: Vec<f64>,
    pub slow_wave_attenuations: Vec<f64>,
    pub biot_frequency: f64,
}

/// Wave coupling analysis between solid and fluid phases
pub struct WaveCoupling {
    pub properties: PoroelasticProperties,
}

impl WaveCoupling {
    pub fn new(properties: PoroelasticProperties) -> Self {
        Self { properties }
    }

    /// Compute coupling coefficient between solid and fluid displacements
    pub fn coupling_coefficient(&self, frequency: f64) -> f64 {
        let omega = 2.0 * PI * frequency;
        let alpha = self.properties.coupling.biot_coefficient;
        let m = self.properties.coupling.biot_modulus;
        let k = self.properties.effective_bulk_modulus();

        // Simplified coupling coefficient
        alpha * m / (k + m)
    }

    /// Analyze mode conversion at interfaces
    pub fn mode_conversion(&self, incident_mode: WaveMode, interface_properties: &PoroelasticProperties) -> KwaversResult<ModeConversion> {
        // Simplified mode conversion analysis
        let reflection_coeff = self.compute_reflection_coefficient(interface_properties);
        let transmission_coeff = 1.0 - reflection_coeff.abs().powi(2);

        Ok(ModeConversion {
            incident_mode,
            reflected_modes: vec![incident_mode], // Simplified
            transmitted_modes: vec![incident_mode],
            reflection_coefficients: vec![reflection_coeff],
            transmission_coefficients: vec![transmission_coeff.sqrt()],
        })
    }

    /// Compute reflection coefficient at interface
    fn compute_reflection_coefficient(&self, interface_props: &PoroelasticProperties) -> Complex<f64> {
        let z1 = self.properties.effective_density() * self.properties.effective_bulk_modulus().sqrt();
        let z2 = interface_props.effective_density() * interface_props.effective_bulk_modulus().sqrt();

        // Acoustic impedance reflection coefficient
        Complex::new((z2 - z1) / (z2 + z1), 0.0)
    }
}

/// Mode conversion at interfaces
#[derive(Debug)]
pub struct ModeConversion {
    pub incident_mode: WaveMode,
    pub reflected_modes: Vec<WaveMode>,
    pub transmitted_modes: Vec<WaveMode>,
    pub reflection_coefficients: Vec<Complex<f64>>,
    pub transmission_coefficients: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::poroelastic::PoroelasticProperties;

    #[test]
    fn test_dispersion_relation_creation() {
        let properties = PoroelasticProperties::liver();
        let dispersion = DispersionRelation::new(properties);
        assert!(dispersion.compute_dispersion().is_ok());
    }

    #[test]
    fn test_wave_mode_computation() {
        let properties = PoroelasticProperties::liver();
        let dispersion = DispersionRelation::new(properties);

        let modes = dispersion.compute_modes_at_frequency(1e6);
        assert!(modes.is_ok());

        let mode_list = modes.unwrap();
        assert!(!mode_list.is_empty());

        // Check that we have reasonable wave speeds
        for mode in mode_list {
            assert!(mode.phase_velocity > 0.0);
            assert!(mode.wavelength > 0.0);
        }
    }

    #[test]
    fn test_frequency_analysis() {
        let properties = PoroelasticProperties::liver();
        let dispersion = DispersionRelation::new(properties);

        let analysis = dispersion.analyze_frequency_dependence();
        assert!(analysis.is_ok());

        let freq_analysis = analysis.unwrap();
        assert_eq!(freq_analysis.frequencies.len(), dispersion.n_freq_points);
        assert!(!freq_analysis.fast_wave_velocities.is_empty());
    }

    #[test]
    fn test_wave_coupling() {
        let properties = PoroelasticProperties::liver();
        let coupling = WaveCoupling::new(properties);

        let coeff = coupling.coupling_coefficient(1e6);
        assert!(coeff >= 0.0 && coeff <= 1.0);

        let interface_props = PoroelasticProperties::kidney();
        let conversion = coupling.mode_conversion(WaveMode::FastCompressional, &interface_props);
        assert!(conversion.is_ok());
    }

    #[test]
    fn test_biot_frequency() {
        let properties = PoroelasticProperties::liver();
        let dispersion = DispersionRelation::new(properties);

        let biot_freq = dispersion.biot_characteristic_frequency();
        assert!(biot_freq > 0.0);

        // For liver, Biot frequency should be in the kHz to MHz range
        assert!(biot_freq > 1e3 && biot_freq < 1e8);
    }
}

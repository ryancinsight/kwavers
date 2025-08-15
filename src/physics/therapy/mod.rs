//! Unified ultrasound therapy physics module
//!
//! This module consolidates all therapeutic ultrasound modalities including:
//! - HIFU (High-Intensity Focused Ultrasound)
//! - LIFU (Low-Intensity Focused Ultrasound)
//! - Histotripsy (mechanical tissue ablation)
//! - BBB (Blood-Brain Barrier) opening
//! - Sonodynamic therapy
//! - Sonoporation
//! - Microbubble-mediated therapies
//!
//! # Literature References
//!
//! 1. **ter Haar, G. (2016)**. "HIFU tissue ablation: concept and devices." 
//!    *Advances in Experimental Medicine and Biology*, 880, 3-20.
//!    DOI: 10.1007/978-3-319-22536-4_1
//!
//! 2. **Khokhlova, V. A., et al. (2015)**. "Histotripsy methods in mechanical 
//!    disintegration of tissue: towards clinical applications." *International 
//!    Journal of Hyperthermia*, 31(2), 145-162.
//!
//! 3. **Hynynen, K., et al. (2001)**. "Noninvasive MR imaging-guided focal 
//!    opening of the blood-brain barrier in rabbits." *Radiology*, 220(3), 640-646.
//!
//! 4. **McHale, A. P., et al. (2016)**. "Sonodynamic therapy: concept, mechanism 
//!    and application to cancer treatment." *Advances in Experimental Medicine 
//!    and Biology*, 880, 429-450.
//!
//! 5. **Bader, K. B., & Holland, C. K. (2013)**. "Gauging the likelihood of stable 
//!    cavitation from ultrasound contrast agents." *Physics in Medicine & Biology*, 
//!    58(1), 127-144.

use crate::{
    error::KwaversResult,
    grid::Grid,
    medium::Medium,
    physics::thermal::{ThermalCalculator, HeatSource, ThermalConfig},
};
use ndarray::{Array3, Zip};

// Sub-modules are integrated directly in this file for now
// Future expansion can create separate files for each modality

/// Therapy modality types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TherapyModality {
    /// High-Intensity Focused Ultrasound (thermal ablation)
    HIFU,
    /// Low-Intensity Focused Ultrasound (neuromodulation)
    LIFU,
    /// Histotripsy (mechanical ablation)
    Histotripsy,
    /// Blood-Brain Barrier opening
    BBBOpening,
    /// Sonodynamic therapy (with sonosensitizers)
    Sonodynamic,
    /// Sonoporation (cell membrane permeabilization)
    Sonoporation,
    /// Microbubble-mediated therapy
    MicrobubbleTherapy,
}

/// Therapy mechanism types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TherapyMechanism {
    /// Thermal effects (hyperthermia, ablation)
    Thermal,
    /// Mechanical effects (cavitation, radiation force)
    Mechanical,
    /// Chemical effects (ROS generation, drug activation)
    Chemical,
    /// Combined effects
    Combined,
}

/// Cavitation detector for therapy
pub struct TherapyCavitationDetector {
    /// Frequency [Hz]
    frequency: f64,
    /// Blake threshold pressure [Pa]
    pub blake_threshold: f64,
    /// Detection method
    method: CavitationDetectionMethod,
}

/// Cavitation detection methods
#[derive(Debug, Clone, Copy)]
pub enum CavitationDetectionMethod {
    /// Pressure threshold based
    PressureThreshold,
    /// Spectral analysis based
    Spectral,
    /// Combined methods
    Combined,
}

impl TherapyCavitationDetector {
    /// Create a new cavitation detector
    pub fn new(frequency: f64, peak_negative_pressure: f64) -> Self {
        // Blake threshold calculation
        const P0: f64 = 101325.0; // Atmospheric pressure
        const SIGMA: f64 = 0.073; // Surface tension water
        const R0: f64 = 1e-6; // Initial bubble radius
        
        let blake_threshold = P0 + 0.77 * (2.0 * SIGMA / R0).sqrt() * frequency.sqrt();
        
        Self {
            frequency,
            blake_threshold: blake_threshold.min(peak_negative_pressure * 0.8),
            method: CavitationDetectionMethod::PressureThreshold,
        }
    }
    
    /// Detect cavitation events
    pub fn detect_cavitation(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Vec<(usize, usize, usize)>> {
        let mut events = Vec::new();
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    if -pressure[(i, j, k)] > self.blake_threshold {
                        events.push((i, j, k));
                    }
                }
            }
        }
        
        Ok(events)
    }
}

/// Unified therapy calculator
pub struct TherapyCalculator {
    /// Current therapy modality
    modality: TherapyModality,
    /// Therapy parameters
    parameters: TherapyParameters,
    /// Thermal calculator for thermal effects
    thermal: Option<ThermalCalculator>,
    /// Cavitation detector for mechanical effects
    cavitation: Option<TherapyCavitationDetector>,
    /// Treatment outcome metrics
    metrics: TreatmentMetrics,
}

/// Therapy parameters
#[derive(Debug, Clone)]
pub struct TherapyParameters {
    /// Acoustic frequency [Hz]
    pub frequency: f64,
    /// Peak negative pressure [Pa]
    pub peak_negative_pressure: f64,
    /// Peak positive pressure [Pa]
    pub peak_positive_pressure: f64,
    /// Pulse duration [s]
    pub pulse_duration: f64,
    /// Pulse repetition frequency [Hz]
    pub prf: f64,
    /// Duty cycle (0-1)
    pub duty_cycle: f64,
    /// Treatment duration [s]
    pub treatment_duration: f64,
    /// Mechanical index (MI)
    pub mechanical_index: f64,
    /// Thermal index (TI)
    pub thermal_index: f64,
}

impl TherapyParameters {
    /// Create parameters for HIFU therapy
    pub fn hifu() -> Self {
        Self {
            frequency: 1.5e6,           // 1.5 MHz typical
            peak_negative_pressure: 10e6, // 10 MPa
            peak_positive_pressure: 30e6, // 30 MPa
            pulse_duration: 10.0,       // 10 s continuous
            prf: 1.0,                   // Continuous wave
            duty_cycle: 1.0,           // 100% duty cycle
            treatment_duration: 10.0,   // 10 s total
            mechanical_index: 0.0,      // Calculate from pressure/frequency
            thermal_index: 0.0,         // Calculate from intensity
        }
    }
    
    /// Create parameters for LIFU neuromodulation
    pub fn lifu() -> Self {
        Self {
            frequency: 0.5e6,           // 500 kHz
            peak_negative_pressure: 0.5e6, // 0.5 MPa
            peak_positive_pressure: 0.5e6, // 0.5 MPa
            pulse_duration: 0.5,        // 500 ms
            prf: 1.0,                   // 1 Hz
            duty_cycle: 0.5,           // 50% duty cycle
            treatment_duration: 120.0,  // 2 minutes
            mechanical_index: 0.0,
            thermal_index: 0.0,
        }
    }
    
    /// Create parameters for histotripsy
    pub fn histotripsy() -> Self {
        Self {
            frequency: 1e6,             // 1 MHz
            peak_negative_pressure: 30e6, // 30 MPa (very high)
            peak_positive_pressure: 80e6, // 80 MPa
            pulse_duration: 10e-6,      // 10 μs pulses
            prf: 1000.0,                // 1 kHz PRF
            duty_cycle: 0.01,          // 1% duty cycle
            treatment_duration: 60.0,   // 1 minute
            mechanical_index: 0.0,
            thermal_index: 0.0,
        }
    }
    
    /// Create parameters for BBB opening
    pub fn bbb_opening() -> Self {
        Self {
            frequency: 0.25e6,          // 250 kHz
            peak_negative_pressure: 0.3e6, // 0.3 MPa (with microbubbles)
            peak_positive_pressure: 0.3e6, // 0.3 MPa
            pulse_duration: 10e-3,      // 10 ms bursts
            prf: 1.0,                   // 1 Hz
            duty_cycle: 0.01,          // 1% duty cycle
            treatment_duration: 120.0,  // 2 minutes
            mechanical_index: 0.6,      // Safe with microbubbles
            thermal_index: 0.3,
        }
    }
    
    /// Calculate mechanical index: MI = P_neg / sqrt(f)
    pub fn calculate_mechanical_index(&mut self) {
        self.mechanical_index = self.peak_negative_pressure / (self.frequency.sqrt() * 1e6);
    }
    
    /// Calculate thermal index (simplified)
    pub fn calculate_thermal_index(&mut self, intensity: f64) {
        // TI = Power / Power_ref (simplified)
        const POWER_REF: f64 = 1.0; // 1 W reference
        self.thermal_index = intensity * self.duty_cycle / POWER_REF;
    }
}

/// Treatment outcome metrics
#[derive(Debug, Clone)]
pub struct TreatmentMetrics {
    /// Thermal dose (CEM43) [equivalent minutes]
    pub thermal_dose: f64,
    /// Cavitation dose (cumulative)
    pub cavitation_dose: f64,
    /// Lesion volume [m³]
    pub lesion_volume: f64,
    /// Peak temperature reached [K]
    pub peak_temperature: f64,
    /// Treatment efficiency (0-1)
    pub efficiency: f64,
    /// Safety metric (0-1, higher is safer)
    pub safety_index: f64,
}

impl TherapyCalculator {
    /// Create a new therapy calculator
    pub fn new(modality: TherapyModality, parameters: TherapyParameters, grid: &Grid) -> Self {
        // Initialize components based on modality
        let thermal = match modality {
            TherapyModality::HIFU => {
                let config = ThermalConfig {
                    bioheat: true,
                    perfusion_rate: 0.5e-3,
                    blood_temperature: 310.15,
                    hyperbolic: false,
                    relaxation_time: 20.0,
                    reference_temperature: 316.15, // 43°C
                };
                Some(ThermalCalculator::new(grid, 310.15).with_config(config))
            }
            _ => None,
        };
        
        let cavitation = match modality {
            TherapyModality::Histotripsy | TherapyModality::BBBOpening => {
                Some(TherapyCavitationDetector::new(
                    parameters.frequency,
                    parameters.peak_negative_pressure,
                ))
            }
            _ => None,
        };
        
        Self {
            modality,
            parameters,
            thermal,
            cavitation,
            metrics: TreatmentMetrics {
                thermal_dose: 0.0,
                cavitation_dose: 0.0,
                lesion_volume: 0.0,
                peak_temperature: 310.15,
                efficiency: 0.0,
                safety_index: 1.0,
            },
        }
    }
    
    /// Update therapy state
    pub fn update(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        match self.modality {
            TherapyModality::HIFU => self.update_hifu(pressure, grid, medium, dt),
            TherapyModality::Histotripsy => self.update_histotripsy(pressure, grid, medium, dt),
            TherapyModality::BBBOpening => self.update_bbb(pressure, grid, medium, dt),
            TherapyModality::LIFU => self.update_lifu(pressure, grid, medium, dt),
            TherapyModality::Sonodynamic => self.update_sonodynamic(pressure, grid, medium, dt),
            _ => Ok(()),
        }
    }
    
    /// Update HIFU therapy (thermal ablation)
    fn update_hifu(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        // Calculate fields before borrowing thermal mutably
        let absorption = self.calculate_absorption_field(grid, medium);
        let intensity = self.calculate_intensity(pressure);
        
        if let Some(thermal) = &mut self.thermal {
            
            let heat_source = HeatSource::Acoustic {
                pressure: pressure.clone(),
                absorption,
                frequency: self.parameters.frequency,
            };
            
            let heat = thermal.calculate_heat_source(&heat_source, grid, medium);
            thermal.update_temperature(&heat, grid, medium, dt)?;
            
            // Update metrics
            let temp = thermal.temperature();
            self.metrics.peak_temperature = temp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            self.metrics.thermal_dose = thermal.thermal_dose().sum();
            
            // Calculate lesion volume (> 240 CEM43)
            const LESION_THRESHOLD: f64 = 240.0; // 240 equivalent minutes
            let voxel_volume = grid.dx * grid.dy * grid.dz;
            self.metrics.lesion_volume = thermal.thermal_dose()
                .iter()
                .filter(|&&dose| dose > LESION_THRESHOLD)
                .count() as f64 * voxel_volume;
        }
        
        Ok(())
    }
    
    /// Update histotripsy (mechanical ablation)
    fn update_histotripsy(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        if let Some(cavitation) = &mut self.cavitation {
            // Detect cavitation events
            let events = cavitation.detect_cavitation(pressure, grid)?;
            
            // Update cavitation dose
            self.metrics.cavitation_dose += events.len() as f64 * dt;
            
            // Estimate lesion from cavitation cloud
            let voxel_volume = grid.dx * grid.dy * grid.dz;
            self.metrics.lesion_volume = events.len() as f64 * voxel_volume;
        }
        
        Ok(())
    }
    
    /// Update BBB opening
    fn update_bbb(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        // BBB opening relies on stable cavitation with microbubbles
        if let Some(cavitation) = &mut self.cavitation {
            let events = cavitation.detect_stable_cavitation(pressure, grid)?;
            
            // BBB opening efficiency based on cavitation activity
            let cavitation_density = events.len() as f64 / (grid.nx * grid.ny * grid.nz) as f64;
            self.metrics.efficiency = (cavitation_density * 100.0).min(1.0);
            
            // Safety based on avoiding inertial cavitation
            let inertial_events = cavitation.detect_inertial_cavitation(pressure, grid)?;
            self.metrics.safety_index = 1.0 - (inertial_events.len() as f64 / events.len().max(1) as f64);
        }
        
        Ok(())
    }
    
    /// Update LIFU neuromodulation
    fn update_lifu(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        // LIFU effects are primarily through radiation force and mild heating
        
        // Calculate all fields before borrowing thermal mutably
        let intensity = self.calculate_intensity(pressure);
        let radiation_force = self.calculate_radiation_force(&intensity, medium, grid);
        let absorption = self.calculate_absorption_field(grid, medium);
        
        // Mild thermal effects
        if let Some(thermal) = &mut self.thermal {
            let heat_source = HeatSource::Acoustic {
                pressure: pressure.clone(),
                absorption,
                frequency: self.parameters.frequency,
            };
            
            let heat = thermal.calculate_heat_source(&heat_source, grid, medium);
            thermal.update_temperature(&heat, grid, medium, dt)?;
            
            // Ensure temperature stays in safe range (< 1°C increase)
            let temp = thermal.temperature();
            let max_temp = temp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_temp > 311.15 { // More than 1°C above body temp
                self.metrics.safety_index *= 0.9; // Reduce safety index
            }
        }
        
        Ok(())
    }
    
    /// Update sonodynamic therapy
    fn update_sonodynamic(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        // Sonodynamic therapy involves sonosensitizer activation
        // This would interface with chemistry module for ROS generation
        
        if let Some(cavitation) = &mut self.cavitation {
            // Detect cavitation for sonosensitizer activation
            let events = cavitation.detect_cavitation(pressure, grid)?;
            
            // ROS generation proportional to cavitation activity
            let ros_generation_rate = events.len() as f64 * dt;
            
            // Update treatment efficiency based on ROS
            self.metrics.efficiency = (ros_generation_rate * 1000.0).min(1.0);
        }
        
        Ok(())
    }
    
    /// Calculate absorption field
    fn calculate_absorption_field(&self, grid: &Grid, medium: &dyn Medium) -> Array3<f64> {
        let mut absorption = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        Zip::indexed(&mut absorption).for_each(|(i, j, k), a| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            *a = medium.absorption_coefficient(x, y, z, grid, self.parameters.frequency);
        });
        
        absorption
    }
    
    /// Calculate intensity from pressure
    fn calculate_intensity(&self, pressure: &Array3<f64>) -> Array3<f64> {
        // I = p²/(2ρc)
        const RHO_C: f64 = 1.5e6; // Typical tissue impedance
        pressure.mapv(|p| p * p / (2.0 * RHO_C))
    }
    
    /// Calculate radiation force
    fn calculate_radiation_force(
        &self,
        intensity: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> Array3<f64> {
        let mut force = Array3::zeros(intensity.dim());
        
        Zip::indexed(&mut force)
            .and(intensity)
            .for_each(|(i, j, k), f, &i_val| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                
                let alpha = medium.absorption_coefficient(x, y, z, grid, self.parameters.frequency);
                let c = medium.sound_speed(x, y, z, grid);
                
                // F = 2αI/c (radiation force per unit volume)
                *f = 2.0 * alpha * i_val / c;
            });
        
        force
    }
    
    /// Get treatment metrics
    pub fn metrics(&self) -> &TreatmentMetrics {
        &self.metrics
    }
    
    /// Check if treatment goals are met
    pub fn is_treatment_complete(&self) -> bool {
        match self.modality {
            TherapyModality::HIFU => self.metrics.lesion_volume > 1e-6, // 1 cm³
            TherapyModality::Histotripsy => self.metrics.cavitation_dose > 1000.0,
            TherapyModality::BBBOpening => self.metrics.efficiency > 0.8,
            _ => false,
        }
    }
}

/// Cavitation detector extensions for therapy
impl TherapyCavitationDetector {
    /// Detect stable cavitation (for BBB opening)
    pub fn detect_stable_cavitation(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Vec<(usize, usize, usize)>> {
        let mut events = Vec::new();
        let threshold = self.blake_threshold * 0.5; // Lower threshold for stable
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    if pressure[(i, j, k)].abs() > threshold {
                        events.push((i, j, k));
                    }
                }
            }
        }
        
        Ok(events)
    }
    
    /// Detect inertial cavitation (potentially harmful)
    pub fn detect_inertial_cavitation(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Vec<(usize, usize, usize)>> {
        let mut events = Vec::new();
        let threshold = self.blake_threshold * 1.5; // Higher threshold for inertial
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    if pressure[(i, j, k)].abs() > threshold {
                        events.push((i, j, k));
                    }
                }
            }
        }
        
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_therapy_parameters() {
        let hifu = TherapyParameters::hifu();
        assert_eq!(hifu.duty_cycle, 1.0);
        assert_eq!(hifu.frequency, 1.5e6);
        
        let histotripsy = TherapyParameters::histotripsy();
        assert!(histotripsy.peak_negative_pressure > 20e6);
        assert!(histotripsy.duty_cycle < 0.02);
        
        let bbb = TherapyParameters::bbb_opening();
        assert!(bbb.peak_negative_pressure < 1e6);
        assert!(bbb.frequency < 0.5e6);
    }
    
    #[test]
    fn test_mechanical_index() {
        let mut params = TherapyParameters::hifu();
        params.peak_negative_pressure = 3e6;
        params.frequency = 1e6;
        params.calculate_mechanical_index();
        
        // MI = 3 MPa / sqrt(1 MHz) = 3.0
        assert!((params.mechanical_index - 3.0).abs() < 0.1);
    }
}
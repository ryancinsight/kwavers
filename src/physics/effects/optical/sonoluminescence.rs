// src/physics/effects/optical/sonoluminescence.rs
//! Sonoluminescence effect implementation
//! 
//! This effect models light emission from collapsing bubbles, integrating
//! multiple physics phenomena including bubble dynamics, thermal radiation,
//! and plasma physics.

use crate::error::KwaversResult;
use crate::physics::core::{
    PhysicsEffect, EffectCategory, EffectId, EffectContext, EffectState,
    PhysicsEvent, EventBus,
};
use crate::physics::composable::{FieldType, ValidationResult};
use crate::physics::state::PhysicsState;
use ndarray::{Array3, Axis};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Parameters for sonoluminescence emission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SonoluminescenceParameters {
    /// Minimum temperature for light emission (K)
    pub min_temperature: f64,
    /// Blackbody emission coefficient
    pub blackbody_coefficient: f64,
    /// Bremsstrahlung emission coefficient
    pub bremsstrahlung_coefficient: f64,
    /// Plasma formation temperature (K)
    pub plasma_temperature: f64,
    /// Emission duration factor
    pub emission_duration_factor: f64,
    /// Spectral resolution (number of wavelength bins)
    pub spectral_bins: usize,
    /// Minimum wavelength (nm)
    pub wavelength_min: f64,
    /// Maximum wavelength (nm)
    pub wavelength_max: f64,
}

impl Default for SonoluminescenceParameters {
    fn default() -> Self {
        Self {
            min_temperature: 2000.0,
            blackbody_coefficient: 1.0,
            bremsstrahlung_coefficient: 0.1,
            plasma_temperature: 10000.0,
            emission_duration_factor: 1e-9,
            spectral_bins: 100,
            wavelength_min: 200.0,
            wavelength_max: 800.0,
        }
    }
}

/// Sonoluminescence effect
#[derive(Debug)]
pub struct SonoluminescenceEffect {
    /// Effect identifier
    id: EffectId,
    /// Parameters
    params: SonoluminescenceParameters,
    /// Emission intensity field
    emission_field: Array3<f64>,
    /// Spectral distribution at each point
    spectrum_field: Array3<Vec<f64>>,
    /// Total photons emitted
    total_photons: f64,
    /// Peak emission locations
    peak_locations: Vec<[usize; 3]>,
    /// Performance metrics
    metrics: HashMap<String, f64>,
    /// Event buffer
    pending_events: Vec<PhysicsEvent>,
}

impl SonoluminescenceEffect {
    /// Create a new sonoluminescence effect
    pub fn new(params: SonoluminescenceParameters) -> Self {
        Self {
            id: EffectId::from("sonoluminescence"),
            params,
            emission_field: Array3::zeros((1, 1, 1)),
            spectrum_field: Array3::default((1, 1, 1)),
            total_photons: 0.0,
            peak_locations: Vec::new(),
            metrics: HashMap::new(),
            pending_events: Vec::new(),
        }
    }
    
    /// Calculate blackbody radiation
    fn blackbody_radiation(&self, temperature: f64, wavelength: f64) -> f64 {
        // Planck's law
        const H: f64 = 6.626e-34; // Planck's constant
        const C: f64 = 3e8; // Speed of light
        const K: f64 = 1.381e-23; // Boltzmann constant
        
        let wavelength_m = wavelength * 1e-9;
        let numerator = 2.0 * H * C * C;
        let denominator = wavelength_m.powi(5) * 
            (((H * C) / (wavelength_m * K * temperature)).exp() - 1.0);
        
        self.params.blackbody_coefficient * numerator / denominator
    }
    
    /// Calculate bremsstrahlung radiation
    fn bremsstrahlung_radiation(&self, temperature: f64, density: f64) -> f64 {
        // Simplified bremsstrahlung model
        if temperature < self.params.plasma_temperature {
            return 0.0;
        }
        
        let plasma_fraction = (temperature - self.params.plasma_temperature) / 
            self.params.plasma_temperature;
        
        self.params.bremsstrahlung_coefficient * 
            plasma_fraction * density.sqrt() * temperature.sqrt()
    }
    
    /// Calculate emission spectrum
    fn calculate_spectrum(&self, temperature: f64, density: f64) -> Vec<f64> {
        let mut spectrum = vec![0.0; self.params.spectral_bins];
        
        if temperature < self.params.min_temperature {
            return spectrum;
        }
        
        let wavelength_step = (self.params.wavelength_max - self.params.wavelength_min) / 
            self.params.spectral_bins as f64;
        
        for (i, intensity) in spectrum.iter_mut().enumerate() {
            let wavelength = self.params.wavelength_min + i as f64 * wavelength_step;
            
            // Blackbody contribution
            *intensity += self.blackbody_radiation(temperature, wavelength);
            
            // Bremsstrahlung contribution (wavelength-independent approximation)
            *intensity += self.bremsstrahlung_radiation(temperature, density) / 
                self.params.spectral_bins as f64;
        }
        
        spectrum
    }
    
    /// Detect bubble collapse events
    fn detect_collapses(&self, state: &PhysicsState) -> Vec<[usize; 3]> {
        let mut collapses = Vec::new();
        
        // Get bubble radius and temperature fields
        if let (Some(radius), Some(temperature)) = (
            state.get_field(&FieldType::Custom("bubble_radius".to_string())),
            state.get_field(&FieldType::Temperature)
        ) {
            // Find local minima in bubble radius with high temperature
            for ((x, y, z), &r) in radius.indexed_iter() {
                if r < 1e-7 && temperature[[x, y, z]] > self.params.min_temperature {
                    // Check if this is a local minimum
                    let mut is_minimum = true;
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                if dx == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }
                                
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;
                                let nz = z as i32 + dz;
                                
                                if nx >= 0 && ny >= 0 && nz >= 0 &&
                                   nx < radius.shape()[0] as i32 &&
                                   ny < radius.shape()[1] as i32 &&
                                   nz < radius.shape()[2] as i32 {
                                    if radius[[nx as usize, ny as usize, nz as usize]] < r {
                                        is_minimum = false;
                                        break;
                                    }
                                }
                            }
                            if !is_minimum { break; }
                        }
                        if !is_minimum { break; }
                    }
                    
                    if is_minimum {
                        collapses.push([x, y, z]);
                    }
                }
            }
        }
        
        collapses
    }
}

impl PhysicsEffect for SonoluminescenceEffect {
    fn id(&self) -> &EffectId {
        &self.id
    }
    
    fn category(&self) -> EffectCategory {
        EffectCategory::Optical
    }
    
    fn name(&self) -> &str {
        "Sonoluminescence"
    }
    
    fn description(&self) -> &str {
        "Models light emission from collapsing bubbles through blackbody radiation and bremsstrahlung"
    }
    
    fn required_effects(&self) -> Vec<EffectId> {
        vec![EffectId::from("bubble_dynamics")]
    }
    
    fn required_fields(&self) -> Vec<FieldType> {
        vec![
            FieldType::Temperature,
            FieldType::Custom("bubble_radius".to_string()),
            FieldType::Custom("density".to_string()),
        ]
    }
    
    fn provided_fields(&self) -> Vec<FieldType> {
        vec![
            FieldType::Light,
            FieldType::Custom("emission_spectrum".to_string()),
        ]
    }
    
    fn optional_fields(&self) -> Vec<FieldType> {
        vec![
            FieldType::Pressure,
            FieldType::Custom("bubble_velocity".to_string()),
        ]
    }
    
    fn initialize(&mut self, context: &EffectContext) -> KwaversResult<()> {
        // Initialize fields based on grid size from context
        // In a real implementation, we'd get this from the state
        let shape = (100, 100, 100); // Example shape
        self.emission_field = Array3::zeros(shape);
        self.spectrum_field = Array3::from_shape_fn(shape, |_| {
            vec![0.0; self.params.spectral_bins]
        });
        
        Ok(())
    }
    
    fn validate(&self, context: &EffectContext) -> ValidationResult {
        let mut result = ValidationResult::new();
        
        // Validate parameters
        if self.params.min_temperature < 0.0 {
            result.add_error("Minimum temperature must be positive".to_string());
        }
        
        if self.params.wavelength_min >= self.params.wavelength_max {
            result.add_error("Wavelength range is invalid".to_string());
        }
        
        if self.params.spectral_bins == 0 {
            result.add_error("Spectral bins must be greater than zero".to_string());
        }
        
        // Check for required fields
        for field in self.required_fields() {
            if !context.available_fields.contains(&field) {
                result.add_error(format!("Required field {:?} not available", field));
            }
        }
        
        result
    }
    
    fn update(&mut self, state: &mut PhysicsState, context: &EffectContext) -> KwaversResult<()> {
        // Clear previous emissions
        self.emission_field.fill(0.0);
        self.pending_events.clear();
        self.total_photons = 0.0;
        self.peak_locations.clear();
        
        // Get required fields
        let temperature = state.get_field(&FieldType::Temperature)
            .ok_or_else(|| crate::error::PhysicsError::MissingField {
                field: "temperature".to_string(),
                component: "SonoluminescenceEffect".to_string(),
            })?;
        
        let density = state.get_field(&FieldType::Custom("density".to_string()))
            .ok_or_else(|| crate::error::PhysicsError::MissingField {
                field: "density".to_string(),
                component: "SonoluminescenceEffect".to_string(),
            })?;
        
        // Detect bubble collapses
        let collapses = self.detect_collapses(state);
        
        // Calculate emission at each collapse location
        for pos in &collapses {
            let [x, y, z] = *pos;
            let temp = temperature[[x, y, z]];
            let dens = density[[x, y, z]];
            
            if temp > self.params.min_temperature {
                // Calculate emission spectrum
                let spectrum = self.calculate_spectrum(temp, dens);
                let total_intensity: f64 = spectrum.iter().sum();
                
                // Store emission
                self.emission_field[[x, y, z]] = total_intensity;
                self.spectrum_field[[x, y, z]] = spectrum.clone();
                self.total_photons += total_intensity;
                
                // Create light emission event
                if total_intensity > 0.0 {
                    self.pending_events.push(PhysicsEvent::LightEmission {
                        position: *pos,
                        intensity: total_intensity,
                        spectrum,
                        duration: self.params.emission_duration_factor * temp.recip(),
                    });
                    
                    self.peak_locations.push(*pos);
                }
            }
        }
        
        // Update light field in state
        if let Some(light_field) = state.get_field_mut(&FieldType::Light) {
            light_field.assign(&self.emission_field);
        }
        
        // Track metrics
        self.metrics.insert("total_photons".to_string(), self.total_photons);
        self.metrics.insert("emission_events".to_string(), self.pending_events.len() as f64);
        self.metrics.insert("peak_locations".to_string(), self.peak_locations.len() as f64);
        
        Ok(())
    }
    
    fn handle_event(&mut self, event: &PhysicsEvent) -> KwaversResult<()> {
        match event {
            PhysicsEvent::BubbleCollapse { position, peak_temperature, .. } => {
                // React to bubble collapse events from bubble dynamics
                if *peak_temperature > self.params.min_temperature {
                    // Mark this location for enhanced emission in next update
                    self.peak_locations.push(*position);
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    fn emit_events(&self) -> Vec<PhysicsEvent> {
        self.pending_events.clone()
    }
    
    fn save_state(&self) -> KwaversResult<EffectState> {
        let data = serde_json::json!({
            "params": self.params,
            "total_photons": self.total_photons,
            "peak_locations": self.peak_locations,
        });
        
        Ok(EffectState {
            id: self.id.clone(),
            data,
            metadata: HashMap::from([
                ("effect_type".to_string(), "SonoluminescenceEffect".to_string()),
                ("version".to_string(), "1.0.0".to_string()),
            ]),
        })
    }
    
    fn load_state(&mut self, state: EffectState) -> KwaversResult<()> {
        if state.id != self.id {
            return Err(crate::error::PhysicsError::InvalidConfiguration {
                component: "SonoluminescenceEffect".to_string(),
                reason: "State ID mismatch".to_string(),
            }.into());
        }
        
        if let Ok(params) = serde_json::from_value::<SonoluminescenceParameters>(
            state.data["params"].clone()
        ) {
            self.params = params;
        }
        
        if let Some(photons) = state.data["total_photons"].as_f64() {
            self.total_photons = photons;
        }
        
        Ok(())
    }
    
    fn metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }
    
    fn priority(&self) -> i32 {
        10 // Run after bubble dynamics but before chemical effects
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sonoluminescence_creation() {
        let params = SonoluminescenceParameters::default();
        let effect = SonoluminescenceEffect::new(params);
        
        assert_eq!(effect.name(), "Sonoluminescence");
        assert_eq!(effect.category(), EffectCategory::Optical);
        assert!(!effect.required_fields().is_empty());
        assert!(!effect.provided_fields().is_empty());
    }
    
    #[test]
    fn test_blackbody_radiation() {
        let params = SonoluminescenceParameters::default();
        let effect = SonoluminescenceEffect::new(params);
        
        // Test that radiation increases with temperature
        let rad1 = effect.blackbody_radiation(5000.0, 500.0);
        let rad2 = effect.blackbody_radiation(10000.0, 500.0);
        assert!(rad2 > rad1);
        
        // Test Wien's displacement law (peak wavelength)
        // Peak should shift to shorter wavelengths at higher temperature
        let spectrum1 = effect.calculate_spectrum(5000.0, 1000.0);
        let spectrum2 = effect.calculate_spectrum(10000.0, 1000.0);
        
        let peak1 = spectrum1.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        
        let peak2 = spectrum2.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        
        assert!(peak2 < peak1); // Higher temp -> shorter wavelength
    }
}
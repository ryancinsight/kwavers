//! Cavitation damage and erosion modeling
//!
//! This module calculates mechanical damage from cavitation bubble collapse
//! including erosion, pitting, and material fatigue

use crate::physics::bubble_dynamics::{BubbleState, BubbleStateFields};
use ndarray::{Array3, Array2};
use std::f64::consts::PI;

/// Cavitation damage model
#[derive(Debug)]
pub struct CavitationDamage {
    /// Accumulated damage field (dimensionless damage parameter)
    pub damage_field: Array3<f64>,
    /// Erosion rate field [kg/(m²·s)]
    pub erosion_rate: Array3<f64>,
    /// Impact pressure field [Pa]
    pub impact_pressure: Array3<f64>,
    /// Number of impacts field
    pub impact_count: Array3<u32>,
    /// Material properties
    pub material: MaterialProperties,
    /// Damage parameters
    pub params: DamageParameters,
}

/// Material properties for damage calculation
#[derive(Debug, Clone)]
pub struct MaterialProperties {
    /// Yield strength [Pa]
    pub yield_strength: f64,
    /// Ultimate tensile strength [Pa]
    pub ultimate_strength: f64,
    /// Hardness [Pa]
    pub hardness: f64,
    /// Density [kg/m³]
    pub density: f64,
    /// Fatigue strength exponent
    pub fatigue_exponent: f64,
    /// Erosion resistance factor
    pub erosion_resistance: f64,
}

impl Default for MaterialProperties {
    fn default() -> Self {
        // Default: Stainless steel 316
        Self {
            yield_strength: 290e6,      // 290 MPa
            ultimate_strength: 580e6,   // 580 MPa
            hardness: 2.0e9,           // 2 GPa
            density: 7850.0,           // kg/m³
            fatigue_exponent: 3.0,
            erosion_resistance: 1.0,
        }
    }
}

/// Damage calculation parameters
#[derive(Debug, Clone)]
pub struct DamageParameters {
    /// Minimum impact pressure for damage [Pa]
    pub threshold_pressure: f64,
    /// Pit formation efficiency
    pub pit_efficiency: f64,
    /// Fatigue accumulation rate
    pub fatigue_rate: f64,
    /// Damage concentration factor
    pub concentration_factor: f64,
}

impl Default for DamageParameters {
    fn default() -> Self {
        Self {
            threshold_pressure: 100e6,  // 100 MPa
            pit_efficiency: 0.01,       // 1% of impacts cause pits
            fatigue_rate: 1e-6,        // Fatigue damage per cycle
            concentration_factor: 2.0,  // Stress concentration
        }
    }
}

impl CavitationDamage {
    /// Create new damage model
    pub fn new(
        grid_shape: (usize, usize, usize),
        material: MaterialProperties,
        params: DamageParameters,
    ) -> Self {
        Self {
            damage_field: Array3::zeros(grid_shape),
            erosion_rate: Array3::zeros(grid_shape),
            impact_pressure: Array3::zeros(grid_shape),
            impact_count: Array3::zeros(grid_shape),
            material,
            params,
        }
    }
    
    /// Update damage from bubble collapse
    pub fn update_damage(
        &mut self,
        bubble_states: &BubbleStateFields,
        liquid_properties: (f64, f64), // (density, sound_speed)
        dt: f64,
    ) {
        let (rho_l, c_l) = liquid_properties;
        
        // Reset instantaneous fields
        self.impact_pressure.fill(0.0);
        
        // Calculate damage at each point
        for ((i, j, k), &is_collapsing) in bubble_states.is_collapsing.indexed_iter() {
            if is_collapsing > 0.5 {
                // Bubble is collapsing - calculate impact
                let r = bubble_states.radius[[i, j, k]];
                let v = bubble_states.velocity[[i, j, k]].abs();
                let p_internal = bubble_states.pressure[[i, j, k]];
                
                // Water hammer pressure from bubble collapse
                let p_hammer = rho_l * c_l * v;
                
                // Jet impact pressure (for asymmetric collapse)
                let p_jet = 0.5 * rho_l * v.powi(2);
                
                // Maximum impact pressure
                let p_impact = p_hammer.max(p_jet).max(p_internal);
                
                if p_impact > self.params.threshold_pressure {
                    self.impact_pressure[[i, j, k]] = p_impact;
                    self.impact_count[[i, j, k]] += 1;
                    
                    // Calculate damage increment
                    let damage_increment = self.calculate_damage_increment(
                        p_impact,
                        r,
                        self.impact_count[[i, j, k]],
                    );
                    
                    self.damage_field[[i, j, k]] += damage_increment * dt;
                    
                    // Calculate erosion rate
                    self.erosion_rate[[i, j, k]] = self.calculate_erosion_rate(
                        p_impact,
                        r,
                        v,
                    );
                }
            }
        }
    }
    
    /// Calculate damage increment from single impact
    fn calculate_damage_increment(
        &self,
        impact_pressure: f64,
        bubble_radius: f64,
        impact_count: u32,
    ) -> f64 {
        // Stress intensity factor
        let stress_intensity = impact_pressure * self.params.concentration_factor;
        
        // Fatigue damage (Miner's rule)
        let cycles_to_failure = (self.material.ultimate_strength / stress_intensity)
            .powf(self.material.fatigue_exponent);
        let fatigue_damage = 1.0 / cycles_to_failure;
        
        // Pit formation probability
        let pit_probability = self.params.pit_efficiency * 
            (impact_pressure / self.material.yield_strength - 1.0).max(0.0);
        
        // Impact area
        let impact_area = PI * bubble_radius.powi(2);
        
        // Total damage increment
        (fatigue_damage + pit_probability) * impact_area * self.params.fatigue_rate
    }
    
    /// Calculate erosion rate from impact
    fn calculate_erosion_rate(
        &self,
        impact_pressure: f64,
        bubble_radius: f64,
        collapse_velocity: f64,
    ) -> f64 {
        if impact_pressure < self.material.hardness {
            return 0.0;
        }
        
        // Erosion model based on impact energy
        let impact_energy = 0.5 * impact_pressure * bubble_radius.powi(3);
        
        // Material removal rate (empirical)
        let removal_efficiency = 1e-9 * self.material.erosion_resistance;
        let volume_removed = removal_efficiency * impact_energy / self.material.hardness;
        
        // Mass erosion rate
        self.material.density * volume_removed * collapse_velocity / bubble_radius
    }
    
    /// Get total accumulated damage
    pub fn total_damage(&self) -> f64 {
        self.damage_field.sum()
    }
    
    /// Get maximum damage location
    pub fn max_damage_location(&self) -> (usize, usize, usize) {
        let mut max_damage = 0.0;
        let mut max_loc = (0, 0, 0);
        
        for ((i, j, k), &damage) in self.damage_field.indexed_iter() {
            if damage > max_damage {
                max_damage = damage;
                max_loc = (i, j, k);
            }
        }
        
        max_loc
    }
    
    /// Calculate mean time to failure based on damage accumulation
    pub fn mean_time_to_failure(&self, dt: f64) -> f64 {
        let max_damage = self.damage_field.iter().cloned().fold(0.0, f64::max);
        
        if max_damage > 0.0 {
            let damage_rate = max_damage / dt;
            1.0 / damage_rate // Time to reach damage = 1
        } else {
            f64::INFINITY
        }
    }
    
    /// Get erosion depth field [m]
    pub fn erosion_depth(&self, time: f64) -> Array3<f64> {
        let mut depth = Array3::zeros(self.erosion_rate.dim());
        
        for ((i, j, k), &rate) in self.erosion_rate.indexed_iter() {
            if rate > 0.0 {
                depth[[i, j, k]] = rate * time / self.material.density;
            }
        }
        
        depth
    }
}

/// Calculate cavitation intensity parameter
pub fn cavitation_intensity(
    bubble_states: &BubbleStateFields,
    liquid_density: f64,
) -> Array3<f64> {
    let shape = bubble_states.radius.dim();
    let mut intensity = Array3::zeros(shape);
    
    for ((i, j, k), &r) in bubble_states.radius.indexed_iter() {
        let v = bubble_states.velocity[[i, j, k]];
        let compression = bubble_states.compression_ratio[[i, j, k]];
        
        // Cavitation intensity based on collapse energy
        let collapse_energy = 0.5 * liquid_density * v.powi(2) * r.powi(3);
        let compression_factor = compression.powf(1.5);
        
        intensity[[i, j, k]] = collapse_energy * compression_factor;
    }
    
    intensity
}

/// Predict erosion patterns
pub struct ErosionPattern;

impl ErosionPattern {
    /// Calculate erosion potential field
    pub fn erosion_potential(
        damage_field: &Array3<f64>,
        flow_velocity: &Array3<f64>,
        surface_normal: &Array3<f64>,
    ) -> Array3<f64> {
        let shape = damage_field.dim();
        let mut potential = Array3::zeros(shape);
        
        for ((i, j, k), &damage) in damage_field.indexed_iter() {
            let v = flow_velocity[[i, j, k]];
            let n = surface_normal[[i, j, k]];
            
            // Erosion enhanced by flow and surface orientation
            let flow_factor = 1.0 + 0.1 * v;
            let angle_factor = n.abs(); // Normal impact most damaging
            
            potential[[i, j, k]] = damage * flow_factor * angle_factor;
        }
        
        potential
    }
    
    /// Identify high-risk areas
    pub fn risk_map(
        damage_field: &Array3<f64>,
        threshold: f64,
    ) -> Array3<bool> {
        damage_field.mapv(|d| d > threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_damage_calculation() {
        let material = MaterialProperties::default();
        let params = DamageParameters::default();
        let mut damage = CavitationDamage::new((10, 10, 10), material, params);
        
        // Create test bubble state
        let mut states = BubbleStateFields::new((10, 10, 10));
        states.is_collapsing[[5, 5, 5]] = 1.0;
        states.radius[[5, 5, 5]] = 1e-6;
        states.velocity[[5, 5, 5]] = -100.0;
        states.pressure[[5, 5, 5]] = 1e9;
        
        // Update damage
        damage.update_damage(&states, (1000.0, 1500.0), 1e-6);
        
        // Check that damage occurred
        assert!(damage.damage_field[[5, 5, 5]] > 0.0);
        assert!(damage.impact_count[[5, 5, 5]] > 0);
    }
    
    #[test]
    fn test_erosion_threshold() {
        let material = MaterialProperties::default();
        let params = DamageParameters::default();
        let damage = CavitationDamage::new((5, 5, 5), material.clone(), params);
        
        // Below threshold pressure - no erosion
        let rate = damage.calculate_erosion_rate(
            material.hardness * 0.5,
            1e-6,
            100.0,
        );
        assert_eq!(rate, 0.0);
        
        // Above threshold - erosion occurs
        let rate = damage.calculate_erosion_rate(
            material.hardness * 2.0,
            1e-6,
            100.0,
        );
        assert!(rate > 0.0);
    }
}
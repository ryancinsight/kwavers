//! Selection criteria for adaptive method choice

use serde::{Deserialize, Serialize};

/// Selection criteria for adaptive method choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    /// Weight for smoothness in decision making (0-1)
    pub smoothness_weight: f64,
    /// Weight for frequency content in decision making (0-1)
    pub frequency_weight: f64,
    /// Weight for material properties in decision making (0-1)
    pub material_weight: f64,
    /// Weight for computational efficiency in decision making (0-1)
    pub efficiency_weight: f64,
    /// Threshold for switching methods (0-1)
    pub switch_threshold: f64,
    /// Hysteresis factor to prevent oscillation (0-1)
    pub hysteresis_factor: f64,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            smoothness_weight: 0.3,
            frequency_weight: 0.25,
            material_weight: 0.25,
            efficiency_weight: 0.2,
            switch_threshold: 0.1,
            hysteresis_factor: 0.05,
        }
    }
}

impl SelectionCriteria {
    /// Validate weights sum to 1.0
    pub fn validate(&self) -> bool {
        let sum = self.smoothness_weight
            + self.frequency_weight
            + self.material_weight
            + self.efficiency_weight;
        
        (sum - 1.0).abs() < 1e-6
    }
    
    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum = self.smoothness_weight
            + self.frequency_weight
            + self.material_weight
            + self.efficiency_weight;
        
        if sum > 0.0 {
            self.smoothness_weight /= sum;
            self.frequency_weight /= sum;
            self.material_weight /= sum;
            self.efficiency_weight /= sum;
        }
    }
    
    /// Compute weighted score
    pub fn compute_score(
        &self,
        smoothness: f64,
        frequency: f64,
        material: f64,
        efficiency: f64,
    ) -> f64 {
        self.smoothness_weight * smoothness
            + self.frequency_weight * frequency
            + self.material_weight * material
            + self.efficiency_weight * efficiency
    }
}
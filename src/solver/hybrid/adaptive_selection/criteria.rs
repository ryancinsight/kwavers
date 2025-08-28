// adaptive_selection/criteria.rs - Selection criteria for method choice

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
            frequency_weight: 0.3,
            material_weight: 0.2,
            efficiency_weight: 0.2,
            switch_threshold: 0.5,
            hysteresis_factor: 0.1,
        }
    }
}

impl SelectionCriteria {
    /// Validate criteria weights sum to 1.0
    pub fn validate(&self) -> Result<(), String> {
        let total = self.smoothness_weight
            + self.frequency_weight
            + self.material_weight
            + self.efficiency_weight;
        
        if (total - 1.0).abs() > 1e-6 {
            return Err(format!("Weights must sum to 1.0, got {}", total));
        }
        
        if self.switch_threshold < 0.0 || self.switch_threshold > 1.0 {
            return Err("Switch threshold must be between 0 and 1".to_string());
        }
        
        if self.hysteresis_factor < 0.0 || self.hysteresis_factor > 0.5 {
            return Err("Hysteresis factor must be between 0 and 0.5".to_string());
        }
        
        Ok(())
    }
}
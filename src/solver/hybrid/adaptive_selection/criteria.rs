// adaptive_selection/criteria.rs - Selection criteria for adaptive method choice

use serde::{Deserialize, Serialize};

/// Criteria for adaptive method selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    /// Weight for field smoothness in selection
    pub smoothness_weight: f64,

    /// Weight for material properties in selection
    pub material_weight: f64,

    /// Weight for computational efficiency
    pub efficiency_weight: f64,

    /// Weight for frequency content
    pub frequency_weight: f64,

    /// Hysteresis factor to prevent oscillation (0.0 to 1.0)
    pub hysteresis_factor: f64,

    /// Discontinuity detection threshold
    pub discontinuity_threshold: f64,

    /// Minimum region size for method switching
    pub min_region_size: usize,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            smoothness_weight: 0.3,
            material_weight: 0.2,
            efficiency_weight: 0.2,
            frequency_weight: 0.3,
            hysteresis_factor: 0.1,
            discontinuity_threshold: 0.1,
            min_region_size: 4,
        }
    }
}

impl SelectionCriteria {
    /// Create criteria optimized for accuracy
    pub fn accuracy_focused() -> Self {
        Self {
            smoothness_weight: 0.4,
            material_weight: 0.3,
            efficiency_weight: 0.1,
            frequency_weight: 0.2,
            hysteresis_factor: 0.05,
            discontinuity_threshold: 0.05,
            min_region_size: 8,
        }
    }

    /// Create criteria optimized for performance
    pub fn performance_focused() -> Self {
        Self {
            smoothness_weight: 0.2,
            material_weight: 0.1,
            efficiency_weight: 0.5,
            frequency_weight: 0.2,
            hysteresis_factor: 0.2,
            discontinuity_threshold: 0.15,
            min_region_size: 2,
        }
    }

    /// Create criteria for heterogeneous media
    pub fn heterogeneous_media() -> Self {
        Self {
            smoothness_weight: 0.2,
            material_weight: 0.4,
            efficiency_weight: 0.2,
            frequency_weight: 0.2,
            hysteresis_factor: 0.15,
            discontinuity_threshold: 0.08,
            min_region_size: 4,
        }
    }

    /// Validate criteria weights sum to 1.0
    pub fn validate(&self) -> bool {
        let total = self.smoothness_weight
            + self.material_weight
            + self.efficiency_weight
            + self.frequency_weight;

        (total - 1.0).abs() < 1e-6
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.smoothness_weight
            + self.material_weight
            + self.efficiency_weight
            + self.frequency_weight;

        if total > 0.0 {
            self.smoothness_weight /= total;
            self.material_weight /= total;
            self.efficiency_weight /= total;
            self.frequency_weight /= total;
        }
    }
}

//! Types for Bjerknes forces calculation

use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE};

/// Configuration for Bjerknes force calculations
#[derive(Debug, Clone, Copy)]
pub struct BjerknesConfig {
    /// Sound speed in medium (m/s)
    pub c0: f64,
    /// Medium density (kg/m³)
    pub rho: f64,
    /// Operating frequency (Hz)
    pub frequency: f64,
    /// Enable primary Bjerknes force
    pub include_primary: bool,
    /// Enable secondary Bjerknes force
    pub include_secondary: bool,
    /// Coalescence threshold distance (m)
    pub coalescence_distance: f64,
    /// Maximum interaction distance (m)
    pub interaction_range: f64,
}

impl Default for BjerknesConfig {
    fn default() -> Self {
        Self {
            c0: SOUND_SPEED_TISSUE,
            rho: DENSITY_WATER_NOMINAL,
            frequency: 1e6,
            include_primary: true,
            include_secondary: true,
            coalescence_distance: 1e-6, // 1 μm
            interaction_range: 100e-6,  // 100 μm
        }
    }
}

/// Results from Bjerknes force calculation
#[derive(Debug, Clone, Copy)]
pub struct BjerknesForce {
    /// Primary Bjerknes force (N) - radiation pressure force
    pub primary: f64,
    /// Secondary Bjerknes force (N) - bubble-bubble interaction
    pub secondary: f64,
    /// Total force (N)
    pub total: f64,
    /// Phase difference between bubbles (radians)
    pub phase_difference: f64,
    /// Interaction type (attractive/repulsive)
    pub interaction_type: InteractionType,
    /// Distance between bubbles (m)
    pub distance: f64,
    /// Whether bubbles will coalesce
    pub coalescing: bool,
}

/// Type of interaction between bubbles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionType {
    /// Attractive interaction (bubbles approach)
    Attractive,
    /// Repulsive interaction (bubbles separate)
    Repulsive,
    /// No significant interaction
    Neutral,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BjerknesConfig::default satisfies the ordering invariant c0 > 0, rho > 0, frequency > 0.
    #[test]
    fn default_config_positive_physical_constants() {
        let cfg = BjerknesConfig::default();
        assert!(cfg.c0 > 0.0, "c0 must be positive");
        assert!(cfg.rho > 0.0, "rho must be positive");
        assert!(cfg.frequency > 0.0, "frequency must be positive");
    }

    /// BjerknesConfig coalescence_distance < interaction_range (physical ordering).
    #[test]
    fn coalescence_distance_less_than_interaction_range() {
        let cfg = BjerknesConfig::default();
        assert!(cfg.coalescence_distance < cfg.interaction_range,
            "coalescence_distance ({}) must be < interaction_range ({})",
            cfg.coalescence_distance, cfg.interaction_range);
    }

    /// InteractionType variants are distinct.
    #[test]
    fn interaction_type_variants_distinct() {
        assert_ne!(InteractionType::Attractive, InteractionType::Repulsive);
        assert_ne!(InteractionType::Repulsive, InteractionType::Neutral);
    }

    /// BjerknesForce stores all fields and Clone works.
    #[test]
    fn bjerknes_force_stores_fields() {
        let f = BjerknesForce {
            primary: 1e-9,
            secondary: -5e-10,
            total: 5e-10,
            phase_difference: 0.3,
            interaction_type: InteractionType::Attractive,
            distance: 50e-6,
            coalescing: false,
        };
        let c = f;
        assert!((c.primary - 1e-9).abs() < 1e-24);
        assert_eq!(c.interaction_type, InteractionType::Attractive);
        assert!(!c.coalescing);
    }
}

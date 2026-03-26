//! Literature value type for kinetics validation
//!
//! References:
//! - Buxton et al. (1988) "Critical review of rate constants for reactions"
//! - Sehested et al. (1991) "Pulse radiolysis of oxygenated aqueous solutions"

/// Literature value for a rate constant with uncertainty
#[derive(Debug, Clone, Copy)]
pub struct LiteratureValue {
    /// Nominal value [M⁻ⁿ·s⁻¹] where n depends on reaction order
    pub nominal: f64,
    /// Minimum reported value (lower bound)
    pub min: f64,
    /// Maximum reported value (upper bound)
    pub max: f64,
    /// Standard deviation / uncertainty
    pub uncertainty: f64,
}

impl LiteratureValue {
    /// Create literature value from nominal ± uncertainty
    pub fn new(nominal: f64, uncertainty: f64) -> Self {
        let percent = uncertainty / nominal;
        Self {
            nominal,
            min: nominal * (1.0 - percent),
            max: nominal * (1.0 + percent),
            uncertainty,
        }
    }

    /// Create from literature range
    pub fn from_range(min: f64, max: f64) -> Self {
        let nominal = (min + max) / 2.0;
        let uncertainty = (max - min) / 2.0;
        Self {
            nominal,
            min,
            max,
            uncertainty,
        }
    }

    /// Check if simulated value is within acceptable range
    pub fn is_within_range(&self, simulated: f64) -> bool {
        simulated >= self.min && simulated <= self.max
    }

    /// Percent difference from nominal
    pub fn percent_difference(&self, simulated: f64) -> f64 {
        100.0 * (simulated - self.nominal).abs() / self.nominal
    }
}

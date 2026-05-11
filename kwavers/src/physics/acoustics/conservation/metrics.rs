//! Conservation validation metrics.

/// Conservation validation metrics for a single timestep.
#[derive(Debug, Clone)]
pub struct ConservationMetrics {
    /// Relative energy error: `|E(t) - E(0)| / E(0)`.
    pub energy_error: f64,
    /// Maximum pointwise mass continuity residual [kg m^-3 s^-1].
    pub mass_error: f64,
    /// Maximum pointwise linearised momentum residual per axis [N m^-3].
    pub momentum_error: (f64, f64, f64),
    /// Volumetric irreversible entropy production rate [W/K].
    pub entropy_production_rate: f64,
    /// True when numerical residuals satisfy tolerances and entropy production is nonnegative.
    pub is_conserved: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ConservationMetrics stores all field values verbatim and Clone is consistent.
    #[test]
    fn conservation_metrics_stores_and_clones_fields() {
        let m = ConservationMetrics {
            energy_error: 1.23,
            mass_error: 4.56,
            momentum_error: (7.0, 8.0, 9.0),
            entropy_production_rate: 10.0,
            is_conserved: true,
        };

        assert!((m.energy_error - 1.23).abs() < 1e-15);
        assert!((m.mass_error - 4.56).abs() < 1e-15);
        assert_eq!(m.momentum_error, (7.0, 8.0, 9.0));
        assert!((m.entropy_production_rate - 10.0).abs() < 1e-15);
        assert!(m.is_conserved);

        let c = m.clone();
        assert!((c.energy_error - m.energy_error).abs() < 1e-15);
        assert!((c.mass_error - m.mass_error).abs() < 1e-15);
        assert_eq!(c.momentum_error, m.momentum_error);
        assert!((c.entropy_production_rate - m.entropy_production_rate).abs() < 1e-15);
        assert_eq!(c.is_conserved, m.is_conserved);
    }

    /// is_conserved = false is stored correctly alongside violated field values.
    #[test]
    fn conservation_metrics_not_conserved_stores_false() {
        let m = ConservationMetrics {
            energy_error: 999.0,
            mass_error: 0.0,
            momentum_error: (0.0, 0.0, 0.0),
            entropy_production_rate: 0.0,
            is_conserved: false,
        };
        assert!(!m.is_conserved);
        assert!(m.energy_error > 1.0, "energy_error must store the large violation value");
    }

    /// Debug output is non-empty (no panic; structural completeness check).
    #[test]
    fn conservation_metrics_debug_non_empty() {
        let m = ConservationMetrics {
            energy_error: 0.0,
            mass_error: 0.0,
            momentum_error: (0.0, 0.0, 0.0),
            entropy_production_rate: 0.0,
            is_conserved: true,
        };
        let s = format!("{m:?}");
        assert!(!s.is_empty(), "Debug output must be non-empty");
    }
}

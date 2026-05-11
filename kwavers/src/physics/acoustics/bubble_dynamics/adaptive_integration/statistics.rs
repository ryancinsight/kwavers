/// Integration statistics for monitoring
#[derive(Debug, Clone)]
pub struct IntegrationStatistics {
    pub total_substeps: usize,
    pub rejected_steps: usize,
    pub min_dt_used: f64,
    pub max_dt_used: f64,
    pub rejection_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// IntegrationStatistics stores all five fields correctly.
    #[test]
    fn stores_fields_correctly() {
        let s = IntegrationStatistics {
            total_substeps: 100,
            rejected_steps: 5,
            min_dt_used: 1e-10,
            max_dt_used: 1e-7,
            rejection_rate: 0.05,
        };
        assert_eq!(s.total_substeps, 100);
        assert_eq!(s.rejected_steps, 5);
        assert!((s.min_dt_used - 1e-10).abs() < 1e-25);
        assert!((s.max_dt_used - 1e-7).abs() < 1e-22);
        assert!((s.rejection_rate - 0.05).abs() < 1e-15);
    }

    /// Clone produces an equal copy.
    #[test]
    fn clone_is_equal() {
        let s = IntegrationStatistics {
            total_substeps: 200,
            rejected_steps: 10,
            min_dt_used: 5e-11,
            max_dt_used: 2e-8,
            rejection_rate: 0.05,
        };
        let c = s.clone();
        assert_eq!(c.total_substeps, 200);
        assert!((c.rejection_rate - s.rejection_rate).abs() < 1e-15);
    }
}

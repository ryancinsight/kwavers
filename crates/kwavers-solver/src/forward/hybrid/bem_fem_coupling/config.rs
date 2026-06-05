/// Configuration for BEM-FEM coupling
#[derive(Debug, Clone)]
pub struct BemFemCouplingConfig {
    /// Coupling interface tolerance
    pub interface_tolerance: f64,
    /// Maximum coupling iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Relaxation factor for iterative coupling
    pub relaxation_factor: f64,
    /// Enable interface smoothing
    pub interface_smoothing: bool,
}

impl Default for BemFemCouplingConfig {
    fn default() -> Self {
        Self {
            interface_tolerance: 1e-6,
            max_iterations: 50,
            convergence_tolerance: 1e-8,
            relaxation_factor: 0.8,
            interface_smoothing: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupling_config_defaults() {
        let config = BemFemCouplingConfig::default();

        assert_eq!(config.max_iterations, 50);
        assert!(config.convergence_tolerance > 0.0);
        assert!(config.relaxation_factor > 0.0 && config.relaxation_factor <= 1.0);
    }
}

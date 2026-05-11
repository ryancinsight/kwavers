#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_em_config_validation() {
        let valid_config = EMConfig::default();
        assert!(GPUEMSolver::validate_config(&valid_config).is_ok());

        let invalid_config = EMConfig {
            grid_size: [0, 64, 64], // Invalid zero dimension
            ..Default::default()
        };
        assert!(GPUEMSolver::validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_cfl_stability() {
        let config = EMConfig {
            spatial_steps: [1e-3, 1e-3, 1e-3],
            time_step: 1e-11, // Too large for stability
            ..Default::default()
        };
        assert!(GPUEMSolver::validate_config(&config).is_err());
    }

    #[test]
    fn test_em_solver_creation() {
        let config = EMConfig::default();
        let solver = GPUEMSolver::new(config);
        let _solver = solver.unwrap();
    }
}

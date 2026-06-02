//! Result and statistics types for core simulation.

/// Simulation result
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub success: bool,
    pub final_step: usize,
    pub total_time: f64,
}

/// Simulation statistics
#[derive(Debug, Clone)]
pub struct CoreSimulationStatistics {
    pub num_sources: usize,
    pub num_sensors: usize,
    pub grid_size: usize,
    pub enabled_features: kwavers_solver::feature::SolverFeatureSet,
}

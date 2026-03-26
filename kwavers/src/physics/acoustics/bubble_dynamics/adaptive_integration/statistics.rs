/// Integration statistics for monitoring
#[derive(Debug, Clone)]
pub struct IntegrationStatistics {
    pub total_substeps: usize,
    pub rejected_steps: usize,
    pub min_dt_used: f64,
    pub max_dt_used: f64,
    pub rejection_rate: f64,
}

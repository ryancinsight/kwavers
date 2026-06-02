use crate::domain::imaging::photoacoustic::OpticalModel;

/// Benchmark descriptor for retained optical solvers.
#[derive(Debug, Clone)]
pub struct OpticalBenchmarkCase {
    pub name: &'static str,
    pub model: OpticalModel,
    pub grid_size: [usize; 3],
}

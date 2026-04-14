/// Benchmark descriptor for reconstruction algorithms.
#[derive(Debug, Clone)]
pub struct ReconstructionBenchmarkCase {
    pub name: &'static str,
    pub geometry: &'static str,
    pub grid_size: [usize; 3],
}

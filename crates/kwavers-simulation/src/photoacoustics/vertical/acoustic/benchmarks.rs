/// Benchmark descriptor for acoustic propagation.
#[derive(Debug, Clone)]
pub struct AcousticBenchmarkCase {
    pub name: &'static str,
    pub grid_size: [usize; 3],
    pub num_time_steps: usize,
}

/// Benchmark descriptor for weighted-average fusion.
#[derive(Debug, Clone)]
pub struct WeightedAverageBenchmarkCase {
    pub name: &'static str,
    pub target_dims: [usize; 3],
    pub modality_count: usize,
}

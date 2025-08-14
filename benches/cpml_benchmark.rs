//! Benchmark comparing C-PML and standard PML performance
//! Currently disabled due to API updates - TODO: Update to match current boundary interface

// Temporarily disabled benchmark - needs API updates
// fn placeholder_benchmark() {
//     // Implementation will be restored after boundary API stabilization
// }

// For now, provide empty benchmark to satisfy cargo check
#[cfg(feature = "benchmark-placeholder")]
mod disabled_benchmark {
    use criterion::{criterion_group, criterion_main, Criterion};
    
    fn placeholder_bench(_c: &mut Criterion) {
        // Placeholder
    }
    
    criterion_group!(benches, placeholder_bench);
    criterion_main!(benches);
}
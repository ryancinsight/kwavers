//! Production readiness integration tests
//!
//! Validates SRS requirements with evidence-based benchmarking

use kwavers_analysis::performance::benchmarks::{run_production_benchmarks, ProductionBenchmarks};

#[test]
fn test_production_benchmark_execution() {
    // Timing lives in criterion benches, never in wallclock asserts: under
    // nextest parallelism and host load, measured durations are nondeterministic.
    let benchmarks = ProductionBenchmarks::new(50, 100); // Smaller for test speed
    let results = benchmarks.run_all();

    // Verify all benchmarks executed
    assert_eq!(results.len(), 3);

    // Verify benchmark structure and value semantics only.
    for result in results {
        assert!(!result.name.is_empty());
        assert!(result.value >= 0.0);
        assert!(result.target > 0.0);
        assert!(!result.unit.is_empty());
    }
}

#[test]
fn test_production_report_generation() {
    let report = run_production_benchmarks();

    // Verify report contains required sections
    assert!(report.contains("Production Performance Benchmark Report"));
    assert!(report.contains("SRS Performance Validation"));
    assert!(report.contains("Evidence-Based Assessment"));

    // Verify report is not empty and contains performance data
    assert!(report.len() > 100);
    assert!(report.contains("PASS") || report.contains("FAIL"));
}

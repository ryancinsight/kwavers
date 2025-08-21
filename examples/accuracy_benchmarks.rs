//! Run numerical accuracy benchmarks
//!
//! This example runs various accuracy benchmarks to validate
//! the numerical methods implementation.

use kwavers::benchmarks::accuracy::{print_results, run_all_benchmarks};

fn main() {
    println!("=== Kwavers Numerical Accuracy Benchmarks ===\n");

    let results = run_all_benchmarks();

    print_results(&results);

    // Summary
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed(0.1)).count();
    let failed = total - passed;

    println!("\n=== Summary ===");
    println!("Total tests: {}", total);
    println!("Passed: {} ({}%)", passed, passed * 100 / total);
    println!("Failed: {} ({}%)", failed, failed * 100 / total);

    if failed == 0 {
        println!("\n✓ All accuracy benchmarks passed!");
    } else {
        println!("\n✗ Some benchmarks failed. Review results above.");
    }
}

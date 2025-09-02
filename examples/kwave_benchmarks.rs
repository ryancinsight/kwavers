//! k-Wave benchmark comparisons example
//!
//! This example runs quantitative benchmarks comparing kwavers against
//! k-Wave MATLAB toolbox reference implementations.

use kwavers::solver::validation::kwave::benchmarks::KWaveBenchmarks;

fn main() {
    // Initialize logging
    kwavers::init_logging();

    println!("=================================================");
    println!("k-Wave Benchmark Comparisons for kwavers");
    println!("=================================================\n");

    // Run all benchmarks
    let results = KWaveBenchmarks::run_all();
    {
            println!("\n=================================================");
            println!("Detailed Results:");
            println!("=================================================");

            for result in &results {
                println!("\nTest: {}", result.test_name);
                println!(
                    "Status: {}",
                    if result.passed {
                        "✓ PASSED"
                    } else {
                        "✗ FAILED"
                    }
                );
                println!("Maximum Error: {:.2}%", result.max_error * 100.0);
                println!("RMS Error: {:.2}%", result.rms_error * 100.0);
            }

            let passed = results.iter().filter(|r| r.passed).count();
            let total = results.len();
            let accuracy = passed as f64 / total as f64 * 100.0;

            println!("\n=================================================");
            println!(
                "Overall Accuracy: {:.1}% ({}/{} tests passed)",
                accuracy, passed, total
            );
            println!("=================================================");

            if passed < total {
                std::process::exit(1);
            }
    }
}

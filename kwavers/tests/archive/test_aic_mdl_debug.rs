use kwavers::analysis::signal_processing::localization::model_order::*;

fn main() {
    // Test case: 1 source with clear gap
    let eigenvalues = vec![10.0, 1.0, 1.0, 1.0];
    let config = ModelOrderConfig::new(4, 100).unwrap();
    let estimator = ModelOrderEstimator::new(config).unwrap();
    let result = estimator.estimate(&eigenvalues).unwrap();
    
    println!("Test 1: Single source [10.0, 1.0, 1.0, 1.0]");
    println!("Estimated sources: {}", result.num_sources);
    println!("Criterion values: {:?}", result.criterion_values);
    println!();
    
    // Test case: 2 sources with clear gap
    let eigenvalues = vec![15.0, 10.0, 1.0, 1.0];
    let result = estimator.estimate(&eigenvalues).unwrap();
    
    println!("Test 2: Two sources [15.0, 10.0, 1.0, 1.0]");
    println!("Estimated sources: {}", result.num_sources);
    println!("Criterion values: {:?}", result.criterion_values);
}

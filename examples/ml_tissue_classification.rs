//! Example demonstrating ML-based tissue classification
//! 
//! This example shows how to use the ML module to classify tissue types
//! based on acoustic properties from simulation data.

use kwavers::{
    ml::{MLEngine, MLBackend, ModelType},
    Grid, HomogeneousMedium,
    KwaversResult,
};
use ndarray::Array3;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("🧠 ML-Based Tissue Classification Example");
    println!("=========================================\n");
    
    // Create simulation grid
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    
    // Create heterogeneous medium with different tissue regions
    let mut medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);
    
    // Simulate different tissue properties in regions
    println!("📊 Setting up tissue regions:");
    println!("  - Region 1: Soft tissue (low density, low speed)");
    println!("  - Region 2: Bone (high density, high speed)");
    println!("  - Region 3: Fat (medium density, low speed)");
    
    // Generate synthetic acoustic field data
    println!("\n🔄 Generating synthetic acoustic data...");
    let acoustic_data = generate_synthetic_acoustic_data(&grid);
    
    // Initialize ML engine
    println!("\n🤖 Initializing ML engine...");
    let mut ml_engine = MLEngine::new(MLBackend::Native)?;
    
    // Attempt to load pre-trained model (will fail with NotImplemented)
    println!("📦 Attempting to load tissue classifier model...");
    match ml_engine.load_model(ModelType::TissueClassifier, "models/tissue_classifier.onnx") {
        Ok(_) => println!("✅ Model loaded successfully"),
        Err(e) => println!("⚠️  Model loading not yet implemented: {}", e),
    }
    
    // Attempt tissue classification
    println!("\n🔬 Attempting tissue classification...");
    match ml_engine.classify_tissue(&acoustic_data) {
        Ok(classification) => {
            println!("✅ Classification complete!");
            analyze_classification_results(&classification);
        }
        Err(e) => {
            println!("⚠️  Classification not yet implemented: {}", e);
            println!("\n📋 Placeholder results:");
            demonstrate_expected_output(&grid);
        }
    }
    
    // Demonstrate parameter optimization
    println!("\n⚙️  Demonstrating parameter optimization...");
    demonstrate_parameter_optimization(&ml_engine)?;
    
    // Demonstrate anomaly detection
    println!("\n🔍 Demonstrating anomaly detection...");
    demonstrate_anomaly_detection(&ml_engine, &acoustic_data)?;
    
    println!("\n✅ ML integration example completed!");
    println!("\n🚀 Next steps for Phase 12:");
    println!("  1. Implement ONNX runtime integration");
    println!("  2. Train tissue classification models");
    println!("  3. Implement real-time inference pipeline");
    println!("  4. Add uncertainty quantification");
    
    Ok(())
}

/// Generate synthetic acoustic data for demonstration
fn generate_synthetic_acoustic_data(grid: &Grid) -> Array3<f64> {
    let mut data = Array3::zeros((grid.nx, grid.ny, grid.nz));
    
    // Create three distinct regions with different acoustic properties
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let x = i as f64 / grid.nx as f64;
                let y = j as f64 / grid.ny as f64;
                let z = k as f64 / grid.nz as f64;
                
                // Region 1: Soft tissue (top-left)
                if x < 0.5 && y < 0.5 {
                    data[[i, j, k]] = 1000.0 + 50.0 * (x + y);
                }
                // Region 2: Bone (top-right)
                else if x >= 0.5 && y < 0.5 {
                    data[[i, j, k]] = 2000.0 + 100.0 * (x - 0.5);
                }
                // Region 3: Fat (bottom)
                else {
                    data[[i, j, k]] = 920.0 + 30.0 * z;
                }
            }
        }
    }
    
    data
}

/// Analyze classification results
fn analyze_classification_results(classification: &Array3<u8>) {
    let mut tissue_counts = [0usize; 4];
    
    for &label in classification.iter() {
        if label < 4 {
            tissue_counts[label as usize] += 1;
        }
    }
    
    println!("\n📊 Classification Results:");
    println!("  - Background: {} voxels", tissue_counts[0]);
    println!("  - Soft tissue: {} voxels", tissue_counts[1]);
    println!("  - Bone: {} voxels", tissue_counts[2]);
    println!("  - Fat: {} voxels", tissue_counts[3]);
}

/// Demonstrate expected output when classification is implemented
fn demonstrate_expected_output(grid: &Grid) {
    let total_voxels = grid.nx * grid.ny * grid.nz;
    let region_size = total_voxels / 4;
    
    println!("  - Expected soft tissue: ~{} voxels", region_size);
    println!("  - Expected bone: ~{} voxels", region_size);
    println!("  - Expected fat: ~{} voxels", region_size * 2);
    println!("\n  Classification accuracy target: >90%");
    println!("  Inference time target: <10ms");
}

/// Demonstrate parameter optimization capabilities
fn demonstrate_parameter_optimization(ml_engine: &MLEngine) -> KwaversResult<()> {
    use std::collections::HashMap;
    
    let mut current_params = HashMap::new();
    current_params.insert("frequency".to_string(), 2.5e6);
    current_params.insert("power".to_string(), 100.0);
    current_params.insert("focus_depth".to_string(), 0.05);
    
    let mut target_metrics = HashMap::new();
    target_metrics.insert("peak_pressure".to_string(), 1e7);
    target_metrics.insert("focal_volume".to_string(), 1e-6);
    target_metrics.insert("heating_rate".to_string(), 10.0);
    
    println!("📊 Current parameters:");
    for (key, value) in &current_params {
        println!("  - {}: {:.2e}", key, value);
    }
    
    println!("\n🎯 Target metrics:");
    for (key, value) in &target_metrics {
        println!("  - {}: {:.2e}", key, value);
    }
    
    match ml_engine.optimize_parameters(&current_params, &target_metrics) {
        Ok(optimized) => {
            println!("\n✅ Optimized parameters:");
            for (key, value) in &optimized {
                println!("  - {}: {:.2e}", key, value);
            }
        }
        Err(e) => {
            println!("\n⚠️  Optimization not yet implemented: {}", e);
            println!("  Expected: 2-5x faster convergence with ML");
        }
    }
    
    Ok(())
}

/// Demonstrate anomaly detection capabilities
fn demonstrate_anomaly_detection(ml_engine: &MLEngine, data: &Array3<f64>) -> KwaversResult<()> {
    println!("🔍 Scanning for anomalies in acoustic field...");
    
    match ml_engine.detect_anomalies(data) {
        Ok(anomalies) => {
            println!("✅ Found {} anomalies:", anomalies.len());
            for (i, anomaly) in anomalies.iter().enumerate() {
                println!("  {}. Location: {:?}, Severity: {:.2}", 
                    i + 1, anomaly.center, anomaly.severity);
            }
        }
        Err(e) => {
            println!("⚠️  Anomaly detection not yet implemented: {}", e);
            println!("  Expected capabilities:");
            println!("  - Cavitation event detection");
            println!("  - Unexpected reflection patterns");
            println!("  - Tissue boundary irregularities");
        }
    }
    
    Ok(())
}
//! Transfer Learning for PINNs Example
//!
//! This example demonstrates the transfer learning capabilities of the PINN framework,
//! showing how pre-trained models can be efficiently adapted to new physics scenarios.

#[cfg(feature = "pinn")]
use burn::backend::{Autodiff, NdArray};
#[cfg(feature = "pinn")]
use kwavers::solver::inverse::pinn::ml::transfer_learning::{
    FreezeStrategy, TransferLearner, TransferLearningConfig,
};
#[cfg(feature = "pinn")]
use kwavers::solver::inverse::pinn::ml::{
    BoundaryCondition2D, BurnPINN2DConfig, BurnPINN2DWave, Geometry2D,
};

#[cfg(feature = "pinn")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    type Backend = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let source_model = BurnPINN2DWave::<Backend>::new(BurnPINN2DConfig::default(), &device)?;

    let transfer_config = TransferLearningConfig {
        fine_tune_lr: 1e-4,
        fine_tune_epochs: 10,
        freeze_strategy: FreezeStrategy::FreezeAllButLast,
        adaptation_strength: 0.0,
        patience: 3,
        wave_speed: 1500.0,
    };

    let mut learner = TransferLearner::new(source_model, transfer_config);

    let target_geometry = Geometry2D::rectangular(0.0, 1.0, 0.0, 1.0);
    let target_conditions: Vec<BoundaryCondition2D> = Vec::new();

    let (_target_model, metrics) =
        learner.transfer_to_geometry(&target_geometry, &target_conditions)?;

    println!("Transfer completed.");
    println!("Initial accuracy: {:.3}", metrics.initial_accuracy);
    println!("Final accuracy: {:.3}", metrics.final_accuracy);
    println!("Convergence epochs: {}", metrics.convergence_epochs);

    Ok(())
}

#[cfg(not(feature = "pinn"))]
fn main() {
    println!("🚫 PINN feature not enabled!");
    println!("   This example requires the 'pinn' feature to be enabled.");
    println!("   Run with: cargo run --example transfer_learning_pinn --features pinn");
    std::process::exit(1);
}
